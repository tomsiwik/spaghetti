"""exp_composition_residual_analysis — measure activation-space composition residual at Gemma 4 E4B.

KCs pre-registered in MATH.md §5:
  K1926 (proxy, structural): tau_final := <||R||_2> / <sum_i ||delta_h_i||_2> over non-pad tokens > 0.10
  K1927 (target, behavioral): max_i |PPL_comp[domain_i] - PPL_adapter_i[domain_i]| / PPL_adapter_i[domain_i] > 0.10

Pipeline:
  Phase A: train 3 LoRA adapters (r=6 q_proj, F#627 recipe) on medical/code/math with DISTINCT per-adapter seeds (addresses F#NEW.c antipattern).
  Phase B: build r=18 LoRALinear model; stack (A_i, B_i) by rank-block to realize sum-of-deltas via one model.
  Phase C: for each of 5 configs {base, ad_med, ad_code, ad_math, composed}, run forward on 3 domain-val splits,
           collect final hidden states + per-token loss; compute residual statistics and per-domain PPL.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, train, default_loss, iterate_batches, CacheDataset
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
DATA_ROOT = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "data"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

R_SINGLE = 6
R_COMPOSED = 18  # 3 * R_SINGLE
SCALE = 6.0
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
ITERS = 100
LR = 1e-4
N_VAL_BATCHES = 15

DOMAINS = ["medical", "code", "math"]
DOMAIN_SEEDS = {"medical": 42, "code": 1337, "math": 2718}

K1926_THRESH = 0.10
K1927_THRESH = 0.10


# ---------------------------------------------------------------------------
# Phase A — train single-domain adapters
# ---------------------------------------------------------------------------

def train_single_domain_adapter(domain: str, seed: int) -> Dict[str, Any]:
    """Train one r=6 LoRA adapter on `domain`. Return adapter path + training stats."""
    print(f"\n=== Phase A: train adapter domain={domain} seed={seed} ===", flush=True)
    t0 = time.time()

    mx.random.seed(seed)

    model, tokenizer = load(BASE_MODEL)
    model.freeze()
    lora_config = {
        "keys": ["self_attn.q_proj"],
        "rank": R_SINGLE,
        "scale": SCALE,
        "dropout": 0.0,
    }
    linear_to_lora_layers(model, num_layers=-1, config=lora_config)
    model.train()

    n_lora = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
    print(f"  n_lora_layers={n_lora}  r={R_SINGLE}  scale={SCALE}", flush=True)

    data_dir = DATA_ROOT / domain
    args = SimpleNamespace(
        data=str(data_dir),
        train=True,
        test=False,
        hf_dataset=False,
        mask_prompt=True,
    )
    train_set_raw, _val_raw, _ = load_dataset(args, tokenizer)
    train_set = CacheDataset(train_set_raw)

    adapter_path = EXP_DIR / f"adapter_{domain}.safetensors"

    losses_recorded: List[float] = []

    class _Cb:
        def on_train_loss_report(self, info):
            losses_recorded.append(float(info.get("train_loss", 0.0)))

        def on_val_loss_report(self, info):
            pass

    optimizer = optim.AdamW(learning_rate=LR)
    train_args = TrainingArgs(
        batch_size=BATCH_SIZE,
        iters=ITERS,
        val_batches=0,
        steps_per_report=10,
        steps_per_eval=ITERS + 1,
        steps_per_save=ITERS + 1,
        max_seq_length=MAX_SEQ_LEN,
        adapter_file=str(adapter_path),
        grad_checkpoint=True,
        grad_accumulation_steps=1,
        clear_cache_threshold=50,
    )

    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=None,
        args=train_args,
        training_callback=_Cb(),
    )

    # Capture trained lora_a / lora_b per layer (copies)
    per_layer: Dict[str, Tuple[mx.array, mx.array]] = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            per_layer[name] = (mx.array(m.lora_a), mx.array(m.lora_b))

    last10 = float(sum(losses_recorded[-10:]) / max(len(losses_recorded[-10:]), 1)) if losses_recorded else -1.0
    elapsed = time.time() - t0
    print(f"  last10_train_loss={last10:.4f}  t={elapsed:.1f}s", flush=True)

    del model, tokenizer, optimizer, train_set, train_set_raw
    gc.collect()
    mx.clear_cache()

    return {
        "domain": domain,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "last10_train_loss": last10,
        "elapsed_sec": elapsed,
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Phase B — build r=18 composed model, populate slots
# ---------------------------------------------------------------------------

def assert_concat_equiv(trained: List[Dict[str, Any]]) -> None:
    """Verify B_stacked.T @ A_stacked.T == sum_i (B_i.T @ A_i.T) on ONE layer as a sanity check."""
    # Take the first layer of the first adapter's naming
    any_layer_name = next(iter(trained[0]["per_layer"].keys()))
    A_list = [t["per_layer"][any_layer_name][0] for t in trained]
    B_list = [t["per_layer"][any_layer_name][1] for t in trained]

    sum_dW = mx.zeros((B_list[0].shape[1], A_list[0].shape[0]))  # (d_out, d_in)
    for A_i, B_i in zip(A_list, B_list):
        sum_dW = sum_dW + (B_i.T @ A_i.T)

    A_stacked = mx.concatenate(A_list, axis=1)  # (d_in, 3r)
    B_stacked = mx.concatenate(B_list, axis=0)  # (3r, d_out)
    comp_dW = B_stacked.T @ A_stacked.T  # (d_out, d_in)

    diff = mx.linalg.norm(sum_dW - comp_dW) / (mx.linalg.norm(sum_dW) + 1e-9)
    mx.eval(diff)
    diff_val = float(diff.item())
    assert diff_val < 1e-5, f"concat-sum equivalence broken: relative diff={diff_val}"
    print(f"  [sanity] concat-sum ΔW equivalence rel-diff = {diff_val:.2e} OK", flush=True)


def build_composed_model() -> Tuple[Any, Any]:
    """Fresh base with LoRALinear r=18 q_proj all-layers."""
    model, tokenizer = load(BASE_MODEL)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=-1,
        config={"keys": ["self_attn.q_proj"], "rank": R_COMPOSED, "scale": SCALE, "dropout": 0.0},
    )
    model.eval()
    return model, tokenizer


def build_stacked_weights(trained: List[Dict[str, Any]]) -> Dict[str, Dict[str, mx.array]]:
    """Return per-layer stacked (A, B) matrices and per-slot submatrices.

    Returns: {layer_name: {"A_stacked", "B_stacked", "A_slots": [A_0, A_1, A_2], "B_slots": [B_0, B_1, B_2]}}
    """
    per_layer: Dict[str, Dict[str, mx.array]] = {}
    layer_names = list(trained[0]["per_layer"].keys())
    for name in layer_names:
        A_slots = [t["per_layer"][name][0] for t in trained]  # each (d_in, r)
        B_slots = [t["per_layer"][name][1] for t in trained]  # each (r, d_out)
        A_stacked = mx.concatenate(A_slots, axis=1)  # (d_in, 3r)
        B_stacked = mx.concatenate(B_slots, axis=0)  # (3r, d_out)
        per_layer[name] = {
            "A_stacked": A_stacked,
            "B_stacked": B_stacked,
            "A_slots": A_slots,
            "B_slots": B_slots,
        }
    return per_layer


def set_config(model, stacked: Dict[str, Dict[str, mx.array]], config_name: str, n_adapters: int = 3) -> None:
    """Overwrite model's LoRALinear (lora_a, lora_b) per config.

    config_name in {"base", "adapter_0", "adapter_1", "adapter_2", "composed"}.
    r=R_COMPOSED = n_adapters * R_SINGLE.
    """
    layer_modules = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    for name, m in layer_modules.items():
        info = stacked[name]
        A_stacked = info["A_stacked"]  # (d_in, 3r)
        B_stacked = info["B_stacked"]  # (3r, d_out)
        d_in, total_r = A_stacked.shape
        _, d_out = B_stacked.shape
        assert total_r == n_adapters * R_SINGLE

        if config_name == "base":
            m.lora_a = mx.zeros((d_in, total_r), dtype=A_stacked.dtype)
            m.lora_b = mx.zeros((total_r, d_out), dtype=B_stacked.dtype)
        elif config_name == "composed":
            m.lora_a = mx.array(A_stacked)
            m.lora_b = mx.array(B_stacked)
        elif config_name.startswith("adapter_"):
            idx = int(config_name.split("_")[1])
            assert 0 <= idx < n_adapters
            # Zero A, B except slot idx
            A_new = mx.zeros((d_in, total_r), dtype=A_stacked.dtype)
            B_new = mx.zeros((total_r, d_out), dtype=B_stacked.dtype)
            lo, hi = idx * R_SINGLE, (idx + 1) * R_SINGLE
            # Use index update for in-place-like assignment
            A_new[:, lo:hi] = info["A_slots"][idx]
            B_new[lo:hi, :] = info["B_slots"][idx]
            m.lora_a = A_new
            m.lora_b = B_new
        else:
            raise ValueError(f"unknown config: {config_name}")


# ---------------------------------------------------------------------------
# Phase C — evaluation: collect hidden states + PPL per (config, domain, batch)
# ---------------------------------------------------------------------------

def collect_eval_batches(tokenizer, domain: str, n_batches: int) -> List[Tuple[mx.array, mx.array]]:
    """Load domain validation set and materialize first n_batches for reuse across configs."""
    data_dir = DATA_ROOT / domain
    args = SimpleNamespace(
        data=str(data_dir),
        train=True,
        test=False,
        hf_dataset=False,
        mask_prompt=True,
    )
    _train, val_raw, _ = load_dataset(args, tokenizer)
    val_set = CacheDataset(val_raw)
    batches: List[Tuple[mx.array, mx.array]] = []
    for batch, lengths in iterate_batches(
        val_set,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LEN,
    ):
        batches.append((mx.array(batch), mx.array(lengths)))
        if len(batches) >= n_batches:
            break
    return batches


def forward_hidden_and_loss(model, batch: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """Compute (final_hidden, per_token_loss_sum, n_valid_tokens) for one batch.

    Returns final-layer hidden state of shape (B, T_in, d) where T_in = T - 1 (inputs drop last tok),
    and scalar total NLL summed over non-pad tokens + ntoks for normalization.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Hidden state pre-lm_head — run the backbone directly
    h = model.language_model.model(inputs)  # (B, T-1, d)

    # Logits via tied / lm_head path (same as model.language_model.__call__ does)
    lm = model.language_model
    if lm.tie_word_embeddings:
        logits = lm.model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    if lm.final_logit_softcapping is not None:
        logits = mx.tanh(logits / lm.final_logit_softcapping) * lm.final_logit_softcapping

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask  # (B, T-1)
    ntoks = mask.sum()
    # Keep sum (not mean) for accurate token-weighted aggregation later
    total_nll = ce.astype(mx.float32).sum()

    return h, total_nll, ntoks, mask


def eval_all_configs(model, stacked, eval_batches_per_domain) -> Dict[str, Any]:
    """For each config × domain × batch: run forward, collect hidden + loss."""
    configs = ["base", "adapter_0", "adapter_1", "adapter_2", "composed"]

    # Per (config, domain): accumulate (sum_nll, total_tokens)
    ppl_acc: Dict[str, Dict[str, Dict[str, float]]] = {c: {d: {"sum_nll": 0.0, "ntoks": 0.0} for d in DOMAINS} for c in configs}

    # Per domain × batch: store 5 hidden states for residual computation
    # Stream: process one batch at a time across all configs (swap weights), immediately compute residual, discard.

    # Aggregated residual stats (final layer, joint over all domains)
    residual_acc = {
        "sum_R_norm": 0.0,
        "sum_sum_delta_norm": 0.0,
        "n_tokens": 0.0,
        # For systematicity: accumulate sum_R_vec (elementwise sum of R across tokens) and sum_R_norm
        "sum_R_vec": None,  # will init on first batch
        "sum_R_entrywise_abs": 0.0,
    }
    residual_per_domain: Dict[str, Dict[str, float]] = {d: {"sum_R_norm": 0.0, "sum_sum_delta_norm": 0.0, "n_tokens": 0.0} for d in DOMAINS}

    total_batches = sum(len(v) for v in eval_batches_per_domain.values())
    print(f"\n=== Phase C: eval 5 configs × {total_batches} batches (≈{total_batches*5} forwards) ===", flush=True)
    t_eval0 = time.time()

    for domain in DOMAINS:
        batches = eval_batches_per_domain[domain]
        print(f"  domain={domain}: {len(batches)} batches", flush=True)
        for bi, (batch, lengths) in enumerate(batches):
            # Forward each config — collect hidden + loss
            hidden_by_config: Dict[str, mx.array] = {}
            for cfg in configs:
                set_config(model, stacked, cfg)
                h, total_nll, ntoks, mask = forward_hidden_and_loss(model, batch, lengths)
                mx.eval(h, total_nll, ntoks, mask)
                hidden_by_config[cfg] = h
                ppl_acc[cfg][domain]["sum_nll"] += float(total_nll.item())
                ppl_acc[cfg][domain]["ntoks"] += float(ntoks.item())

            # Compute residual in activation space
            h_base = hidden_by_config["base"]
            h_0 = hidden_by_config["adapter_0"]
            h_1 = hidden_by_config["adapter_1"]
            h_2 = hidden_by_config["adapter_2"]
            h_comp = hidden_by_config["composed"]

            # δh_i = h_i - h_base
            # R = h_comp - (h_base + Σ δh_i) = h_comp - h_0 - h_1 - h_2 + 2·h_base
            R = h_comp - h_0 - h_1 - h_2 + 2.0 * h_base

            # Per-token L2 norms along last dim → (B, T)
            R_norm = mx.linalg.norm(R, axis=-1)
            d0 = mx.linalg.norm(h_0 - h_base, axis=-1)
            d1 = mx.linalg.norm(h_1 - h_base, axis=-1)
            d2 = mx.linalg.norm(h_2 - h_base, axis=-1)
            sum_delta_norm = d0 + d1 + d2  # (B, T)

            # Mask out padded positions (mask shape (B, T))
            mask_f = mask.astype(mx.float32)

            sum_R = mx.sum(R_norm * mask_f)
            sum_delta = mx.sum(sum_delta_norm * mask_f)
            n_tok = mx.sum(mask_f)

            # For systematicity: elementwise sum of R across tokens (per batch), weighted by mask
            mask_broad = mx.expand_dims(mask_f, axis=-1)  # (B, T, 1)
            R_sum_vec_batch = mx.sum(R * mask_broad, axis=(0, 1))  # (d,)
            R_entrywise_abs_sum = mx.sum(mx.abs(R) * mask_broad)

            mx.eval(sum_R, sum_delta, n_tok, R_sum_vec_batch, R_entrywise_abs_sum)

            residual_acc["sum_R_norm"] += float(sum_R.item())
            residual_acc["sum_sum_delta_norm"] += float(sum_delta.item())
            residual_acc["n_tokens"] += float(n_tok.item())
            if residual_acc["sum_R_vec"] is None:
                residual_acc["sum_R_vec"] = mx.array(R_sum_vec_batch)
            else:
                residual_acc["sum_R_vec"] = residual_acc["sum_R_vec"] + R_sum_vec_batch
            residual_acc["sum_R_entrywise_abs"] += float(R_entrywise_abs_sum.item())

            rpd = residual_per_domain[domain]
            rpd["sum_R_norm"] += float(sum_R.item())
            rpd["sum_sum_delta_norm"] += float(sum_delta.item())
            rpd["n_tokens"] += float(n_tok.item())

            # Free per-batch tensors
            del hidden_by_config, h_base, h_0, h_1, h_2, h_comp, R, R_norm, d0, d1, d2, sum_delta_norm
            if bi % 5 == 4:
                mx.clear_cache()

    eval_elapsed = time.time() - t_eval0

    # Finalize stats
    def _ppl(sum_nll: float, ntoks: float) -> Tuple[float, float]:
        if ntoks <= 0:
            return (float("nan"), float("nan"))
        mean_nll = sum_nll / ntoks
        return (mean_nll, float(math.exp(mean_nll)))

    per_config_per_domain_ppl: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cfg in configs:
        per_config_per_domain_ppl[cfg] = {}
        for d in DOMAINS:
            mean_nll, ppl = _ppl(ppl_acc[cfg][d]["sum_nll"], ppl_acc[cfg][d]["ntoks"])
            per_config_per_domain_ppl[cfg][d] = {
                "nll": mean_nll,
                "ppl": ppl,
                "n_tokens": ppl_acc[cfg][d]["ntoks"],
            }

    tau_final = residual_acc["sum_R_norm"] / max(residual_acc["sum_sum_delta_norm"], 1e-12)

    tau_per_domain: Dict[str, float] = {}
    for d, rpd in residual_per_domain.items():
        tau_per_domain[d] = rpd["sum_R_norm"] / max(rpd["sum_sum_delta_norm"], 1e-12)

    # Systematicity (P3): mean_R_vec over n_tokens, L2-norm ratio to elementwise-mean-|R|
    mean_R_vec = residual_acc["sum_R_vec"] / max(residual_acc["n_tokens"], 1.0)
    mean_R_vec_l2 = float(mx.linalg.norm(mean_R_vec).item())
    # Mean entrywise |R| per token: sum_R_entrywise_abs / (n_tokens * d)
    d = int(mean_R_vec.shape[0])
    mean_abs_R_entry = residual_acc["sum_R_entrywise_abs"] / max(residual_acc["n_tokens"] * d, 1.0)
    # Per-token RMS of mean_R_vec entries
    mean_R_vec_rms = mean_R_vec_l2 / math.sqrt(d)
    systematicity_ratio = mean_R_vec_rms / max(mean_abs_R_entry, 1e-12)

    print(f"  eval elapsed: {eval_elapsed:.1f}s  n_tokens={int(residual_acc['n_tokens'])}", flush=True)

    return {
        "per_config_per_domain_ppl": per_config_per_domain_ppl,
        "residual": {
            "tau_final_layer": tau_final,
            "tau_per_domain": tau_per_domain,
            "systematicity_ratio_mean_over_entrywise": systematicity_ratio,
            "n_eval_tokens": residual_acc["n_tokens"],
            "mean_R_vec_l2": mean_R_vec_l2,
            "mean_abs_R_entry": mean_abs_R_entry,
            "mean_R_vec_rms": mean_R_vec_rms,
        },
        "eval_elapsed_sec": eval_elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Base model: {BASE_MODEL}", flush=True)
    print(f"Domains: {DOMAINS}", flush=True)
    print(f"Seeds: {DOMAIN_SEEDS}", flush=True)
    print(f"Iters/adapter: {ITERS}  r={R_SINGLE}  scale={SCALE}  batch={BATCH_SIZE}  lr={LR}", flush=True)

    t_all = time.time()

    # Phase A: train 3 adapters
    trained: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        res = train_single_domain_adapter(domain, DOMAIN_SEEDS[domain])
        trained.append(res)

    # Sanity check concat-sum equivalence
    print(f"\n=== Sanity: concat-stack equivalence ===", flush=True)
    assert_concat_equiv(trained)

    # Phase B: build composed model (r=18) and stacked weights
    print(f"\n=== Phase B: build r={R_COMPOSED} model, stack weights ===", flush=True)
    stacked = build_stacked_weights(trained)
    model, tokenizer = build_composed_model()

    # Prepare per-domain eval batches (reuse same batches across configs)
    print(f"\n=== Phase C.0: prepare eval batches ===", flush=True)
    eval_batches_per_domain: Dict[str, List[Tuple[mx.array, mx.array]]] = {}
    for domain in DOMAINS:
        eval_batches_per_domain[domain] = collect_eval_batches(tokenizer, domain, N_VAL_BATCHES)
        print(f"  {domain}: {len(eval_batches_per_domain[domain])} batches", flush=True)

    # Phase C: eval all configs
    eval_out = eval_all_configs(model, stacked, eval_batches_per_domain)

    # Kill-criteria evaluation
    tau_final = eval_out["residual"]["tau_final_layer"]
    k1926_result = "pass" if tau_final > K1926_THRESH else "fail"

    ppls = eval_out["per_config_per_domain_ppl"]
    # Behavioral: max_i |PPL_comp[domain_i] - PPL_adapter_i[domain_i]| / PPL_adapter_i[domain_i]
    #   with adapter_0=medical, adapter_1=code, adapter_2=math
    deltas: Dict[str, float] = {}
    for i, domain in enumerate(DOMAINS):
        cfg_i = f"adapter_{i}"
        p_adapter = ppls[cfg_i][domain]["ppl"]
        p_comp = ppls["composed"][domain]["ppl"]
        if p_adapter and p_adapter > 0 and not math.isnan(p_adapter):
            deltas[domain] = abs(p_comp - p_adapter) / p_adapter
        else:
            deltas[domain] = float("nan")
    delta_max = max(v for v in deltas.values() if not math.isnan(v))
    k1927_result = "pass" if delta_max > K1927_THRESH else "fail"

    # Under-training sanity: at least one adapter should meaningfully reduce PPL on its own domain vs base
    base_ppls = {d: ppls["base"][d]["ppl"] for d in DOMAINS}
    adapter_ppls_on_own = {DOMAINS[i]: ppls[f"adapter_{i}"][DOMAINS[i]]["ppl"] for i in range(len(DOMAINS))}
    adapter_lift = {d: (base_ppls[d] - adapter_ppls_on_own[d]) / base_ppls[d] for d in DOMAINS}
    min_lift = min(adapter_lift.values())
    adequately_trained = min_lift > 0.05

    # Verdict per MATH.md §5 (+ adequacy downgrade)
    if k1926_result == "pass" and k1927_result == "pass":
        verdict = "SUPPORTED" if adequately_trained else "PROVISIONAL"
    elif k1926_result == "fail" and k1927_result == "fail":
        verdict = "KILLED"
    else:
        verdict = "PROVISIONAL"

    results = {
        "experiment_id": "exp_composition_residual_analysis",
        "config": {
            "base_model": BASE_MODEL,
            "r_single": R_SINGLE,
            "r_composed": R_COMPOSED,
            "scale": SCALE,
            "targets": ["self_attn.q_proj"],
            "iters": ITERS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "n_val_batches": N_VAL_BATCHES,
            "domains": DOMAINS,
            "domain_seeds": DOMAIN_SEEDS,
            "k1926_thresh": K1926_THRESH,
            "k1927_thresh": K1927_THRESH,
        },
        "per_adapter": [
            {
                "domain": t["domain"],
                "seed": t["seed"],
                "last10_train_loss": t["last10_train_loss"],
                "elapsed_sec": t["elapsed_sec"],
                "adapter_path": t["adapter_path"],
            }
            for t in trained
        ],
        "per_config_per_domain_ppl": ppls,
        "residual": eval_out["residual"],
        "adapter_lift_on_own_domain": adapter_lift,
        "adequately_trained": adequately_trained,
        "behavioral_deltas_per_domain": deltas,
        "behavioral_delta_max": delta_max,
        "kill_criteria": {
            "1926": {
                "text": "Residual term > 10% of individual adapter magnitudes (non-additive)",
                "value": tau_final,
                "thresh": K1926_THRESH,
                "result": k1926_result,
                "type": "proxy_structural",
            },
            "1927": {
                "text": "Residual term is systematic (not noise) — indicates nonlinear interaction",
                "value": delta_max,
                "thresh": K1927_THRESH,
                "result": k1927_result,
                "type": "target_behavioral",
            },
        },
        "verdict": verdict,
        "all_pass": (k1926_result == "pass" and k1927_result == "pass"),
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t_all,
    }

    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote {out_path}", flush=True)
    print(f"Verdict: {verdict}  all_pass={results['all_pass']}  adequately_trained={adequately_trained}", flush=True)
    print(f"K1926 (τ_final): {tau_final:.4f}  (thr={K1926_THRESH}) → {k1926_result}", flush=True)
    print(f"K1927 (Δ_max):  {delta_max:.4f}  (thr={K1927_THRESH}) → {k1927_result}", flush=True)
    print(f"Per-domain τ: {eval_out['residual']['tau_per_domain']}", flush=True)
    print(f"Adapter lift on own domain: {adapter_lift}", flush=True)
    print(f"Elapsed total: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

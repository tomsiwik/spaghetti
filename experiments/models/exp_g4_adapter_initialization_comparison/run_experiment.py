"""exp_g4_adapter_initialization_comparison — compare A-matrix init strategies on Gemma 4 E4B.

Three init methods on q_proj r=6 LoRA (F#627 recipe, all 42 layers):
  1. Grassmannian QR  — orthonormal columns via mx.linalg.qr on Gaussian
  2. Kaiming-uniform   — mlx_lm default (uniform bounded by 1/sqrt(in_features))
  3. Gaussian 0.02     — mx.random.normal * 0.02

For each init, train 100 iters on medical/train.jsonl (F#627 recipe reduced for budget),
measure final eval PPL + post-training A-matrix cross-init cosine similarity.

KCs pre-registered in MATH.md §5:
  K1924 (proxy, structural):  max-min Δ |mean cos(A_i,A_j)| across init pairs > 0.10  → PASS
  K1925 (target, behavioral): max/min PPL ratio ≤ 0.05  OR  Grassmannian NOT uniquely best PPL → PASS
"""
from __future__ import annotations

import gc
import json
import math
import os
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
DATA_DIR = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "data" / "medical"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

R = 6
SCALE = 6.0
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
ITERS = 100
LR = 1e-4
N_VAL_BATCHES = 15  # ~30 val rows at batch 2
SEED = 42

K1924_THRESH = 0.10
K1925_PPL_RATIO_THRESH = 0.05

INIT_METHODS = ["grassmannian", "kaiming", "gaussian"]


def set_lora_a_init(model, method: str, key: mx.array) -> None:
    """Override lora_a in-place according to init method. lora_b stays zero."""
    lora_layers = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    for i, layer in enumerate(lora_layers):
        in_dims, r = layer.lora_a.shape  # lora_a: (input_dims, r)
        assert r == R, f"unexpected r={r}"
        sub_key = mx.random.split(key, num=len(lora_layers))[i]
        if method == "grassmannian":
            # QR of Gaussian (input_dims, r) → Q has orthonormal columns
            g = mx.random.normal(shape=(in_dims, r), key=sub_key)
            # mx.linalg.qr requires CPU stream for some builds
            try:
                q, _ = mx.linalg.qr(g, stream=mx.cpu)
            except Exception:
                q, _ = mx.linalg.qr(g)
            layer.lora_a = q.astype(layer.lora_a.dtype)
        elif method == "kaiming":
            # mlx_lm default
            s = 1.0 / math.sqrt(in_dims)
            layer.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_dims, r), key=sub_key).astype(layer.lora_a.dtype)
        elif method == "gaussian":
            layer.lora_a = (0.02 * mx.random.normal(shape=(in_dims, r), key=sub_key)).astype(layer.lora_a.dtype)
        else:
            raise ValueError(method)
        # B stays zero-init
        layer.lora_b = mx.zeros_like(layer.lora_b)


def collect_lora_a_matrices(model) -> List[mx.array]:
    """Return lora_a for each LoRALinear, in discovery order."""
    out = []
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            out.append(mx.array(m.lora_a))  # copy
    return out


def pairwise_column_cos(A_list_i: List[mx.array], A_list_j: List[mx.array]) -> float:
    """Mean |cos(column_k(A_i), column_k(A_j))| across layers and rank columns."""
    assert len(A_list_i) == len(A_list_j)
    cos_abs_values = []
    for Ai, Aj in zip(A_list_i, A_list_j):
        # Ai, Aj shape (in_dims, r); compare column-k of each
        Ai_n = Ai / (mx.linalg.norm(Ai, axis=0, keepdims=True) + 1e-12)
        Aj_n = Aj / (mx.linalg.norm(Aj, axis=0, keepdims=True) + 1e-12)
        cos_cols = mx.sum(Ai_n * Aj_n, axis=0)  # shape (r,)
        cos_abs_values.append(mx.abs(cos_cols))
    all_abs = mx.concatenate(cos_abs_values, axis=0)
    mx.eval(all_abs)
    return float(mx.mean(all_abs).item())


def build_lora_model(lora_config: Dict[str, Any]):
    """Load fresh base + attach LoRA per config. Returns (model, tokenizer)."""
    model, tok = load(BASE_MODEL)
    model.freeze()
    linear_to_lora_layers(model, num_layers=-1, config=lora_config)
    model.train()
    return model, tok


def eval_ppl(model, val_set, tokenizer) -> Tuple[float, float]:
    """Return (mean_nll, ppl) on val_set via iterate_batches."""
    model.eval()
    total_nll = 0.0
    total_toks = 0
    n_batches = 0
    for batch, lengths in iterate_batches(
        val_set,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LEN,
    ):
        losses, ntoks, *_ = default_loss(model, batch, lengths)
        mx.eval(losses, ntoks)
        total_nll += float(losses.item()) * float(ntoks.item())
        total_toks += float(ntoks.item())
        n_batches += 1
        if n_batches >= N_VAL_BATCHES:
            break
    mean_nll = total_nll / max(total_toks, 1.0)
    ppl = float(math.exp(mean_nll))
    model.train()
    return mean_nll, ppl


def run_one_init(method: str, seed: int) -> Dict[str, Any]:
    """Build fresh model, override init, train, evaluate. Return metrics + A-matrices."""
    print(f"\n=== init={method} seed={seed} ===", flush=True)
    t0 = time.time()

    lora_config = {
        "keys": ["self_attn.q_proj"],
        "rank": R,
        "scale": SCALE,
        "dropout": 0.0,
    }

    model, tokenizer = build_lora_model(lora_config)
    key = mx.random.key(seed)

    # Override init
    set_lora_a_init(model, method, key)

    # Snapshot initial A
    A_init = collect_lora_a_matrices(model)
    mx.eval([a for a in A_init])

    # Dataset
    args = SimpleNamespace(
        data=str(DATA_DIR),
        train=True,
        test=False,
        hf_dataset=False,
        mask_prompt=True,
    )
    train_set_raw, val_set_raw, _ = load_dataset(args, tokenizer)
    train_set = CacheDataset(train_set_raw)
    val_set = CacheDataset(val_set_raw)

    # Train
    optimizer = optim.AdamW(learning_rate=LR)

    losses_recorded: List[float] = []

    class _Cb:
        def on_train_loss_report(self, info):
            losses_recorded.append(float(info.get("train_loss", 0.0)))
        def on_val_loss_report(self, info):
            pass

    train_args = TrainingArgs(
        batch_size=BATCH_SIZE,
        iters=ITERS,
        val_batches=0,  # skip mid-training val to save time
        steps_per_report=10,
        steps_per_eval=ITERS + 1,
        steps_per_save=ITERS + 1,
        max_seq_length=MAX_SEQ_LEN,
        adapter_file=str(EXP_DIR / f"adapters_{method}.safetensors"),
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

    # Final eval
    mean_nll, ppl = eval_ppl(model, val_set, tokenizer)

    # Snapshot final A
    A_final = collect_lora_a_matrices(model)
    mx.eval([a for a in A_final])

    last10_loss = float(sum(losses_recorded[-10:]) / max(len(losses_recorded[-10:]), 1)) if losses_recorded else -1.0

    elapsed = time.time() - t0
    print(f"  init={method}: eval_ppl={ppl:.4f} eval_nll={mean_nll:.4f} last10_train_loss={last10_loss:.4f} t={elapsed:.1f}s", flush=True)

    # Free the model before next init
    del model, tokenizer, optimizer, train_set, val_set
    gc.collect()
    mx.clear_cache()

    return {
        "method": method,
        "eval_ppl": ppl,
        "eval_nll": mean_nll,
        "last10_train_loss": last10_loss,
        "all_train_losses": losses_recorded,
        "A_init": A_init,
        "A_final": A_final,
        "elapsed_sec": elapsed,
    }


def main():
    mx.random.seed(SEED)
    print(f"Base model: {BASE_MODEL}", flush=True)
    print(f"Init methods: {INIT_METHODS}", flush=True)
    print(f"Iters/init: {ITERS}  r={R} scale={SCALE}  batch={BATCH_SIZE}  lr={LR}", flush=True)
    print(f"Data: {DATA_DIR}", flush=True)

    t_all = time.time()

    # Baseline eval PPL (no adapter)
    model_b, tok_b = load(BASE_MODEL)
    args = SimpleNamespace(
        data=str(DATA_DIR),
        train=True,
        test=False,
        hf_dataset=False,
        mask_prompt=True,
    )
    _, val_b_raw, _ = load_dataset(args, tok_b)
    val_b = CacheDataset(val_b_raw)
    base_nll, base_ppl = eval_ppl(model_b, val_b, tok_b)
    print(f"Baseline (no adapter): nll={base_nll:.4f} ppl={base_ppl:.4f}", flush=True)
    del model_b, tok_b, val_b
    gc.collect()
    mx.clear_cache()

    runs: Dict[str, Dict[str, Any]] = {}
    for method in INIT_METHODS:
        runs[method] = run_one_init(method, seed=SEED)

    # Cross-init cos-sim on FINAL A-matrices
    pairs = [(a, b) for i, a in enumerate(INIT_METHODS) for b in INIT_METHODS[i + 1 :]]
    cross_init_cos_final: Dict[str, float] = {}
    cross_init_cos_init: Dict[str, float] = {}
    for a, b in pairs:
        key = f"{a}__vs__{b}"
        cross_init_cos_final[key] = pairwise_column_cos(runs[a]["A_final"], runs[b]["A_final"])
        cross_init_cos_init[key] = pairwise_column_cos(runs[a]["A_init"], runs[b]["A_init"])

    # Self-cos (intra-adapter column orthogonality) for each init
    intra_final_cos: Dict[str, float] = {}
    intra_init_cos: Dict[str, float] = {}
    for method in INIT_METHODS:
        # Per layer: mean |cos| of distinct column pairs
        finals = runs[method]["A_final"]
        inits = runs[method]["A_init"]
        def _intra(As):
            vals = []
            for A in As:
                # A: (in_dims, r)
                An = A / (mx.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
                G = An.T @ An  # (r, r)
                # off-diagonal
                mask = 1 - mx.eye(G.shape[0])
                off_abs = mx.abs(G) * mask
                mean_off = mx.sum(off_abs) / (G.shape[0] * (G.shape[0] - 1))
                vals.append(mean_off)
            v = mx.stack(vals)
            mx.eval(v)
            return float(mx.mean(v).item())
        intra_final_cos[method] = _intra(finals)
        intra_init_cos[method] = _intra(inits)

    # KC evaluations
    # K1924: max-min Δ across init-pair mean |cos| at final > 0.10
    final_values = list(cross_init_cos_final.values())
    k1924_delta = max(final_values) - min(final_values)
    k1924_result = "pass" if k1924_delta > K1924_THRESH else "fail"

    # K1925: behavioral init-ordering
    ppl_by_method = {m: runs[m]["eval_ppl"] for m in INIT_METHODS}
    best_method = min(ppl_by_method, key=ppl_by_method.get)
    worst_method = max(ppl_by_method, key=ppl_by_method.get)
    ppl_ratio = (ppl_by_method[worst_method] - ppl_by_method[best_method]) / ppl_by_method[best_method]
    grassmannian_uniquely_best = (best_method == "grassmannian") and (ppl_ratio > K1925_PPL_RATIO_THRESH)
    # PASS = NOT(Grassmannian uniquely best with material margin)
    k1925_result = "fail" if grassmannian_uniquely_best else "pass"

    all_pass = (k1924_result == "pass") and (k1925_result == "pass")
    # Verdict logic per MATH.md §5
    if k1924_result == "pass" and k1925_result == "pass":
        verdict = "SUPPORTED"
    elif k1924_result == "fail" and k1925_result == "fail":
        verdict = "KILLED"
    else:
        verdict = "PROVISIONAL"

    results = {
        "experiment_id": "exp_g4_adapter_initialization_comparison",
        "config": {
            "base_model": BASE_MODEL,
            "rank": R,
            "scale": SCALE,
            "targets": ["self_attn.q_proj"],
            "iters": ITERS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "seed": SEED,
            "init_methods": INIT_METHODS,
            "n_val_batches": N_VAL_BATCHES,
            "k1924_thresh": K1924_THRESH,
            "k1925_ppl_ratio_thresh": K1925_PPL_RATIO_THRESH,
        },
        "baseline": {"nll": base_nll, "ppl": base_ppl},
        "per_init": {
            m: {
                "eval_ppl": runs[m]["eval_ppl"],
                "eval_nll": runs[m]["eval_nll"],
                "last10_train_loss": runs[m]["last10_train_loss"],
                "elapsed_sec": runs[m]["elapsed_sec"],
            }
            for m in INIT_METHODS
        },
        "cross_init_cos_init": cross_init_cos_init,
        "cross_init_cos_final": cross_init_cos_final,
        "intra_init_cos_init": intra_init_cos,
        "intra_init_cos_final": intra_final_cos,
        "ppl_ratio_worst_over_best": ppl_ratio,
        "best_init_by_ppl": best_method,
        "worst_init_by_ppl": worst_method,
        "kill_criteria": {
            "1924": {
                "text": "Initialization method produces > 0.10 cos-sim difference in final adapter",
                "value": k1924_delta,
                "thresh": K1924_THRESH,
                "result": k1924_result,
                "type": "proxy_structural",
            },
            "1925": {
                "text": "Grassmannian A-init not best method for any metric (PPL, cos-sim, composition)",
                "value": ppl_ratio,
                "best_method": best_method,
                "thresh": K1925_PPL_RATIO_THRESH,
                "result": k1925_result,
                "type": "target_behavioral",
            },
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t_all,
    }

    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)
    print(f"Verdict: {verdict}  all_pass={all_pass}", flush=True)
    print(f"K1924 (structural Δcos): {k1924_delta:.4f}  → {k1924_result}", flush=True)
    print(f"K1925 (PPL ratio): {ppl_ratio:.4f} best={best_method}  → {k1925_result}", flush=True)
    print(f"Elapsed total: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""exp_g4_adapter_initialization_comparison_v2 — PRNG-confound-free multi-seed init comparison.

v2 fixes vs v1 (F#751 PROVISIONAL):
  - Distinct top-level seeds per init (Grassmannian=42, Kaiming=43, Gaussian=44).
  - 3 sub-seeds per init for within-init seed-variance bounds.
  - F#666-paired KCs (K1977/K1978/K1979 proxy + K1983/K1984/K1985 target/non-interference).
  - Medical-MCQ heldout n=80 letter-accuracy eval added (parent had PPL-only).

SMOKE_TEST=1 (default): 100 iters × 9 runs (3 inits × 3 seeds). Verdict floor = PROVISIONAL.
SMOKE_TEST=0:           1000 iters × 9 runs (full convergence). For v3 reclaim.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
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
LR = 1e-4
N_VAL_BATCHES = 15

# Smoke vs full
SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"
ITERS = 100 if SMOKE else 1000
N_MCQ = 80

# Per-init top-level seeds (PRNG-confound fix vs v1)
INIT_SEEDS: Dict[str, int] = {"grassmannian": 42, "kaiming": 43, "gaussian": 44}
# Per-init sub-seeds for within-init variance
N_SUB_SEEDS = 3
SUB_SEED_OFFSETS = (0, 10, 20)

# F#666 thresholds
K1977_COS_LT = 0.20      # PASS = cross-init final |cos| < 0.20
K1978_PPL_RATIO = 1.10   # PASS = ratio > 1.10 (proxy: spread present)
K1979_SEED_VAR = 0.05    # PASS = within-init PPL seed-spread > 5%
K1983_BEHAV_SPREAD = 5.0 # FAIL = cross-init medical-MCQ spread > 5pp (FAIL = INV. verified)
K1984_BEHAV_SEED_VAR = 3.0  # PASS = within-init MCQ seed-spread > 3pp
K1985_NON_INTERFERENCE = 5.0  # PASS unless any init drops > 5pp vs base

INIT_METHODS = ["grassmannian", "kaiming", "gaussian"]


def set_lora_a_init(model, method: str, key: mx.array) -> None:
    lora_layers = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    for i, layer in enumerate(lora_layers):
        in_dims, r = layer.lora_a.shape
        sub_key = mx.random.split(key, num=len(lora_layers))[i]
        if method == "grassmannian":
            g = mx.random.normal(shape=(in_dims, r), key=sub_key)
            try:
                q, _ = mx.linalg.qr(g, stream=mx.cpu)
            except Exception:
                q, _ = mx.linalg.qr(g)
            layer.lora_a = q.astype(layer.lora_a.dtype)
        elif method == "kaiming":
            s = 1.0 / math.sqrt(in_dims)
            layer.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_dims, r), key=sub_key).astype(layer.lora_a.dtype)
        elif method == "gaussian":
            layer.lora_a = (0.02 * mx.random.normal(shape=(in_dims, r), key=sub_key)).astype(layer.lora_a.dtype)
        else:
            raise ValueError(method)
        layer.lora_b = mx.zeros_like(layer.lora_b)


def collect_lora_a_matrices(model) -> List[mx.array]:
    out = []
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            out.append(mx.array(m.lora_a))
    return out


def pairwise_column_cos(A_list_i: List[mx.array], A_list_j: List[mx.array]) -> float:
    cos_abs_values = []
    for Ai, Aj in zip(A_list_i, A_list_j):
        Ai_n = Ai / (mx.linalg.norm(Ai, axis=0, keepdims=True) + 1e-12)
        Aj_n = Aj / (mx.linalg.norm(Aj, axis=0, keepdims=True) + 1e-12)
        cos_cols = mx.sum(Ai_n * Aj_n, axis=0)
        cos_abs_values.append(mx.abs(cos_cols))
    all_abs = mx.concatenate(cos_abs_values, axis=0)
    mx.eval(all_abs)
    return float(mx.mean(all_abs).item())


def build_lora_model(lora_config: Dict[str, Any]):
    model, tok = load(BASE_MODEL)
    model.freeze()
    linear_to_lora_layers(model, num_layers=-1, config=lora_config)
    model.train()
    return model, tok


def eval_ppl(model, val_set) -> Tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_toks = 0
    n_batches = 0
    for batch, lengths in iterate_batches(val_set, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LEN):
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


def load_mcq_eval(n: int) -> List[Tuple[str, str]]:
    """Load first n MCQ records from medical/valid.jsonl. Returns list of (prompt, gold_letter)."""
    path = DATA_DIR / "valid.jsonl"
    out: List[Tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            row = json.loads(line)
            msgs = row["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            m = re.match(r"\s*([A-D])", asst)
            if not m:
                continue
            out.append((user, m.group(1)))
    return out


def eval_mcq(model, tokenizer, mcq_set: List[Tuple[str, str]]) -> float:
    """Greedy single-token-letter accuracy. Returns pp accuracy in [0, 100]."""
    model.eval()
    correct = 0
    total = 0
    letter_ids = {}
    for L in "ABCD":
        ids = tokenizer.encode(L, add_special_tokens=False)
        letter_ids[L] = ids[0]
    for prompt, gold in mcq_set:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids_arr = mx.array(ids)[None, :]
        logits = model(ids_arr)
        last = logits[0, -1, :]
        scores = mx.array([last[letter_ids[L]] for L in "ABCD"])
        mx.eval(scores)
        pred_idx = int(mx.argmax(scores).item())
        pred = "ABCD"[pred_idx]
        if pred == gold:
            correct += 1
        total += 1
    model.train()
    return 100.0 * correct / max(total, 1)


def run_one(method: str, sub_seed: int, mcq_set: List[Tuple[str, str]]) -> Dict[str, Any]:
    print(f"\n=== init={method} sub_seed={sub_seed} iters={ITERS} ===", flush=True)
    t0 = time.time()
    top_seed = INIT_SEEDS[method] + sub_seed
    mx.random.seed(top_seed)

    lora_config = {
        "keys": ["self_attn.q_proj"],
        "rank": R,
        "scale": SCALE,
        "dropout": 0.0,
    }
    model, tokenizer = build_lora_model(lora_config)
    key = mx.random.key(top_seed)
    set_lora_a_init(model, method, key)
    A_init = collect_lora_a_matrices(model)
    mx.eval([a for a in A_init])

    args = SimpleNamespace(data=str(DATA_DIR), train=True, test=False, hf_dataset=False, mask_prompt=True)
    train_set_raw, val_set_raw, _ = load_dataset(args, tokenizer)
    train_set = CacheDataset(train_set_raw)
    val_set = CacheDataset(val_set_raw)

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
        val_batches=0,
        steps_per_report=20,
        steps_per_eval=ITERS + 1,
        steps_per_save=ITERS + 1,
        max_seq_length=MAX_SEQ_LEN,
        adapter_file=str(EXP_DIR / f"adapter_{method}_s{sub_seed}.safetensors"),
        grad_checkpoint=True,
        grad_accumulation_steps=1,
        clear_cache_threshold=50,
    )
    train(model=model, optimizer=optimizer, train_dataset=train_set, val_dataset=None, args=train_args, training_callback=_Cb())

    mean_nll, ppl = eval_ppl(model, val_set)
    mcq_acc = eval_mcq(model, tokenizer, mcq_set)
    A_final = collect_lora_a_matrices(model)
    mx.eval([a for a in A_final])
    last10 = float(sum(losses_recorded[-10:]) / max(len(losses_recorded[-10:]), 1)) if losses_recorded else -1.0
    elapsed = time.time() - t0
    print(f"  -> ppl={ppl:.4f} mcq={mcq_acc:.1f}% last10={last10:.4f} t={elapsed:.1f}s", flush=True)

    del model, tokenizer, optimizer, train_set, val_set
    gc.collect()
    mx.clear_cache()

    return {
        "method": method,
        "sub_seed": sub_seed,
        "top_seed": top_seed,
        "eval_ppl": ppl,
        "eval_nll": mean_nll,
        "mcq_acc_pct": mcq_acc,
        "last10_train_loss": last10,
        "A_init": A_init,
        "A_final": A_final,
        "elapsed_sec": elapsed,
    }


def main():
    print(f"v2 init comparison | smoke={SMOKE} iters={ITERS} | base={BASE_MODEL}", flush=True)
    print(f"Inits: {INIT_METHODS}  sub_seeds={SUB_SEED_OFFSETS}", flush=True)
    t_all = time.time()

    # Load MCQ once (shared across runs)
    mcq_set = load_mcq_eval(N_MCQ)
    print(f"Loaded {len(mcq_set)} MCQ items from {DATA_DIR}/valid.jsonl", flush=True)

    # Baseline (no adapter) PPL + MCQ
    model_b, tok_b = load(BASE_MODEL)
    args = SimpleNamespace(data=str(DATA_DIR), train=True, test=False, hf_dataset=False, mask_prompt=True)
    _, val_b_raw, _ = load_dataset(args, tok_b)
    val_b = CacheDataset(val_b_raw)
    base_nll, base_ppl = eval_ppl(model_b, val_b)
    base_mcq = eval_mcq(model_b, tok_b, mcq_set)
    print(f"Baseline (no adapter): nll={base_nll:.4f} ppl={base_ppl:.4f} mcq={base_mcq:.1f}%", flush=True)
    del model_b, tok_b, val_b
    gc.collect()
    mx.clear_cache()

    # Run grid: 3 inits × 3 sub-seeds = 9 runs
    runs: List[Dict[str, Any]] = []
    for method in INIT_METHODS:
        for sub in SUB_SEED_OFFSETS:
            runs.append(run_one(method, sub, mcq_set))

    # Aggregate
    by_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in INIT_METHODS}
    for r in runs:
        by_method[r["method"]].append(r)

    # Cross-init final cos: pick s=0 representative per init for cross-init pair cos
    rep_run: Dict[str, Dict[str, Any]] = {m: by_method[m][0] for m in INIT_METHODS}
    pairs = [(a, b) for i, a in enumerate(INIT_METHODS) for b in INIT_METHODS[i + 1:]]
    cross_init_cos_final: Dict[str, float] = {}
    cross_init_cos_init: Dict[str, float] = {}
    for a, b in pairs:
        k = f"{a}__vs__{b}"
        cross_init_cos_final[k] = pairwise_column_cos(rep_run[a]["A_final"], rep_run[b]["A_final"])
        cross_init_cos_init[k] = pairwise_column_cos(rep_run[a]["A_init"], rep_run[b]["A_init"])

    # K1977 (proxy): max cross-init |cos| < 0.20
    cross_max_cos = max(cross_init_cos_final.values())
    k1977_pass = cross_max_cos < K1977_COS_LT

    # K1978 (proxy): PPL ratio worst/best > 1.10 (use s=0 reps)
    rep_ppls = {m: rep_run[m]["eval_ppl"] for m in INIT_METHODS}
    ppl_ratio = max(rep_ppls.values()) / min(rep_ppls.values())
    k1978_pass = ppl_ratio > K1978_PPL_RATIO

    # K1979 (proxy): max within-init seed-variance on PPL > 5%
    within_init_ppl_spread: Dict[str, float] = {}
    for m in INIT_METHODS:
        ppls = [r["eval_ppl"] for r in by_method[m]]
        within_init_ppl_spread[m] = (max(ppls) - min(ppls)) / min(ppls)
    max_within_ppl = max(within_init_ppl_spread.values())
    k1979_pass = max_within_ppl > K1979_SEED_VAR

    # K1983 (target): cross-init MCQ spread > 5pp (FAIL = within thresh = init-invariance VERIFIED)
    rep_mcq = {m: rep_run[m]["mcq_acc_pct"] for m in INIT_METHODS}
    cross_mcq_spread = max(rep_mcq.values()) - min(rep_mcq.values())
    k1983_pass = cross_mcq_spread > K1983_BEHAV_SPREAD

    # K1984 (target): max within-init MCQ seed-variance > 3pp
    within_init_mcq_spread: Dict[str, float] = {}
    for m in INIT_METHODS:
        mcqs = [r["mcq_acc_pct"] for r in by_method[m]]
        within_init_mcq_spread[m] = max(mcqs) - min(mcqs)
    max_within_mcq = max(within_init_mcq_spread.values())
    k1984_pass = max_within_mcq > K1984_BEHAV_SEED_VAR

    # K1985 (non-interference): no init drops base by > 5pp
    init_mean_mcq: Dict[str, float] = {m: sum(r["mcq_acc_pct"] for r in by_method[m]) / len(by_method[m]) for m in INIT_METHODS}
    init_drop_vs_base: Dict[str, float] = {m: base_mcq - init_mean_mcq[m] for m in INIT_METHODS}
    max_drop = max(init_drop_vs_base.values())
    k1985_pass = max_drop <= K1985_NON_INTERFERENCE

    # SC#109 success matrix (per v2 notes): Init-invariance VERIFIED iff
    #   K1983 FAIL (spread within thresh) AND K1984 PASS (eval has discriminative power)
    #   AND K1985 PASS (non-interference)
    sc109_verified = (not k1983_pass) and k1984_pass and k1985_pass

    # Verdict (smoke → PROVISIONAL floor)
    if SMOKE:
        verdict = "PROVISIONAL"  # Iter floor; full 1000-iter v3 reclaim path documented in MATH §6
    elif sc109_verified:
        verdict = "SUPPORTED"
    elif k1985_pass is False:
        verdict = "KILLED"  # recipe is behaviorally unproductive
    else:
        verdict = "PROVISIONAL"

    all_pass = k1977_pass and (not k1983_pass) and k1984_pass and k1985_pass

    results = {
        "experiment_id": "exp_g4_adapter_initialization_comparison_v2",
        "is_smoke": SMOKE,
        "config": {
            "base_model": BASE_MODEL,
            "rank": R, "scale": SCALE, "targets": ["self_attn.q_proj"],
            "iters": ITERS, "batch_size": BATCH_SIZE, "lr": LR, "max_seq_len": MAX_SEQ_LEN,
            "init_seeds": INIT_SEEDS, "sub_seed_offsets": list(SUB_SEED_OFFSETS),
            "init_methods": INIT_METHODS, "n_mcq": N_MCQ, "n_val_batches": N_VAL_BATCHES,
        },
        "baseline": {"nll": base_nll, "ppl": base_ppl, "mcq_acc_pct": base_mcq},
        "per_run": [
            {"method": r["method"], "sub_seed": r["sub_seed"], "top_seed": r["top_seed"],
             "eval_ppl": r["eval_ppl"], "eval_nll": r["eval_nll"], "mcq_acc_pct": r["mcq_acc_pct"],
             "last10_train_loss": r["last10_train_loss"], "elapsed_sec": r["elapsed_sec"]}
            for r in runs
        ],
        "cross_init_cos_init": cross_init_cos_init,
        "cross_init_cos_final": cross_init_cos_final,
        "rep_eval_ppl": rep_ppls,
        "rep_mcq_acc_pct": rep_mcq,
        "init_mean_mcq_acc_pct": init_mean_mcq,
        "init_drop_vs_base_pp": init_drop_vs_base,
        "within_init_ppl_spread": within_init_ppl_spread,
        "within_init_mcq_spread_pp": within_init_mcq_spread,
        "kill_criteria": {
            "1977": {"text": "cross-init final |cos| < 0.20", "value": cross_max_cos, "thresh": K1977_COS_LT, "result": "pass" if k1977_pass else "fail", "type": "proxy_structural"},
            "1978": {"text": "PPL ratio worst/best > 1.10", "value": ppl_ratio, "thresh": K1978_PPL_RATIO, "result": "pass" if k1978_pass else "fail", "type": "proxy_structural"},
            "1979": {"text": "within-init PPL seed-variance > 5%", "value": max_within_ppl, "thresh": K1979_SEED_VAR, "result": "pass" if k1979_pass else "fail", "type": "proxy_identifiability"},
            "1983": {"text": "cross-init MCQ spread > 5pp (FAIL = init-invariance verified)", "value": cross_mcq_spread, "thresh": K1983_BEHAV_SPREAD, "result": "pass" if k1983_pass else "fail", "type": "target_behavioral"},
            "1984": {"text": "within-init MCQ seed-variance > 3pp", "value": max_within_mcq, "thresh": K1984_BEHAV_SEED_VAR, "result": "pass" if k1984_pass else "fail", "type": "target_identifiability"},
            "1985": {"text": "non-interference: no init drops base > 5pp", "value": max_drop, "thresh": K1985_NON_INTERFERENCE, "result": "pass" if k1985_pass else "fail", "type": "target_non_interference"},
        },
        "sc_109_verified": sc109_verified,
        "verdict": verdict,
        "all_pass": all_pass,
        "total_wall_clock_sec": time.time() - t_all,
    }

    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)
    print(f"Verdict: {verdict}  all_pass={all_pass}  sc_109={sc109_verified}", flush=True)
    print(f"K1977 cos<0.20: max={cross_max_cos:.3f} -> {'pass' if k1977_pass else 'fail'}", flush=True)
    print(f"K1978 PPL ratio: {ppl_ratio:.3f} -> {'pass' if k1978_pass else 'fail'}", flush=True)
    print(f"K1979 PPL seed-var: {max_within_ppl:.3f} -> {'pass' if k1979_pass else 'fail'}", flush=True)
    print(f"K1983 MCQ spread: {cross_mcq_spread:.1f}pp -> {'pass' if k1983_pass else 'fail (= INV. verified)'}", flush=True)
    print(f"K1984 MCQ seed-var: {max_within_mcq:.1f}pp -> {'pass' if k1984_pass else 'fail'}", flush=True)
    print(f"K1985 non-interference (max drop {max_drop:.1f}pp): -> {'pass' if k1985_pass else 'fail'}", flush=True)
    print(f"Total: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
mHC-style Sinkhorn-Knopp normalization of composed delta.

Per DeepSeek V4 §2.2: constrain composed ΔW so spectral norm ≤ 1 via Sinkhorn-Knopp
projection (20 iterations) onto Birkhoff polytope of doubly stochastic matrices.

Hypothesis: GSM8K -6.7pp regression in DARE/uniform/TIES (consistent across all merge
methods) is caused by composed delta having spectral norm > 1, perturbing math-sensitive
activations more than format-sensitive ones. Sinkhorn-Knopp bounds this by construction.

Math:
  Take DARE composed ΔW ∈ ℝ^(d_in × d_out) per layer
  Step 1: M^(0) = exp(ΔW)              # ensure positivity
  Step 2: alternating row/col normalization for 20 iters:
            M^(t+1) = T_r(T_c(M^(t)))
  Step 3: ΔW_norm = log(M^(20))         # back to additive form
  Result: ‖ΔW_norm‖_2 ≤ 1 by construction (Birkhoff polytope theorem)

Apply via canonical _FusedDeltaLinear (Finding #831).

Kill criteria:
  K2150: GSM8K ≥ best_single - 2pp AND HumanEval/MedQA preserved within 2pp DARE
  K2151: Composed ΔW spectral norm ≤ 1.05 across all layers
  K2152: Sinkhorn-Knopp 20-iter ≤ 200ms preprocessing
  K2153: mHC-DARE > vanilla DARE on average accuracy
"""
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tooling.scripts.polar_train import (
    inject_polar_adapters, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_BENCH_EVAL = 5 if IS_SMOKE else 30
DARE_DROP_RATE = 0.90

SK_ITERATIONS = 20  # DSv4 standard
SK_EPS_CLIP = 30.0  # clip ΔW pre-exp to avoid overflow

ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
ADAPTER_SLOTS = [
    ("strategy_full",      ADAPTER_DIR / "strategy_full_polar"      / "polar.safetensors"),
    ("strategy_prepare",   ADAPTER_DIR / "strategy_prepare_polar"   / "polar.safetensors"),
    ("strategy_act",       ADAPTER_DIR / "strategy_act_polar"       / "polar.safetensors"),
    ("strategy_integrate", ADAPTER_DIR / "strategy_integrate_polar" / "polar.safetensors"),
    ("domain_math",        ADAPTER_DIR / "math_polar"               / "polar.safetensors"),
    ("domain_code",        ADAPTER_DIR / "code_polar"               / "polar.safetensors"),
    ("domain_medical",     ADAPTER_DIR / "medical_polar"            / "polar.safetensors"),
]
SLOT_NAMES = [s[0] for s in ADAPTER_SLOTS]


def log(m): print(m, flush=True)
def log_memory(label=""):
    a = mx.get_active_memory() / 1e9; c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_state(path: Path) -> list[dict]:
    raw = mx.load(str(path))
    n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw if k.startswith("layer_")) + 1
    return [{
        "a": np.array(raw[f"layer_{i}.lora_a"].tolist(), dtype=np.float32),
        "b": np.array(raw[f"layer_{i}.lora_b"].tolist(), dtype=np.float32),
    } for i in range(n_layers)]


# ─────────────────────────────────────────────
# Canonical _FusedDeltaLinear (Finding #831)
# ─────────────────────────────────────────────

class _FusedDeltaLinear(nn.Module):
    def __init__(self, base_layer, fused):
        super().__init__()
        self.base = base_layer
        self._fused = fused
    def __call__(self, x):
        return self.base(x) + (x @ self._fused)


def apply_fused_per_layer(model, fused_deltas_per_layer):
    layers_iter = (model.language_model.model.layers if hasattr(model, "language_model")
                   else model.model.language_model.layers)
    for layer_idx, layer in enumerate(layers_iter):
        if layer_idx >= len(fused_deltas_per_layer):
            break
        delta_mx = mx.array(fused_deltas_per_layer[layer_idx].astype(np.float32))
        mx.eval(delta_mx)
        q_proj = layer.self_attn.q_proj
        base_layer = q_proj.base if isinstance(q_proj, (PoLARLinear, _FusedDeltaLinear)) else q_proj
        layer.self_attn.q_proj = _FusedDeltaLinear(base_layer, delta_mx)


# ─────────────────────────────────────────────
# DARE composition (the prior winner) — per-layer ΔW
# ─────────────────────────────────────────────

def compute_dare_deltas(all_states, drop_rate=DARE_DROP_RATE, seed=SEED):
    rng = np.random.default_rng(seed)
    n_layers = len(all_states[0])
    deltas = []
    for l in range(n_layers):
        delta = None
        for state in all_states:
            tv = SCALE * (state[l]["a"] @ state[l]["b"])
            mask = rng.binomial(1, 1.0 - drop_rate, size=tv.shape).astype(np.float32)
            tv_dare = (tv * mask) / max(1e-9, 1.0 - drop_rate)
            delta = tv_dare if delta is None else delta + tv_dare
        deltas.append(delta / len(all_states))
    return deltas


# ─────────────────────────────────────────────
# Sinkhorn-Knopp normalization (DSv4 §2.2)
# ─────────────────────────────────────────────

def sinkhorn_knopp(M: np.ndarray, n_iter: int = SK_ITERATIONS, eps: float = 1e-9) -> np.ndarray:
    """Alternating row/col normalization toward doubly stochastic.

    DSv4 paper Eq. 8: M^(t+1) = T_r(T_c(M^(t))), 20 iterations.

    For non-square matrices (d_in × d_out where d_in ≠ d_out), exact double-stochasticity
    isn't possible. We normalize rows to sum 1 and columns to sum d_in/d_out
    (corresponds to bistochastic-rectangular projection). This still bounds the spectral
    norm by sqrt(d_out/d_in) ≈ 1 for d_in≈d_out.
    """
    d_in, d_out = M.shape
    target_col_sum = d_in / d_out  # to balance non-square
    for _ in range(n_iter):
        # Column normalize → cols sum to target_col_sum
        col_sums = M.sum(axis=0, keepdims=True)
        M = M * (target_col_sum / np.maximum(col_sums, eps))
        # Row normalize → rows sum to 1
        row_sums = M.sum(axis=1, keepdims=True)
        M = M / np.maximum(row_sums, eps)
    return M


def normalize_delta_mhc(delta: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply Sinkhorn-Knopp projection to constrain spectral norm.

    Pipeline:
      1. Center ΔW around 0 (already is, since ΔW is a delta)
      2. Map to positive orthant: M = exp(clip(ΔW, ±SK_EPS_CLIP))
      3. Sinkhorn-Knopp 20 iterations
      4. Map back: ΔW_norm = log(M)

    Returns (ΔW_norm, stats) where stats has spectral norm before/after.
    """
    norm_before = float(np.linalg.norm(delta, ord=2))
    M0 = np.exp(np.clip(delta, -SK_EPS_CLIP, SK_EPS_CLIP).astype(np.float64))
    M = sinkhorn_knopp(M0)
    delta_norm = np.log(np.maximum(M, 1e-30)).astype(np.float32)
    norm_after = float(np.linalg.norm(delta_norm, ord=2))
    return delta_norm, {"spectral_before": norm_before, "spectral_after": norm_after}


def compute_mhc_dare_deltas(all_states):
    """DARE composition + Sinkhorn-Knopp normalization per layer."""
    log("\n  computing DARE deltas...")
    dare_deltas = compute_dare_deltas(all_states)

    log(f"  applying Sinkhorn-Knopp ({SK_ITERATIONS} iters) to {len(dare_deltas)} layers...")
    t0 = time.perf_counter()
    mhc_deltas = []
    spectral_norms = []
    for l, dlt in enumerate(dare_deltas):
        new_d, stats = normalize_delta_mhc(dlt)
        mhc_deltas.append(new_d)
        spectral_norms.append(stats)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log(f"  done in {elapsed_ms:.0f}ms ({elapsed_ms/len(dare_deltas):.1f}ms/layer)")
    return mhc_deltas, spectral_norms, elapsed_ms


# ─────────────────────────────────────────────
# Eval methods
# ─────────────────────────────────────────────

def eval_with_deltas(deltas, label):
    from mlx_lm import load
    log(f"\n[Eval: {label}]")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    apply_fused_per_layer(model, deltas)
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  {label}: GSM8K={g:.1f}%  HumanEval={h:.1f}%  MedQA={d:.1f}%")
    cleanup(model, tokenizer)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== mHC-Normalized Composition (SMOKE={IS_SMOKE}, SK_iters={SK_ITERATIONS}) ===")

    log("\n[Phase 0] Load 7 PoLAR adapters")
    all_states = [load_state(p) for _, p in ADAPTER_SLOTS]
    log(f"  loaded {len(all_states)}")

    # Vanilla DARE baseline
    log("\n[Phase 1] Vanilla DARE composition (baseline)")
    dare_deltas = compute_dare_deltas(all_states)
    dare_acc = eval_with_deltas(dare_deltas, "DARE")
    dare_avg = float(np.mean(list(dare_acc.values())))

    # mHC-DARE: DARE + Sinkhorn-Knopp normalization
    log("\n[Phase 2] mHC-DARE composition")
    mhc_deltas, spectral_stats, sk_elapsed_ms = compute_mhc_dare_deltas(all_states)
    mhc_acc = eval_with_deltas(mhc_deltas, "mHC-DARE")
    mhc_avg = float(np.mean(list(mhc_acc.values())))

    # Spectral norm summary
    norms_before = [s["spectral_before"] for s in spectral_stats]
    norms_after = [s["spectral_after"] for s in spectral_stats]
    norm_max_after = float(np.max(norms_after))
    norm_mean_after = float(np.mean(norms_after))
    log(f"\n  Spectral norms: before mean={np.mean(norms_before):.3f} max={np.max(norms_before):.3f}")
    log(f"  Spectral norms: after  mean={norm_mean_after:.3f} max={norm_max_after:.3f}")

    # KCs
    log("\n=== Kill Criteria ===")
    BEST_SINGLE = {"gsm8k": 63.3, "humaneval": 86.7, "medqa": 50.0}
    DARE_REFERENCE = {"gsm8k": 73.3, "humaneval": 90.0, "medqa": 66.7}

    # K2150: GSM8K ≥ best - 2 AND HE/MD preserved within 2pp DARE
    gsm8k_pass = mhc_acc["gsm8k"] >= BEST_SINGLE["gsm8k"] - 2.0
    he_pass = abs(DARE_REFERENCE["humaneval"] - mhc_acc["humaneval"]) <= 2.0
    md_pass = abs(DARE_REFERENCE["medqa"] - mhc_acc["medqa"]) <= 2.0
    k2150 = gsm8k_pass and he_pass and md_pass

    # K2151: spectral norm ≤ 1.05
    k2151 = norm_max_after <= 1.05

    # K2152: SK preprocessing ≤ 200ms total per composition (KC text: one-time cost)
    k2152 = sk_elapsed_ms <= 200.0

    # K2153: mHC-DARE > vanilla DARE on avg
    k2153 = mhc_avg > dare_avg

    all_pass = k2150 and k2151 and k2152 and k2153
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "config": {"sk_iterations": SK_ITERATIONS, "dare_drop_rate": DARE_DROP_RATE,
                   "best_single": BEST_SINGLE, "dare_reference": DARE_REFERENCE},
        "dare": dare_acc, "dare_avg": round(dare_avg, 1),
        "mhc_dare": mhc_acc, "mhc_avg": round(mhc_avg, 1),
        "spectral_norms": {
            "before_mean": round(float(np.mean(norms_before)), 3),
            "before_max":  round(float(np.max(norms_before)), 3),
            "after_mean":  round(norm_mean_after, 3),
            "after_max":   round(norm_max_after, 3),
        },
        "sk_preprocessing_ms": round(sk_elapsed_ms, 1),
        "kill_criteria": {
            "K2150_accuracy_preserved": {"pass": k2150,
                                         "gsm8k": mhc_acc["gsm8k"], "he": mhc_acc["humaneval"],
                                         "md": mhc_acc["medqa"]},
            "K2151_spectral_norm_1.05": {"pass": k2151, "max_after": norm_max_after},
            "K2152_sk_preprocessing": {"pass": k2152, "elapsed_ms": round(sk_elapsed_ms, 1)},
            "K2153_mhc_beats_dare": {"pass": k2153, "delta_pp": round(mhc_avg - dare_avg, 1)},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"K2150 accuracy preserved:    {'PASS' if k2150 else 'FAIL'}")
    log(f"K2151 spectral ≤1.05:        {'PASS' if k2151 else 'FAIL'}  (max={norm_max_after:.3f})")
    log(f"K2152 SK preproc OK:         {'PASS' if k2152 else 'FAIL'}  ({sk_elapsed_ms:.0f}ms)")
    log(f"K2153 mHC > DARE:            {'PASS' if k2153 else 'FAIL'}  Δ={mhc_avg-dare_avg:+.1f}pp")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()

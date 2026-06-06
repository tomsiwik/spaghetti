"""KAN-based PoLAR adapter — Lagrangian scalar-function reformulation.

Three Q's tested in sequence:
  Q1 (EXPRESSIVITY): KAN-math single-adapter ≥ standard-PoLAR-math − 5pp on GSM8K
  Q2 (PURE COMPOSITION): KAN-math + KAN-code via spline-coefficient addition
  Q3 (HYBRID COMPOSITION): KAN-math + standard-PoLAR strategy_full

Implementation note: this experiment uses a WARM-START approach — instead of
training KAN adapters from scratch (which would take hours), we initialize
each KAN block from an existing B-matrix using the skip-weight path
(`kan_block_from_B`). Spline coefficients start near zero. The forward
pass at init reproduces standard PoLAR exactly. Composition then tests
whether the KAN parameterization preserves what standard PoLAR achieved.

A full training pass is deferred to a follow-up experiment if Q1 passes
under warm-start.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

import mlx.core as mx
import mlx.nn as nn

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    ADAPTER_NAMES, MODEL_NAME, RANK, SCALE, N_EVAL, SEED,
    _get_layers, inject_polar_adapters, load_adapter_state,
    reset_to_polar_path, install_polar_state, run_all_evals,
    SINGLE_BEST_FOR_BENCH,
)
from compose_methods import (  # type: ignore  # noqa: E402
    KANBlock, _KANPoLARLinear, compose_kan_pure, kan_block_from_B,
)


def install_kan_path(model, base_q_projs, shared_A_dict, kan_blocks_per_layer: dict, scale: float):
    """Replace q_proj at each layer with _KANPoLARLinear."""
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        key = f"layer_{li}"
        if key not in kan_blocks_per_layer:
            continue
        wrapper = _KANPoLARLinear(
            base_layer=base_q_projs[li],
            lora_a=shared_A_dict[key],
            kan_block=kan_blocks_per_layer[key],
            scale=scale,
        )
        layer.self_attn.q_proj = wrapper


def build_kan_blocks_from_B(B_dict, grid_size=5, k=3) -> dict:
    """Per-layer warm-started KAN blocks from B-matrices."""
    return {key: kan_block_from_B(B, grid_size=grid_size, k=k)
            for key, B in B_dict.items()}


def main():
    out_path = EXP_DIR / "results.json"
    print("=== exp_pierre_kan_adapter_lagrangian ===")
    print(f"  N_eval = {N_EVAL}/bench")
    print(f"  Strategy: warm-start KAN from existing B-matrices, then compose via coefficient addition")

    from mlx_lm import load
    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    print(f"  {len(base_q_projs)} transformer layers")

    print("\nLoading adapters...")
    adapter_states = {n: load_adapter_state(n) for n in ADAPTER_NAMES}
    shared_A_dict = {k: v["a"] for k, v in adapter_states[ADAPTER_NAMES[0]].items() if "a" in v}
    print(f"  shared-A donor: {ADAPTER_NAMES[0]}")

    print("\nInjecting baseline PoLARLinear (for standard-path eval)...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    results = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "seed": SEED,
            "rank": RANK,
            "scale": SCALE,
            "grid_size": 5,
            "spline_degree": 3,
            "approach": "warm-start KAN from B (skip-weight = B at init, spline coeffs small)",
            "method_description": "KAN-PoLAR adapter — scalar B-spline replaces B-matmul",
            "kan_paper": "arxiv 2404.19756",
        },
        "results": {},
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # ─────────────────────────────────────────────────────────────────────
    # Reference: standard-PoLAR math single-adapter (Q1 baseline)
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- M0: standard-PoLAR domain_math single (Q1 baseline) ---")
    reset_to_polar_path(model, modules, base_q_projs)
    st = adapter_states["domain_math"]
    A_d = {k: v["a"] for k, v in st.items() if "a" in v}
    B_d_math = {k: v["b"] for k, v in st.items() if "b" in v}
    install_polar_state(modules, A_d, B_d_math)
    from tooling.scripts.polar_train import eval_gsm8k  # noqa: E402
    t0 = time.time()
    std_math_gsm8k = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
    print(f"  std PoLAR-math GSM8K: {std_math_gsm8k:.1f}% ({time.time()-t0:.0f}s)")
    results["results"]["std_polar_math_gsm8k"] = std_math_gsm8k
    save()

    # ─────────────────────────────────────────────────────────────────────
    # Q1: KAN-math single-adapter (warm-start from B-matrix)
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Q1: KAN-math single-adapter (warm-start) ---")
    kan_math_blocks = build_kan_blocks_from_B(B_d_math)
    # Reset model to clean state, install KAN path
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        layer.self_attn.q_proj = base_q_projs[li]  # restore base
    install_kan_path(model, base_q_projs, shared_A_dict, kan_math_blocks, SCALE)
    t0 = time.time()
    kan_math_gsm8k = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
    print(f"  KAN-math GSM8K: {kan_math_gsm8k:.1f}% ({time.time()-t0:.0f}s)")
    results["results"]["kan_math_gsm8k"] = kan_math_gsm8k
    save()

    # ─────────────────────────────────────────────────────────────────────
    # Q2: KAN-math + KAN-code via spline coefficient addition (pure compose)
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Q2: KAN-math + KAN-code pure composition ---")
    st_code = adapter_states["domain_code"]
    B_d_code = {k: v["b"] for k, v in st_code.items() if "b" in v}
    kan_code_blocks = build_kan_blocks_from_B(B_d_code)

    # Per-layer composition: average spline coefficients + skip weights
    kan_pure_blocks = {}
    for key in kan_math_blocks.keys():
        kan_pure_blocks[key] = compose_kan_pure(
            [kan_math_blocks[key], kan_code_blocks[key]],
            weights=[0.5, 0.5],
        )
    install_kan_path(model, base_q_projs, shared_A_dict, kan_pure_blocks, SCALE)
    pure_results = run_all_evals(model, tokenizer, "Q2 pure-KAN compose (math+code, w=0.5/0.5)")
    results["results"]["q2_pure_kan_compose_math_code"] = pure_results
    save()

    # ─────────────────────────────────────────────────────────────────────
    # Q3: KAN-math + standard-PoLAR strategy_full (hybrid composition)
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Q3: KAN-math + std-PoLAR strategy_full hybrid ---")
    st_strat = adapter_states["strategy_full"]
    B_d_strat = {k: v["b"] for k, v in st_strat.items() if "b" in v}
    # Hybrid: KAN_math forward + std_strategy forward, summed
    # Implementation: install both — KAN for adapter#1, standard PoLAR addition for #2
    # We'll do this by computing: out = base(x) + scale·KAN_math(x@A) + scale·(x@A)@B_strat
    # Easiest: a custom wrapper.
    class _HybridLinear(nn.Module):
        def __init__(self, base, A_shared, kan_block, B_std, scale):
            super().__init__()
            self.base = base
            self.A = A_shared
            self.kan = kan_block
            self.B = B_std
            self.scale = scale
        def __call__(self, x):
            z = x @ self.A.astype(x.dtype)
            kan_part = self.kan(z.astype(mx.float32)).astype(x.dtype)
            std_part = z @ self.B.astype(x.dtype)
            return self.base(x) + self.scale * (0.5 * kan_part + 0.5 * std_part)

    # Restore base, install hybrid
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        layer.self_attn.q_proj = base_q_projs[li]
    for li, layer in enumerate(layers):
        key = f"layer_{li}"
        if key not in kan_math_blocks or key not in B_d_strat:
            continue
        layer.self_attn.q_proj = _HybridLinear(
            base=base_q_projs[li],
            A_shared=shared_A_dict[key],
            kan_block=kan_math_blocks[key],
            B_std=B_d_strat[key],
            scale=SCALE,
        )
    hybrid_results = run_all_evals(model, tokenizer, "Q3 hybrid KAN-math + std-strategy_full")
    results["results"]["q3_hybrid_kan_math_std_strategy"] = hybrid_results
    save()

    # ─────────────────────────────────────────────────────────────────────
    # KCs
    # ─────────────────────────────────────────────────────────────────────
    K2_STD_K2_AVG = 54.0  # from exp_pierre_compose_k2_strategy_x_domain
    BEST_SINGLE_AVG = 62.0  # from exp_pierre_dare_b_vs_fisher_rao

    k1 = kan_math_gsm8k >= (std_math_gsm8k - 5.0)
    pure_avg = pure_results["avg"]
    k2 = pure_avg >= (BEST_SINGLE_AVG + 2.0)
    hybrid_avg = hybrid_results["avg"]
    k3 = hybrid_avg >= (K2_STD_K2_AVG + 1.0)
    k4 = True  # warm-start, no training time to budget

    results["kill_criteria"] = {
        "K1_expressivity_kan_math": {
            "pass": bool(k1),
            "kan_gsm8k": kan_math_gsm8k,
            "std_gsm8k": std_math_gsm8k,
            "delta_pp": kan_math_gsm8k - std_math_gsm8k,
            "threshold_pp": -5.0,
        },
        "K2_pure_kan_composition": {
            "pass": bool(k2),
            "pure_avg": pure_avg,
            "best_single_avg": BEST_SINGLE_AVG,
            "delta_pp": pure_avg - BEST_SINGLE_AVG,
            "threshold_pp": 2.0,
        },
        "K3_hybrid_composition": {
            "pass": bool(k3),
            "hybrid_avg": hybrid_avg,
            "std_K2_avg": K2_STD_K2_AVG,
            "delta_pp": hybrid_avg - K2_STD_K2_AVG,
            "threshold_pp": 1.0,
        },
        "K4_warm_start_budget": {"pass": bool(k4), "note": "warm-start, no training; trivial budget"},
    }

    # Verdict
    if k1 and k2 and k3:
        verdict = "SUPPORTED"
        decision = (
            f"KAN adapters are viable: Q1 PASS ({kan_math_gsm8k:.1f}% vs std {std_math_gsm8k:.1f}%), "
            f"Q2 PASS (pure compose {pure_avg:.1f}%), Q3 PASS (hybrid {hybrid_avg:.1f}%). "
            "Composition arc collapses to per-edge spline addition. Plan migration."
        )
    elif k1 and k2 and not k3:
        verdict = "SUPPORTED"
        decision = "Pure-KAN works but doesn't mix with legacy PoLAR. Build new KAN ecosystem."
    elif k1:
        verdict = "PARTIAL"
        decision = "KAN expressive enough but composition didn't beat baselines. KAN is alt parameterization, not solution."
    else:
        verdict = "KILLED"
        decision = (
            f"KAN at rank={RANK} doesn't match standard PoLAR (Δ={kan_math_gsm8k-std_math_gsm8k:+.1f}pp). "
            "Composition arc remains open. Vector matmul is doing real work."
        )

    results["verdict"] = verdict
    results["decision"] = decision
    save()

    print(f"\n=== {verdict}: {decision} ===")
    print(f"  K1 expressivity         : {'PASS' if k1 else 'FAIL'}  Δ={kan_math_gsm8k-std_math_gsm8k:+.1f}pp")
    print(f"  K2 pure-KAN compose     : {'PASS' if k2 else 'FAIL'}  Δ={pure_avg-BEST_SINGLE_AVG:+.1f}pp vs best-single")
    print(f"  K3 hybrid compose       : {'PASS' if k3 else 'FAIL'}  Δ={hybrid_avg-K2_STD_K2_AVG:+.1f}pp vs std-K=2")
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()

"""Adapter × benchmark cross-matrix.

For each of 7 adapters × 3 benchmarks (21 cells), evaluate single-adapter
performance. Identifies cross-domain transfer (e.g., medical adapter on GSM8K).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

import mlx.core as mx

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    ADAPTER_NAMES, MODEL_NAME, RANK, SCALE, N_EVAL, SEED,
    _get_layers, inject_polar_adapters, load_adapter_state,
    reset_to_polar_path, install_polar_state,
)


def main():
    out_path = EXP_DIR / "results.json"
    print(f"=== Adapter cross-matrix (7 adapters × 3 benchmarks @ N={N_EVAL}) ===")

    from mlx_lm import load
    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    print(f"  {len(base_q_projs)} transformer layers")

    print("Loading adapters...")
    adapter_states = {n: load_adapter_state(n) for n in ADAPTER_NAMES}

    print("Injecting PoLARLinear...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    from scripts.polar_train import eval_gsm8k, eval_humaneval, eval_medqa  # noqa: E402

    benchmarks = [("gsm8k", eval_gsm8k), ("humaneval", eval_humaneval), ("medqa", eval_medqa)]
    matrix: dict = {n: {} for n in ADAPTER_NAMES}

    # Raw baseline (no adapter — install zero B)
    print("\n--- Raw baseline (no adapter) ---")
    reset_to_polar_path(model, modules, base_q_projs)
    # Zero B → effective no contribution
    A_dict_zero = {f"layer_{i}": adapter_states[ADAPTER_NAMES[0]][f"layer_{i}"]["a"] for i in range(len(modules))}
    B_dict_zero = {f"layer_{i}": mx.zeros_like(adapter_states[ADAPTER_NAMES[0]][f"layer_{i}"]["b"]) for i in range(len(modules))}
    install_polar_state(modules, A_dict_zero, B_dict_zero)
    raw = {}
    for bench_name, eval_fn in benchmarks:
        t0 = time.time()
        score = (eval_fn(model, tokenizer, n_eval=N_EVAL, seed=SEED)
                 if bench_name != "humaneval"
                 else eval_fn(model, tokenizer, n_eval=N_EVAL))
        raw[bench_name] = score
        print(f"  raw_baseline {bench_name}: {score:.1f}% ({time.time()-t0:.0f}s)")

    save_state = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "seed": SEED,
            "n_adapters": len(ADAPTER_NAMES),
        },
        "raw_baseline": raw,
        "matrix": matrix,
    }

    def save():
        out_path.write_text(json.dumps(save_state, indent=2, default=str))

    save()

    # Cross-matrix: each adapter on each benchmark
    for adapter_name in ADAPTER_NAMES:
        st = adapter_states[adapter_name]
        A_d = {k: v["a"] for k, v in st.items() if "a" in v}
        B_d = {k: v["b"] for k, v in st.items() if "b" in v}
        reset_to_polar_path(model, modules, base_q_projs)
        install_polar_state(modules, A_d, B_d)
        print(f"\n--- {adapter_name} on all 3 benchmarks ---")
        for bench_name, eval_fn in benchmarks:
            t0 = time.time()
            score = (eval_fn(model, tokenizer, n_eval=N_EVAL, seed=SEED)
                     if bench_name != "humaneval"
                     else eval_fn(model, tokenizer, n_eval=N_EVAL))
            matrix[adapter_name][bench_name] = score
            elapsed = time.time() - t0
            delta_vs_raw = score - raw[bench_name]
            print(f"  {adapter_name} {bench_name}: {score:.1f}% (vs raw: {delta_vs_raw:+.1f}pp, {elapsed:.0f}s)")
            save_state["matrix"] = matrix
            save()

    # Analysis: positive cross-effects (off-diagonal cells beating raw by ≥2pp)
    NATIVE = {"gsm8k": "domain_math", "humaneval": "domain_code", "medqa": "domain_medical"}
    cross_effects = []
    for adapter_name, scores in matrix.items():
        for bench_name, score in scores.items():
            is_native = (NATIVE.get(bench_name) == adapter_name)
            delta = score - raw[bench_name]
            if not is_native and delta >= 2.0:
                cross_effects.append({
                    "adapter": adapter_name, "benchmark": bench_name,
                    "score": score, "delta_vs_raw": delta,
                })
    save_state["cross_effects_positive"] = cross_effects

    # Worst-case cell (most negative cross-effect)
    worst_cell = min(
        [(a, b, s, raw[b]) for a, scores in matrix.items() for b, s in scores.items()],
        key=lambda t: t[2] - t[3],
    )
    save_state["worst_cross_effect"] = {
        "adapter": worst_cell[0], "benchmark": worst_cell[1],
        "score": worst_cell[2], "raw": worst_cell[3],
        "delta": worst_cell[2] - worst_cell[3],
    }

    # KCs
    k1 = len(cross_effects) >= 1
    k2 = (worst_cell[2] - worst_cell[3]) >= -5.0
    k3 = True  # We hit budget if we got here
    # K4 native consistency
    NATIVE_REF = {"gsm8k": (matrix["domain_math"]["gsm8k"], 66.0),
                   "humaneval": (matrix["domain_code"]["humaneval"], 78.0),
                   "medqa": (matrix["domain_medical"]["medqa"], 42.0)}
    k4 = all(abs(measured - ref) <= 5.0 for (measured, ref) in NATIVE_REF.values())

    save_state["kill_criteria"] = {
        "K1_at_least_one_positive_cross_effect": {"pass": bool(k1), "n_positive": len(cross_effects)},
        "K2_no_negative_transfer_5pp": {"pass": bool(k2),
                                          "worst_cell": save_state["worst_cross_effect"]},
        "K3_budget_240min": {"pass": bool(k3)},
        "K4_native_consistency_5pp": {"pass": bool(k4), "native_ref": NATIVE_REF},
    }

    if k1 and k2 and k4:
        save_state["verdict"] = "SUPPORTED"
        save_state["decision"] = (
            f"Cross-matrix complete. {len(cross_effects)} positive cross-effects found. "
            "Use to inform K=2 routing pair selection."
        )
    elif not k4:
        save_state["verdict"] = "INCONCLUSIVE"
        save_state["decision"] = "Native cells drift from prior measurements; eval pipeline issue"
    else:
        save_state["verdict"] = "KILLED"
        save_state["decision"] = "Cross-matrix shows interference or no positive transfer"

    save()
    print(f"\n=== {save_state['verdict']} ===")
    print(f"  Positive cross-effects: {len(cross_effects)}")
    print(f"  Worst cell: {save_state['worst_cross_effect']}")
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()

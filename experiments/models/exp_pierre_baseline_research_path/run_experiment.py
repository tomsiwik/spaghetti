"""Pierre vs raw Gemma 4 — research-path SOTA baseline."""
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
    stack_B_dicts, reset_to_polar_path, install_polar_state,
    run_all_evals, compose_fisher_rao,
    SINGLE_BEST_FOR_BENCH,
)


def main():
    out_path = EXP_DIR / "results.json"
    print(f"=== Pierre vs raw Gemma 4 — research-path SOTA baseline ===")
    print(f"  N_eval = {N_EVAL}/bench")

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

    results = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "seed": SEED,
            "n_adapters": len(ADAPTER_NAMES),
            "adapter_names": ADAPTER_NAMES,
            "method_description": "Pierre vs Raw Gemma 4 baseline establishment",
        },
        "methods": {},
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # M_raw: Raw Gemma 4 (no PoLAR injection — base model untouched)
    print("\n--- M_raw: Raw Gemma 4 (no adapters) ---")
    raw_results = run_all_evals(model, tokenizer, "raw_gemma_4")
    results["methods"]["raw_gemma_4"] = raw_results
    save()

    # Now inject PoLARLinear (with zero B at first, so still effectively raw)
    print("\nInjecting PoLARLinear...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    # M_fr: Pierre + Fisher-Rao K=7
    print("\n--- M_fr: Pierre + Fisher-Rao K=7 (current product default) ---")
    reset_to_polar_path(model, modules, base_q_projs)
    B_lists = stack_B_dicts([adapter_states[n] for n in ADAPTER_NAMES])
    fr_B = compose_fisher_rao(B_lists)
    install_polar_state(modules, shared_A_dict, fr_B)
    fr_results = run_all_evals(model, tokenizer, "pierre_fisher_rao_K7")
    results["methods"]["pierre_fisher_rao_K7"] = fr_results
    save()

    # M_oracle: best single adapter per benchmark (oracle routing)
    print("\n--- M_oracle: best single adapter per benchmark ---")
    oracle_per_bench = {}
    for bench, adapter_name in SINGLE_BEST_FOR_BENCH.items():
        reset_to_polar_path(model, modules, base_q_projs)
        st = adapter_states[adapter_name]
        A_d = {k: v["a"] for k, v in st.items() if "a" in v}
        B_d = {k: v["b"] for k, v in st.items() if "b" in v}
        install_polar_state(modules, A_d, B_d)
        from tooling.scripts.polar_train import eval_gsm8k, eval_humaneval, eval_medqa  # noqa: E402
        if bench == "gsm8k":
            score = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
        elif bench == "humaneval":
            score = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
        else:
            score = eval_medqa(model, tokenizer, n_eval=N_EVAL, seed=SEED)
        oracle_per_bench[bench] = score
        print(f"  {bench}: {score:.1f}% (using {adapter_name})")
    oracle_avg = sum(oracle_per_bench.values()) / 3.0
    results["methods"]["pierre_oracle_routing"] = {"per_bench": oracle_per_bench, "avg": oracle_avg}
    save()

    # KCs
    raw_avg = raw_results["avg"]
    fr_avg = fr_results["avg"]

    k1 = (fr_avg - raw_avg) >= 3.0
    k2 = all(fr_results[b] >= raw_results[b] for b in ["gsm8k", "humaneval", "medqa"])
    k3 = all(fr_results[b] >= raw_results[b] - 2.0 for b in ["gsm8k", "humaneval", "medqa"])
    k4 = (raw_results["gsm8k"] >= 50.0
          and raw_results["humaneval"] >= 65.0
          and raw_results["medqa"] >= 35.0)

    results["kill_criteria"] = {
        "K1_pierre_beats_raw_3pp": {
            "pass": bool(k1), "delta_pp": fr_avg - raw_avg, "threshold_pp": 3.0,
            "raw_avg": raw_avg, "pierre_fr_avg": fr_avg,
        },
        "K2_per_benchmark_win": {
            "pass": bool(k2),
            "per_bench_deltas": {b: fr_results[b] - raw_results[b]
                                 for b in ["gsm8k", "humaneval", "medqa"]},
        },
        "K3_no_per_bench_regression_2pp": {
            "pass": bool(k3),
            "per_bench_deltas": {b: fr_results[b] - raw_results[b]
                                 for b in ["gsm8k", "humaneval", "medqa"]},
        },
        "K4_raw_gemma_sanity": {
            "pass": bool(k4),
            "raw_gsm8k": raw_results["gsm8k"],
            "raw_humaneval": raw_results["humaneval"],
            "raw_medqa": raw_results["medqa"],
            "thresholds": {"gsm8k": 50.0, "humaneval": 65.0, "medqa": 35.0},
        },
    }

    if not k4:
        verdict = "INCONCLUSIVE"
        decision = "Raw Gemma 4 numbers below sanity floor; eval pipeline drift suspected"
    elif k1 and k2:
        verdict = "SUPPORTED"
        decision = (
            f"Pierre+Fisher-Rao beats raw Gemma 4 by {fr_avg-raw_avg:+.1f}pp avg "
            f"({fr_avg:.1f}% vs {raw_avg:.1f}%). Wins on all 3 benchmarks. "
            "Composition adapters demonstrably add product value over base."
        )
    elif k1:
        verdict = "SUPPORTED"
        decision = (
            f"Pierre wins on average ({fr_avg-raw_avg:+.1f}pp) but not on every benchmark "
            f"(K2 fail). Investigate which benchmark regresses."
        )
    else:
        verdict = "KILLED"
        decision = (
            f"Pierre+Fisher-Rao does not beat raw Gemma 4 by required margin "
            f"(Δ={fr_avg-raw_avg:+.1f}pp). Composition is not paying its cost on these benchmarks."
        )

    results["verdict"] = verdict
    results["decision"] = decision
    save()

    print(f"\n=== {verdict}: {decision} ===")
    print(f"  K1 Pierre beats raw +3pp : {'PASS' if k1 else 'FAIL'}  Δ={fr_avg-raw_avg:+.1f}pp")
    print(f"  K2 wins on every bench   : {'PASS' if k2 else 'FAIL'}")
    print(f"  K3 no per-bench regress  : {'PASS' if k3 else 'FAIL'}")
    print(f"  K4 raw Gemma sanity      : {'PASS' if k4 else 'FAIL'}")
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()

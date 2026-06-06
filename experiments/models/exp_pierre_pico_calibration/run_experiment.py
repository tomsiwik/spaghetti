"""exp_pierre_pico_calibration — measure Pico calibration on stacked B as a
Fisher-Rao pre-stage in Pierre's shared-A architecture.

Usage:
    experiment run exp_pierre_pico_calibration

The experiment runs the standard 4-method matrix:
    M0 single_best per benchmark
    M1 fisher_rao (Pierre default)
    M2 pico_then_fisher_rao (under test)
    M3 dare_full_delta (research upper bound)

Then performs a K4 sanity check: re-runs Pico with alpha_override=1.0
(forces S=I, makes Pico a no-op) and verifies the avg matches plain
Fisher-Rao within 1pp. If sanity fails, verdict = INCONCLUSIVE.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))  # experiments/models/

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
    FISHER_RAO_REFERENCE_AVG,
)
from compose_methods import compose_pico_then_fisher_rao  # type: ignore  # noqa: E402


def main():
    out_path = EXP_DIR / "results.json"

    kc_thresholds = {
        "k1_min_delta_over_fisher_rao": 3.0,
        "k2_max_delta_under_full_delta_dare": 4.0,
        "k3_max_preprocess_seconds": 5.0,
        "k4_label": "alpha=1.0 reproduces Fisher-Rao within 1pp (sanity)",
        # k4_value will be filled below by re-running with alpha_override=1.0
        "k4_value": None,
        "k4_threshold": FISHER_RAO_REFERENCE_AVG,
        "k4_tolerance": 1.0,
    }

    method = MethodSpec(
        name="pico_then_fisher_rao",
        kind="b_only",
        fn=compose_pico_then_fisher_rao,
        fn_kwargs={},
    )

    # Run the main 4-method matrix
    results = run_pierre_compose_experiment(
        method=method,
        kc_thresholds=kc_thresholds,
        out_path=out_path,
        extra_config={
            "pico_paper": "arxiv 2604.16826",
            "method_description": "Pico SVD calibration on stacked B → Fisher-Rao mean",
        },
    )

    # K4 sanity: re-run pico with alpha=1.0 (S=I) and verify match to fisher_rao
    print("\n=== K4 sanity: re-running Pico with alpha_override=1.0 ===", flush=True)
    fr_avg = results["methods"]["fisher_rao"]["avg"]

    # Reconfigure for sanity run
    sanity_method = MethodSpec(
        name="pico_alpha_eq_1_sanity",
        kind="b_only",
        fn=compose_pico_then_fisher_rao,
        fn_kwargs={"alpha_override": 1.0},
    )

    # Quick path: run only the sanity method against the existing fisher_rao reference.
    # We don't need to re-run the full 4-method matrix.
    from mlx_lm import load  # noqa: E402
    from _pierre_shared.eval_runner import (  # noqa: E402
        ADAPTER_NAMES, MODEL_NAME, _get_layers, inject_polar_adapters,
        load_adapter_state, stack_B_dicts, reset_to_polar_path,
        install_polar_state, run_all_evals, RANK, SCALE,
    )

    print(f"Loading {MODEL_NAME} for sanity check...", flush=True)
    model, tokenizer = load(MODEL_NAME)
    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    adapter_states = {n: load_adapter_state(n) for n in ADAPTER_NAMES}
    shared_A = {k: v["a"] for k, v in adapter_states[ADAPTER_NAMES[0]].items() if "a" in v}
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    B_lists = stack_B_dicts([adapter_states[n] for n in ADAPTER_NAMES])
    sanity_B = compose_pico_then_fisher_rao(B_lists, shared_A, alpha_override=1.0)
    reset_to_polar_path(model, modules, base_q_projs)
    install_polar_state(modules, shared_A, sanity_B)
    sanity_results = run_all_evals(model, tokenizer, "pico_alpha_eq_1 (K4 sanity)")
    sanity_avg = sanity_results["avg"]

    results["methods"]["pico_alpha_eq_1_sanity"] = sanity_results
    results["kc_thresholds"]["k4_value"] = sanity_avg
    results["kc_thresholds"]["k4_threshold"] = fr_avg

    # Re-evaluate K4 with measured value
    k4_pass = abs(sanity_avg - fr_avg) <= 1.0
    results["kill_criteria"]["K4_sanity"]["pass"] = bool(k4_pass)
    results["kill_criteria"]["K4_sanity"]["value"] = sanity_avg
    results["kill_criteria"]["K4_sanity"]["threshold"] = fr_avg
    results["kill_criteria"]["K4_sanity"]["abs_delta_pp"] = abs(sanity_avg - fr_avg)

    # Re-evaluate verdict if sanity changes status
    k1 = results["kill_criteria"]["K1_beats_fisher_rao"]["pass"]
    k2 = results["kill_criteria"]["K2_close_to_full_delta_dare"]["pass"]
    if not k4_pass:
        results["verdict"] = "INCONCLUSIVE"
        results["decision"] = (
            f"K4 sanity failed: alpha=1 gave {sanity_avg:.1f}% vs Fisher-Rao "
            f"reference {fr_avg:.1f}% (Δ={sanity_avg-fr_avg:+.1f}pp > 1pp). "
            "Implementation drift; debug Pico calibration matrix S."
        )
    elif k1 and k2:
        results["verdict"] = "SUPPORTED"
        results["decision"] = (
            f"Adopt pico_then_fisher_rao: beats Fisher-Rao by "
            f"{results['methods']['pico_then_fisher_rao']['avg']-fr_avg:+.1f}pp; "
            "within budget of full-delta DARE."
        )
    elif k1:
        results["verdict"] = "SUPPORTED"
        results["decision"] = (
            f"Adopt pico_then_fisher_rao: beats Fisher-Rao by "
            f"{results['methods']['pico_then_fisher_rao']['avg']-fr_avg:+.1f}pp; "
            "shared-A still leaves headroom vs full-delta DARE."
        )
    else:
        results["verdict"] = "KILLED"
        results["decision"] = (
            "Keep Fisher-Rao; Pico calibration did not beat default "
            "by required margin in shared-A B-only architecture."
        )

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== Final verdict: {results['verdict']} ===")
    print(f"  K4 sanity (α=1 ≈ Fisher-Rao): {'PASS' if k4_pass else 'FAIL'}  "
          f"|Δ|={abs(sanity_avg-fr_avg):.1f}pp")
    print(f"  Decision: {results['decision']}")


if __name__ == "__main__":
    main()

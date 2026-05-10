"""Does Pico calibration rescue the failed B-space DARE?"""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)
from compose_methods import compose_pico_rescues_dare_b  # type: ignore  # noqa: E402


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="pico_then_dare_b",
            kind="b_only",
            fn=compose_pico_rescues_dare_b,
            fn_kwargs={"drop_rate": 0.9, "seed": 42},
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 3.0,
            "k2_max_delta_under_full_delta_dare": 4.0,
            "k3_max_preprocess_seconds": 7.0,
            "k4_label": "MedQA recovery from prior 30% collapse (≥50%)",
            "k4_value": None,
            "k4_threshold": 50.0,
            "k4_tolerance": 50.0,  # we want value ≥ threshold, not |Δ| ≤ tolerance
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "papers": ["arxiv 2604.16826 (Pico)", "arxiv 2311.03099 (DARE)"],
            "tests": "Does Pico's B-direction calibration rescue the dare_b failure?",
            "prior_dare_b_avg": 55.3,
            "prior_dare_b_medqa": 30.0,
        },
    )


if __name__ == "__main__":
    main()

"""Pico calibration + TIES merge (combinatorial test)."""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)
from compose_methods import compose_pico_then_ties  # type: ignore  # noqa: E402


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="pico_then_ties",
            kind="fused_delta",
            fn=compose_pico_then_ties,
            fn_kwargs={"keep_frac": 0.3},
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 3.0,
            "k2_max_delta_under_full_delta_dare": 4.0,
            "k3_max_preprocess_seconds": 35.0,
            "k4_label": "Pico+TIES > max(Pico+FR, TIES alone)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "papers": ["arxiv 2604.16826 (Pico)", "arxiv 2306.01708 (TIES)"],
            "method_description": "Pico SVD calibration on B-stack → TIES on materialized deltas",
            "tests": "orthogonality of B-space calibration and full-delta sign-aware merge",
        },
    )


if __name__ == "__main__":
    main()

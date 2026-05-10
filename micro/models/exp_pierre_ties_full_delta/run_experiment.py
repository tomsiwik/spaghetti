"""TIES-Merging on shared-A materialized deltas."""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)
from compose_methods import compose_ties_full_delta  # type: ignore  # noqa: E402


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="ties_full_delta",
            kind="fused_delta",
            fn=compose_ties_full_delta,
            fn_kwargs={"keep_frac": 0.3},
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 3.0,
            "k2_max_delta_under_full_delta_dare": 4.0,
            "k3_max_preprocess_seconds": 30.0,
            "k4_label": "TIES configured per paper defaults (keep_frac=0.3)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "ties_paper": "arxiv 2306.01708",
            "ties_repo": "https://github.com/prateeky2806/ties-merging",
            "method_description": "Trim+Sign-Elect+Disjoint mean on shared-A materialized deltas",
        },
    )


if __name__ == "__main__":
    main()

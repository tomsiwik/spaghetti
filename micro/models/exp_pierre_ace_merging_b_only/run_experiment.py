"""exp_pierre_ace_merging_b_only — measure ACE-Merging adapted to Pierre's
shared-A architecture, comparing against Fisher-Rao default and full-delta
DARE upper bound.

Usage:
    experiment run exp_pierre_ace_merging_b_only
"""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)
from compose_methods import compose_ace_merging_b_only  # type: ignore  # noqa: E402


def main():
    out_path = EXP_DIR / "results.json"

    kc_thresholds = {
        "k1_min_delta_over_fisher_rao": 3.0,
        "k2_max_delta_under_full_delta_dare": 4.0,
        "k3_max_preprocess_seconds": 30.0,  # ACE is heavier than Pico (per-layer SVD + inv)
        "k4_label": "ACE configured per released code defaults",
        "k4_value": None,
        "k4_threshold": None,
    }

    method = MethodSpec(
        name="ace_merging",
        kind="fused_delta",
        fn=compose_ace_merging_b_only,
        fn_kwargs={
            "eps": 1e-2,
            "tau": 0.3,
            "k_frac": 0.3,
            "force_disable_spectral": False,
        },
    )

    run_pierre_compose_experiment(
        method=method,
        kc_thresholds=kc_thresholds,
        out_path=out_path,
        extra_config={
            "ace_paper": "arxiv 2603.02945",
            "ace_repo": "https://github.com/unravel-xu/ACE-Merging",
            "method_description": "Closed-form covariance-weighted merge on shared-A materialized deltas",
        },
    )


if __name__ == "__main__":
    main()

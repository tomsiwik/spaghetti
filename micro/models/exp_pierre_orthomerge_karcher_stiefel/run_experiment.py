"""exp_pierre_orthomerge_karcher_stiefel — measure OrthoMerge (Karcher mean
on Stiefel + magnitude correction) ported to Pierre's shared-A architecture.

Usage:
    experiment run exp_pierre_orthomerge_karcher_stiefel
"""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)
from compose_methods import compose_orthomerge_karcher  # type: ignore  # noqa: E402


def main():
    out_path = EXP_DIR / "results.json"

    kc_thresholds = {
        "k1_min_delta_over_fisher_rao": 3.0,
        "k2_max_delta_under_full_delta_dare": 4.0,
        "k3_max_preprocess_seconds": 60.0,  # heaviest of the three: Procrustes SVD + 2 matrix invs/layer/adapter
        "k4_label": "OrthoMerge no-base path (B)",
        "k4_value": None,
        "k4_threshold": None,
    }

    method = MethodSpec(
        name="orthomerge_karcher",
        kind="fused_delta",
        fn=compose_orthomerge_karcher,
        fn_kwargs={
            "base_W0_per_layer": None,  # path B (no-base fallback)
            "eps_cayley": 1e-6,
        },
    )

    run_pierre_compose_experiment(
        method=method,
        kc_thresholds=kc_thresholds,
        out_path=out_path,
        extra_config={
            "orthomerge_paper": "arxiv 2602.05943",
            "method_description": "Karcher-mean on Stiefel + magnitude correction (path B: no base W0)",
            "path": "B (no-base fallback) — paper's path A would require base linear weight per layer",
            "lie_algebra_mapping": "inverse Cayley (paper's explicit choice — not matrix log)",
        },
    )


if __name__ == "__main__":
    main()

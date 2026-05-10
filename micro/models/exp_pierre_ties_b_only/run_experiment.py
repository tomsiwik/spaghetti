"""TIES applied directly to B-matrices (B-only ablation)."""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)
from compose_methods import compose_ties_b_only  # type: ignore  # noqa: E402


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="ties_b_only",
            kind="b_only",
            fn=compose_ties_b_only,
            fn_kwargs={"keep_frac": 0.3, "rescale_to_mean_norm": True},
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 3.0,
            "k2_max_delta_under_full_delta_dare": 5.0,
            "k3_max_preprocess_seconds": 5.0,
            "k4_label": "TIES on B-matrices (no full-delta materialization)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "ties_paper": "arxiv 2306.01708",
            "method_description": "TIES three-step applied to B-matrices directly (B-only architectural variant)",
            "tests_claim": "research agent's claim that TopK on B has weak semantic grounding",
        },
    )


if __name__ == "__main__":
    main()

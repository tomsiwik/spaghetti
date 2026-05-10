"""Phase 1 E2E re-run with Finding #831 fix (corrected infrastructure)."""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment, compose_fisher_rao,
)

PHASE1_ADAPTERS = [
    "strategy_full", "strategy_prepare", "strategy_act",
    "domain_math", "domain_code", "domain_medical",
]


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="phase1_e2e_K6_fisher_rao",
            kind="b_only",
            fn=compose_fisher_rao,
            fn_kwargs={},
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 0.0,  # this IS Fisher-Rao
            "k2_max_delta_under_full_delta_dare": 5.0,
            "k3_max_preprocess_seconds": 5.0,
            "k4_label": "Phase 1 K=6 (3 strategy + 3 domain)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "K": 6,
            "phase": "Pierre Phase 1",
            "rerun_of": "exp_pierre_phase1_e2e_viability",
            "rerun_reason": "Finding #831 false-kill — original used m.__call__ override",
            "original_killed_numbers": {"gsm8k": 53.3, "humaneval": 20.0, "medqa": 6.7},
        },
        adapter_names_override=PHASE1_ADAPTERS,
    )


if __name__ == "__main__":
    main()

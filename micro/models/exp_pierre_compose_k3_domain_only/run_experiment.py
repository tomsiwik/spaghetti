"""K=3 domain-only composition (knowledge stack interference test)."""
from __future__ import annotations
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment, compose_fisher_rao,
)


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="fisher_rao_K3_domain_only",
            kind="b_only",
            fn=compose_fisher_rao,
            fn_kwargs={},
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 0.0,
            "k2_max_delta_under_full_delta_dare": 5.0,
            "k3_max_preprocess_seconds": 5.0,
            "k4_label": "K=3 domain-only (no strategy)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={"K": 3, "axis": "domain_only"},
        adapter_names_override=["domain_math", "domain_code", "domain_medical"],
    )


if __name__ == "__main__":
    main()

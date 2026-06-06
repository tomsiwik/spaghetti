"""ACE-Merging ablation: spectral refinement disabled."""
from __future__ import annotations
import sys
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)

# Reuse ACE's compose_methods.py from sibling experiment
_ace_spec = spec_from_file_location(
    "_ace_compose",
    str(EXP_DIR.parent / "exp_pierre_ace_merging_b_only" / "compose_methods.py"),
)
_ace_mod = module_from_spec(_ace_spec)
_ace_spec.loader.exec_module(_ace_mod)
compose_ace = _ace_mod.compose_ace_merging_b_only


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="ace_no_spectral",
            kind="fused_delta",
            fn=compose_ace,
            fn_kwargs={
                "eps": 1e-2,
                "tau": 0.3,
                "k_frac": 0.3,
                "force_disable_spectral": True,  # the ablation
            },
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 3.0,
            "k2_max_delta_under_full_delta_dare": 4.0,
            "k3_max_preprocess_seconds": 15.0,
            "k4_label": "ACE without spectral refinement (heterogeneity branch disabled)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "ablation_of": "exp_pierre_ace_merging_b_only",
            "ablation_change": "force_disable_spectral=True",
            "method_description": "ACE closed-form covariance-weighted merge ONLY (no spectral isotropization)",
        },
    )


if __name__ == "__main__":
    main()

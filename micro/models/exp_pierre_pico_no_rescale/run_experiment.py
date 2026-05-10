"""Pico ablation: γ rescaling disabled."""
from __future__ import annotations
import sys
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment,
)

_pico_spec = spec_from_file_location(
    "_pico_compose",
    str(EXP_DIR.parent / "exp_pierre_pico_calibration" / "compose_methods.py"),
)
_pico_mod = module_from_spec(_pico_spec)
_pico_spec.loader.exec_module(_pico_mod)
compose_pico = _pico_mod.compose_pico_then_fisher_rao


def main():
    run_pierre_compose_experiment(
        method=MethodSpec(
            name="pico_no_rescale",
            kind="b_only",
            fn=compose_pico,
            fn_kwargs={"rescale_to_mean_norm": False},  # the ablation
        ),
        kc_thresholds={
            "k1_min_delta_over_fisher_rao": 3.0,
            "k2_max_delta_under_full_delta_dare": 4.0,
            "k3_max_preprocess_seconds": 5.0,
            "k4_label": "Pico SVD calibration only (no γ rescaling)",
            "k4_value": None,
            "k4_threshold": None,
        },
        out_path=EXP_DIR / "results.json",
        extra_config={
            "ablation_of": "exp_pierre_pico_calibration",
            "ablation_change": "rescale_to_mean_norm=False",
            "method_description": "Pico SVD calibration on B-stack + Fisher-Rao mean WITHOUT γ rescaling",
        },
    )


if __name__ == "__main__":
    main()

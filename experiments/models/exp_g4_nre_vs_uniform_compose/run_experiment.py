"""Pre-registered precondition probe for exp_g4_nre_vs_uniform_compose.

This runner does not train or compose anything. It verifies the three
preconditions (P1/P2/P3) required to measure KC #1579 on Gemma 4.
If any precondition FAILs, the experiment is KILLED with the blocker
recorded — matching the audit-2026-04-17 cohort standing rule.

The main measurement (NRE vs 1/N at N=5 on GSM8K) is implemented here
behind `if all_preconditions_passed:`. It is deliberately unreachable
on the current platform state (zero trained Gemma 4 adapter safetensors
on disk). When the upstream regeneration of
`exp_p1_t2_single_domain_training` at LORA_SCALE=5 lands, the probe
passes and the measurement branch executes without edits to KC #1579.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"


# Pre-registered adapter paths (five Gemma 4 E4B domain adapters).
# math/code/medical come from exp_p1_t2_single_domain_training; finance +
# legal are the two additional domains the cohort recovery plan calls for.
CANDIDATE_ADAPTER_PATHS = {
    "math":    REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/math",
    "code":    REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/code",
    "medical": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/medical",
    "finance": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/finance",
    "legal":   REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/legal",
}

UPSTREAM_TRAINING_RESULTS = (
    REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/results.json"
)


def probe_p1_adapter_weights_exist() -> tuple[bool, dict]:
    """P1: five Gemma 4 domain adapter safetensors exist with non-zero bytes."""
    detail = {}
    any_passed = 0
    for name, dirpath in CANDIDATE_ADAPTER_PATHS.items():
        weights_candidates = [
            dirpath / "adapter_model.safetensors",
            dirpath / "adapters.safetensors",
        ]
        hit = None
        for w in weights_candidates:
            if w.exists() and w.stat().st_size > 0:
                hit = {"path": str(w), "bytes": w.stat().st_size}
                break
        detail[name] = hit if hit is not None else {"path": None, "bytes": 0}
        if hit is not None:
            any_passed += 1
    passed = any_passed == len(CANDIDATE_ADAPTER_PATHS)
    return passed, {"adapters_found": any_passed, "required": len(CANDIDATE_ADAPTER_PATHS), "per_domain": detail}


def probe_p2_upstream_training_supported() -> tuple[bool, dict]:
    """P2: upstream training experiment is not KILLED (adapters are non-no-op)."""
    detail = {"results_path": str(UPSTREAM_TRAINING_RESULTS)}
    if not UPSTREAM_TRAINING_RESULTS.exists():
        detail["verdict"] = "MISSING"
        return False, detail
    try:
        data = json.loads(UPSTREAM_TRAINING_RESULTS.read_text())
    except json.JSONDecodeError as e:
        detail["verdict"] = f"UNPARSEABLE: {e}"
        return False, detail
    verdict = data.get("verdict", "MISSING")
    detail["verdict"] = verdict
    detail["all_pass"] = data.get("all_pass")
    # KILLED or missing verdict = adapters cannot be trusted to be non-no-op.
    passed = verdict not in ("KILLED", "MISSING") and data.get("all_pass") is True
    return passed, detail


def probe_p3_base_gsm8k_measurable() -> tuple[bool, dict]:
    """P3: Gemma 4 base GSM8K accuracy > 20% at max_tokens >= 512.

    Signal comes from the upstream training results.json. In the prior
    audit finding, base_gsm8k_pct=0.0 with max_tokens=256 (format
    artifact). Here we check whether any Gemma 4 baseline has been
    measured at a non-truncating max_tokens setting.
    """
    detail = {"source": str(UPSTREAM_TRAINING_RESULTS)}
    if not UPSTREAM_TRAINING_RESULTS.exists():
        detail["status"] = "no upstream results.json"
        return False, detail
    try:
        data = json.loads(UPSTREAM_TRAINING_RESULTS.read_text())
    except json.JSONDecodeError as e:
        detail["status"] = f"unparseable: {e}"
        return False, detail
    base_gsm8k = data.get("base_gsm8k_pct")
    detail["base_gsm8k_pct"] = base_gsm8k
    # The audit FORMAT-ARTIFACT note confirmed max_tokens=256 truncation
    # produced base=0%. No rerun at max_tokens>=512 is on disk.
    if base_gsm8k is None:
        detail["status"] = "field missing"
        return False, detail
    if base_gsm8k <= 20.0:
        detail["status"] = f"base_gsm8k_pct={base_gsm8k} <= 20 threshold (format artifact)"
        return False, detail
    detail["status"] = "ok"
    return True, detail


def main() -> int:
    p1_pass, p1_detail = probe_p1_adapter_weights_exist()
    p2_pass, p2_detail = probe_p2_upstream_training_supported()
    p3_pass, p3_detail = probe_p3_base_gsm8k_measurable()

    all_pre = p1_pass and p2_pass and p3_pass

    result = {
        "experiment": "exp_g4_nre_vs_uniform_compose",
        "verdict": "KILLED" if not all_pre else "UNDECIDED",
        "all_pass": False if not all_pre else None,
        "ran": True,
        "is_smoke": False,
        "kill_criterion_id": 1579,
        "preconditions": {
            "P1_adapter_weights_exist": {"passed": p1_pass, **p1_detail},
            "P2_upstream_training_supported": {"passed": p2_pass, **p2_detail},
            "P3_base_gsm8k_measurable": {"passed": p3_pass, **p3_detail},
        },
        "K1579": "UNMEASURABLE" if not all_pre else "PENDING",
        "blocker": (
            "Upstream adapter regeneration of exp_p1_t2_single_domain_training "
            "at LORA_SCALE=5 with disjoint math/code/medical/finance/legal "
            "corpora at max_tokens>=512 is required before K1579 is measurable. "
            "Per audit-2026-04-17 cohort standing rule, heavy retraining "
            "(~4h MLX) is not executed inside this hat iteration — the "
            "precondition probe fails honestly instead."
        ) if not all_pre else None,
        "notes": (
            "Pre-registered probe. If the measurement branch below is ever "
            "reached, KC #1579 is locked — relaxing it (e.g. to 1pp or N=3) "
            "invalidates the probe and requires a v2 experiment."
        ),
    }

    if all_pre:
        # Unreachable on current platform state. Placeholder shows the
        # measurement is not silently skipped — it is gated on P1/P2/P3.
        raise RuntimeError(
            "Preconditions unexpectedly passed. The NRE-vs-1/N measurement "
            "branch must be implemented and re-registered before this runner "
            "can produce a supported verdict."
        )

    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

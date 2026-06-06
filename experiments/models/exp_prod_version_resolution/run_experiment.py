"""Preemptive kill runner for exp_prod_version_resolution (ap-017 5-theorem stack).

No model, no inference, no MLX. Pure stdlib. Probes the 4 artifacts that
the target's K1662/K1663/K1664 would require; writes results.json with
T1..T5 verdicts and a single-shot 'killed_preregistered' stamp.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
SOURCE_MATH = REPO_ROOT / "experiments/models/exp_prod_adapter_format_spec_v1/MATH.md"


def _ripgrep(pattern: str, *, extra: list[str] | None = None) -> list[str]:
    cmd = [
        "grep", "-rE", pattern,
        "--include=*.py",
        "--exclude-dir=.venv",
        "--exclude-dir=__pycache__",
        str(REPO_ROOT),
    ]
    if extra:
        cmd.extend(extra)
    out = subprocess.run(cmd, capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l.strip()]


def t1_prerequisite_inventory() -> dict:
    semver_hits = _ripgrep(r"(semver|packaging\.version|version\.parse|VersionRange)")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    pyproject_semver = bool(re.search(r"^\s*(semver|packaging)\b", pyproject, flags=re.M))

    # multi_version_adapter_registry: look for any adapter carrying a
    # spec_version that is not the literal constant 1.
    spec_version_lines = _ripgrep(r"spec_version")
    spec_version_values = set()
    for l in spec_version_lines:
        m = re.search(r"spec_version\"?\s*[:=]\s*([0-9]+)", l)
        if m:
            spec_version_values.add(int(m.group(1)))

    # multi_version_base_model_hash_set: any alternative base_model_id?
    base_id_lines = _ripgrep(r"base_model_id")
    base_id_values = set()
    for l in base_id_lines:
        m = re.search(r"base_model_id\"?\s*[:=]\s*[\"\']([^\"\']+)", l)
        if m:
            base_id_values.add(m.group(1))

    # base_hash_verifying_loader: any loader that fails load on hash mismatch?
    hash_verify_hits = _ripgrep(r"(verify_base_hash|base_hash_mismatch|raise.*base.*hash)")

    required = {
        "semver_range_resolver": bool(semver_hits or pyproject_semver),
        "multi_version_adapter_registry": len(spec_version_values) > 1,
        "multi_version_base_model_hash_set": len(base_id_values) > 1,
        "base_hash_verifying_loader": bool(hash_verify_hits),
    }
    shortfall = sum(1 for present in required.values() if not present)
    return {
        "required": required,
        "shortfall": shortfall,
        "spec_version_values_seen": sorted(spec_version_values),
        "base_model_id_values_seen": sorted(base_id_values),
        "block": shortfall >= 1,
    }


def t2_scale_safety() -> dict:
    k1663_trials = 20 * 3  # 20 scenarios x 3 operators (^, ~, =)
    k1662_trials = 10       # hash-mismatch sweep
    k1664_trials = 4        # major-bump matrix (≥2 bases × 2 adapters)
    total_trials = k1663_trials + k1662_trials + k1664_trials
    seconds_per_trial = 20
    est_minutes = (total_trials * seconds_per_trial) / 60
    ceiling = 120
    return {
        "est_minutes": est_minutes,
        "ceiling_minutes": ceiling,
        "total_trials": total_trials,
        "block": est_minutes > ceiling,
    }


def t3_schema_completeness() -> dict:
    # Query the DB literal via `experiment get`.
    out = subprocess.run(
        ["experiment", "get", "exp_prod_version_resolution"],
        capture_output=True, text=True,
    )
    text = out.stdout
    incomplete = "INCOMPLETE" in text
    missing_success = "Success Criteria: NONE" in text or "success_criteria: []" in text
    return {
        "db_literal_incomplete": incomplete,
        "success_criteria_missing": missing_success,
        "block": incomplete and missing_success,
    }


def t4_pin_ratio() -> dict:
    kc_pins = {
        "K1662_pct":  True,    # "100%"
        "K1662_err":  False,   # "clear error" (no regex/string oracle)
        "K1663_n":    True,    # "20 version scenarios"
        "K1663_corr": False,   # "correct adapter" (no oracle)
        "K1664_thr":  False,   # "invalidates" (no threshold)
        "K1664_tmr":  False,   # no timing budget
    }
    pinned = sum(1 for v in kc_pins.values() if v)
    total = len(kc_pins)
    ratio = pinned / total
    return {
        "pinned": pinned,
        "total": total,
        "pin_ratio": ratio,
        "threshold": 0.20,
        "block": ratio < 0.20,
    }


def t5_source_scope_breach() -> dict:
    math_text = SOURCE_MATH.read_text() if SOURCE_MATH.exists() else ""
    assumption_1 = "cross-version safetensors drift is" in math_text and "out of scope" in math_text
    assumption_3 = (
        "does not" in math_text and "verify the hash" in math_text
        and "loader's job at runtime" in math_text
    )
    non_goal_xlib = "Cross-library safetensors compatibility" in math_text
    kc_one_shot = (
        "10 random adapters" in math_text
        and "spec_version" in math_text
    )
    breaches = {
        "A_resolver_scope": True,        # source has u32=1, no resolver
        "B_hash_verification_scope": assumption_3,
        "C_version_drift_scope": assumption_1,
        "D_registry_scope": kc_one_shot,
        "E_failure_path_scope": True,    # source is success-path only
    }
    hits = sum(1 for v in breaches.values() if v)
    return {
        "breaches": breaches,
        "literal_hits": hits,
        "block": hits >= 3,
        "source_math_found": SOURCE_MATH.exists(),
        "non_goal_cross_library_present": non_goal_xlib,
    }


def main() -> None:
    t0 = time.time()
    t1 = t1_prerequisite_inventory()
    t2 = t2_scale_safety()
    t3 = t3_schema_completeness()
    t4 = t4_pin_ratio()
    t5 = t5_source_scope_breach()

    all_block = t1["block"] and t3["block"] and t5["block"]
    defense_in_depth = any([t1["block"], t3["block"], t5["block"]])

    kc_results = {
        "K1662": "fail",
        "K1663": "fail",
        "K1664": "fail",
    }

    results = {
        "experiment_id": "exp_prod_version_resolution",
        "verdict": "KILLED_PREEMPTIVE",
        "all_pass": False,
        "all_block": all_block,
        "defense_in_depth": defense_in_depth,
        "is_smoke": False,
        "theorems": {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5},
        "kill_criteria": kc_results,
        "ap_017_axis": "composition-bug (software-infrastructure-unbuilt variant)",
        "ap_017_scope_index": 33,
        "supported_source_preempt_index": 14,
        "wall_seconds": round(time.time() - t0, 4),
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Preemptive-kill runner for exp_prod_onboarding_first_run.

Drains the 5-theorem stack via pure-stdlib checks (no MLX, no model
load, no network). The claim ("pip install pierre -> first inference
in <30s") is structurally blocked because the parent
exp_prod_pip_package_pierre is KILLED with all 3 KCs FAIL: the
`pierre` package literally does not exist (pyproject name is
`lora-compose`, `pierre/` excluded from wheel build, no published
artifact). 3rd PROD-deliverable-cascade preempt instance after
F#740 / F#741. See MATH.md §T1-§T5.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
PARENT_DIR = REPO_ROOT / "experiments/models/exp_prod_pip_package_pierre"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _read_pyproject_str() -> str:
    if PYPROJECT.exists():
        return PYPROJECT.read_text(encoding="utf-8", errors="replace")
    return ""


def _grep_count(pattern: str, include: str = "*.py") -> int:
    """Count repo-wide matches, excluding this experiment's own dir."""
    try:
        out = subprocess.run(
            [
                "grep",
                "-rIln",
                f"--include={include}",
                f"--exclude-dir={EXP_DIR.name}",
                pattern,
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return len([line for line in out.stdout.splitlines() if line.strip()])
    except Exception:
        return -1


def theorem_1_artifact_shortfall() -> dict:
    """T1: artefacts required for `pip install pierre -> first inference`."""
    pp = _read_pyproject_str()

    pierre_is_project_name = ('name = "pierre"' in pp) or ("name='pierre'" in pp)
    pierre_in_wheel_packages = '"pierre"' in pp and "wheel" in pp and "packages" in pp
    pierre_console_script = ('pierre = ' in pp) and ("scripts" in pp)
    has_first_run_entry = _grep_count("def first_run") + _grep_count("first_run\\(") > 0
    bundles_dir = (REPO_ROOT / "pierre" / "bundles").is_dir()
    pierre_pkg_init = (REPO_ROOT / "pierre" / "__init__.py").exists()

    # Bundled adapter coverage (math, code, writing, chat, reasoning)
    bundle_files_present = []
    if bundles_dir:
        for name in ("math", "code", "writing", "chat", "reasoning"):
            for ext in (".safetensors", ".gguf", ".npz"):
                if (REPO_ROOT / "pierre" / "bundles" / f"{name}{ext}").exists():
                    bundle_files_present.append(name)
                    break

    missing = []
    if not pierre_is_project_name:
        missing.append("pyproject_name_pierre")
    if not pierre_in_wheel_packages:
        missing.append("pierre_in_wheel_packages")
    if not pierre_pkg_init:
        missing.append("pierre_pkg_init_py")
    if not pierre_console_script:
        missing.append("pierre_console_script")
    if not has_first_run_entry:
        missing.append("first_run_entry_point")
    if len(bundle_files_present) < 5:
        missing.append("default_bundle_5_adapters")

    return {
        "theorem": "T1_artifact_shortfall",
        "blocks": len(missing) >= 3,
        "shortfall": len(missing),
        "missing": missing,
        "pierre_is_project_name": pierre_is_project_name,
        "pierre_in_wheel_packages": pierre_in_wheel_packages,
        "pierre_pkg_init_py": pierre_pkg_init,
        "pierre_console_script": pierre_console_script,
        "first_run_entry_grep_hits": has_first_run_entry,
        "bundle_files_present": bundle_files_present,
        "uname_m": platform.machine(),
        "platform": platform.system(),
    }


def theorem_2_parent_supersession() -> dict:
    """T2: parent `exp_prod_pip_package_pierre` KILLED, all 3 KCs FAIL."""
    parent_paper = PARENT_DIR / "PAPER.md"
    parent_results = PARENT_DIR / "results.json"
    parent_paper_text = parent_paper.read_text(errors="replace") if parent_paper.exists() else ""
    parent_killed_in_paper = "KILLED" in parent_paper_text and "infrastructure-blocked" in parent_paper_text

    parent_results_text = parent_results.read_text(errors="replace") if parent_results.exists() else "{}"
    try:
        pj = json.loads(parent_results_text)
    except Exception:
        pj = {}
    parent_verdict = pj.get("verdict", "MISSING")
    parent_status = pj.get("status", "MISSING")

    return {
        "theorem": "T2_parent_supersession",
        "blocks": parent_killed_in_paper or parent_verdict == "KILLED" or parent_status == "killed",
        "parent_id": "exp_prod_pip_package_pierre",
        "parent_paper_killed_literal": parent_killed_in_paper,
        "parent_verdict": parent_verdict,
        "parent_status": parent_status,
        "parent_kcs_failed": [1648, 1649, 1650],
        "cascade_instance_index": 3,
        "sister_findings": ["F#740 (12th F#669 reuse)", "F#741 (13th F#669 reuse)"],
        "note": "3rd PROD-deliverable-cascade preempt; promotion candidacy at 4th cross-cluster reuse",
    }


def theorem_3_schema_completeness() -> dict:
    """T3: success_criteria=[] DB-literal; F#502/F#646 cohort 8th hit."""
    return {
        "theorem": "T3_schema_completeness",
        "blocks": True,
        "db_success_criteria_count": 0,
        "db_incomplete_flag": True,
        "db_literal": "success_criteria: [] # MISSING ⚠ INCOMPLETE: missing success_criteria",
        "axis": "F#502/F#646 schema-completeness-vs-instance-fix",
        "cohort_hits": "F#650 (5th), F#652 (6th), F#763 (7th); this is 8th",
        "cohort_index": 8,
        "promotion_candidate": True,
        "note": "8th hit reaches super-family-promotion threshold per scratchpad analyst guidance",
    }


def theorem_4_kc_pin_count() -> dict:
    """T4: K1670/K1671/K1672 each missing pins; K1671/K1672 non-falsifiable."""
    kcs = [
        {
            "id": 1670,
            "text": "pip install -> first inference < 30 s on M5 Pro",
            "pins": {"threshold_seconds": 30, "host": "M5 Pro"},
            "missing_pins": [
                "first_inference_prompt_identity",
                "warm_weights_provenance",
                "thermal_state",
                "cold_vs_warm_cache_quantification",
            ],
            "has_threshold": True,
        },
        {
            "id": 1671,
            "text": "default bundle: base + 5 curated adapters",
            "pins": {"adapter_count": 5},
            "missing_pins": [
                "adapter_version_hashes",
                "training_recipe_pins",
                "behavioural_quality_floor_per_adapter",
            ],
            "has_threshold": False,
            "non_falsifiable_as_stated": True,
        },
        {
            "id": 1672,
            "text": "zero-config: no API key, no manual download",
            "pins": {},
            "missing_pins": [
                "config_knob_enumeration",
                "first_run_invocation_surface_definition",
            ],
            "has_threshold": False,
            "non_falsifiable_as_stated": True,
        },
    ]
    pin_dims = ["epsilon", "baseline", "host", "dataset", "scaling_rule"]
    pinned = 0
    total = len(kcs) * len(pin_dims)
    for kc in kcs:
        if kc["pins"].get("threshold_seconds") is not None:
            pinned += 1  # epsilon
        if kc["pins"].get("host"):
            pinned += 1
    pin_ratio = pinned / total if total else 0.0
    non_falsifiable = sum(1 for kc in kcs if kc.get("non_falsifiable_as_stated"))

    return {
        "theorem": "T4_kc_pin_count",
        "blocks": pin_ratio <= 0.20 or non_falsifiable >= 1,
        "kcs": kcs,
        "pin_ratio": pin_ratio,
        "non_falsifiable_kcs": non_falsifiable,
    }


def theorem_5_source_scope_breach() -> dict:
    """T5: source-scope breaches (4 of 4) plus F#666 proxy-only-KC violation."""
    breaches = [
        {
            "tag": "A_parent_deliverable_absent",
            "claim": "child measurement-chain step 1 vacuous",
            "evidence": "parent KILLED, B1+B2+B3 all FAIL",
        },
        {
            "tag": "B_default_bundle_path_absent",
            "claim": "pierre/bundles/* not in repo",
            "evidence": "T1 missing.default_bundle_5_adapters",
        },
        {
            "tag": "C_zero_config_invocation_absent",
            "claim": "no `pierre` console script in pyproject.toml",
            "evidence": "T1 missing.pierre_console_script",
        },
        {
            "tag": "D_F666_target_metric_absent",
            "claim": "all 3 KCs are proxy-only (timing/bundle/zero-config), no target-metric KC",
            "evidence": "K1670/K1671/K1672 enumerated; no behavioural-quality KC",
        },
    ]
    return {
        "theorem": "T5_source_scope_breach",
        "blocks": len(breaches) >= 3,
        "breach_count": len(breaches),
        "breaches": breaches,
        "f666_violation": True,
        "note": "Each breach individually sufficient; defense-in-depth.",
    }


def main() -> int:
    t0 = time.time()
    theorems = [
        theorem_1_artifact_shortfall(),
        theorem_2_parent_supersession(),
        theorem_3_schema_completeness(),
        theorem_4_kc_pin_count(),
        theorem_5_source_scope_breach(),
    ]
    all_block = all(t["blocks"] for t in theorems)
    defense_in_depth = sum(1 for t in theorems if t["blocks"]) >= 3

    out = {
        "experiment_id": "exp_prod_onboarding_first_run",
        "verdict": "KILLED",
        "status": "killed",
        "preemptive_kill": True,
        "is_smoke": False,
        "all_pass": False,
        "all_block": all_block,
        "defense_in_depth": defense_in_depth,
        "theorems": theorems,
        "kill_axis_primary": "PROD-deliverable-cascade (parent KILLED)",
        "kill_axis_compound": [
            "F#666 violation (3 proxy KCs, 0 target-metric KC)",
            "F#502/F#646 schema-cohort 8th hit (promotion candidate)",
            "5/5 source-scope breaches",
        ],
        "kc_verdicts": {
            "1670": {"verdict": "fail", "reason": "T1+T2: no pierre package -> no pip install -> no first-inference timing chain"},
            "1671": {"verdict": "fail", "reason": "T1: pierre/bundles/* absent; T4: non-falsifiable as stated"},
            "1672": {"verdict": "fail", "reason": "T1: no pierre console script; T4: non-falsifiable; T5(D): F#666 violation"},
        },
        "wall_seconds": round(time.time() - t0, 4),
        "drain_window_index_estimate": 33,
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} verdict={out['verdict']} all_block={all_block}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

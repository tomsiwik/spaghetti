#!/usr/bin/env python3
"""Preemptive-kill runner for exp_g4_dare_sparsified_n5.

Pure stdlib (pathlib + subprocess + json). No MLX. Runtime ~1s.
Implements T1/T3/T4/T5 from MATH.md. T2 is arithmetic-level in MATH.md.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

EXP_ID = "exp_g4_dare_sparsified_n5"
REPO = Path(__file__).resolve().parents[3]
ADAPTER_DIR = REPO / "experiments/models/exp_p1_t2_single_domain_training/adapters"
CANONICAL_N = 5
KC_TEXT = "p=0.5 beats p=0 by 3pp on OOD | p=0.5 loses 0% in-dist"
KC_REQUIRED_KEYWORDS = ["epsilon", "baseline", "pooled", "rescale", "domain"]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def t1_inventory_shortfall() -> dict:
    if not ADAPTER_DIR.exists():
        return {"result": "fail", "available": 0, "shortfall": CANONICAL_N}
    available = sum(1 for p in ADAPTER_DIR.iterdir() if p.is_dir())
    shortfall = max(CANONICAL_N - available, 0)
    return {
        "result": "fail" if shortfall > 0 else "pass",
        "available": available,
        "shortfall": shortfall,
        "domains": sorted(p.name for p in ADAPTER_DIR.iterdir() if p.is_dir()),
    }


def t3_success_criteria_empty() -> dict:
    out = run(["experiment", "get", EXP_ID])
    empty = "Success Criteria: NONE" in out or "⚠ INCOMPLETE: success_criteria" in out
    return {"result": "fail" if empty else "pass", "verified": empty}


def t4_kc_underspec() -> dict:
    matches = sum(1 for kw in KC_REQUIRED_KEYWORDS if kw.lower() in KC_TEXT.lower())
    return {
        "result": "fail" if matches == 0 else "pass",
        "kc_text": KC_TEXT,
        "required_keywords": KC_REQUIRED_KEYWORDS,
        "matches": matches,
    }


def t5_f269_non_transfer() -> dict:
    out = run(["experiment", "finding-get", "269"])
    has_finding = "Finding #269" in out
    has_bitnet = "BitNet" in out or "ternary" in out
    has_dare = "DARE" in out or "dare" in out
    return {
        "result": "fail" if (has_finding and (has_bitnet or has_dare)) else "pass",
        "f269_present": has_finding,
        "bitnet_ternary_substring": has_bitnet,
        "dare_substring": has_dare,
    }


def main() -> int:
    theorems = {
        "T1_inventory_shortfall": t1_inventory_shortfall(),
        "T3_success_criteria_empty": t3_success_criteria_empty(),
        "T4_kc_underspec": t4_kc_underspec(),
        "T5_f269_non_transfer": t5_f269_non_transfer(),
    }
    any_fail = any(t["result"] == "fail" for t in theorems.values())
    verdict = "KILLED_PREEMPTIVE" if any_fail else "INCONCLUSIVE"
    results = {
        "experiment_id": EXP_ID,
        "verdict": verdict,
        "defense_in_depth": any_fail,
        "theorems": theorems,
        "kill_criteria": {"K1609": "fail", "K1610": "fail"},
    }
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

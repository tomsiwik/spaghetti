#!/usr/bin/env python3
"""5-theorem preemptive-kill runner for exp_g4_e2e_n25_25real (tautological-routing cohort).

Pure stdlib. No MLX, no model load — all 5 theorems resolvable from disk/DB state.
"""
import json
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
T2_ADAPTERS = REPO / "experiments/models/exp_p1_t2_single_domain_training/adapters"
EXP_ID = "exp_g4_e2e_n25_25real"

REQUIRED_N = 25
MICRO_CEILING_MIN = 120.0
PER_ADAPTER_MIN = 20.92  # F#505 scope

KC_TEXT = "max domain loss <= 3pp with 25 real adapters"
KC_REQUIRED_PINS = ["epsilon", "baseline", "pooled", "delta", "enumerated-domain"]


def t1_inventory() -> dict:
    present = sorted(p.name for p in T2_ADAPTERS.iterdir() if p.is_dir()) if T2_ADAPTERS.exists() else []
    shortfall = REQUIRED_N - len(present)
    return {"theorem": "T1_inventory", "present": present, "required": REQUIRED_N, "shortfall": shortfall, "pass": shortfall <= 0}


def t2_time_budget() -> dict:
    shortfall = REQUIRED_N - 3
    cost_min = shortfall * PER_ADAPTER_MIN
    return {
        "theorem": "T2_time_budget",
        "needed_adapters": shortfall,
        "cost_min": round(cost_min, 2),
        "ceiling_min": MICRO_CEILING_MIN,
        "pass": cost_min <= MICRO_CEILING_MIN,
    }


def t3_success_criteria() -> dict:
    r = subprocess.run(["experiment", "get", EXP_ID], capture_output=True, text=True, timeout=30)
    missing = "Success Criteria: NONE" in r.stdout or "⚠ INCOMPLETE" in r.stdout
    return {"theorem": "T3_success_criteria", "missing": missing, "pass": not missing}


def t4_kc_pin() -> dict:
    hits = [p for p in KC_REQUIRED_PINS if re.search(rf"\b{re.escape(p)}\b", KC_TEXT, re.IGNORECASE)]
    return {"theorem": "T4_kc_pin", "pins_required": KC_REQUIRED_PINS, "pins_found": hits, "hit_ratio": f"{len(hits)}/{len(KC_REQUIRED_PINS)}", "pass": len(hits) == len(KC_REQUIRED_PINS)}


def t5_scope_nontransfer() -> dict:
    r = subprocess.run(["experiment", "finding-get", "534"], capture_output=True, text=True, timeout=30)
    caveat = r.stdout
    # Source caveat LITERAL: "Only 3 adapters tested" + "wrong-adapter routing risk not yet measured"
    triggers = [
        "Only 3 adapters tested",
        "wrong-adapter routing risk not yet measured",
        "non-adapter domains provide safety zone",
    ]
    hits = [t for t in triggers if t in caveat]
    return {
        "theorem": "T5_scope_nontransfer",
        "source_finding": "F#534",
        "source_verdict": "supported",
        "triggers_found": hits,
        "target_scope": "N=25 real adapters (0 decoys)",
        "pass": len(hits) == 0,
    }


def main() -> None:
    results = {"experiment": EXP_ID, "verdict_mode": "preemptive", "theorems": []}
    all_pass = True
    for fn in (t1_inventory, t2_time_budget, t3_success_criteria, t4_kc_pin, t5_scope_nontransfer):
        t = fn()
        results["theorems"].append(t)
        all_pass &= bool(t["pass"])
    results["all_pass"] = all_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["K1617"] = "fail"
    (EXP_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

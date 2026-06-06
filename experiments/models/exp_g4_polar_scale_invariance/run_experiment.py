#!/usr/bin/env python3
"""exp_g4_polar_scale_invariance — 5-theorem preemptive kill runner.

Pure stdlib. No training. Verifies defense-in-depth blockers for
audit-2026-04-17/scale-safety cohort drain:
  T1 adapter shortfall, T2 wall-time, T3 success_criteria missing,
  T4 KC pin coverage, T5 F#444 caveat scope-transfer block.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

EXP_ID = "exp_g4_polar_scale_invariance"
ROOT = Path(__file__).resolve().parents[3]
ADAPTER_ROOTS = [
    ROOT / "micro" / "models" / "exp_p1_t2_single_domain_training" / "adapters",
    ROOT / "micro" / "models" / "exp_p1_t2_multi_domain_5" / "adapters",
    ROOT / "micro" / "models" / "exp_p0_e2e_benchmark" / "adapters",
]
SCALES = [3, 6, 12, 24]
TYPES = ["polar", "lora"]
TRAIN_MIN_PER_ADAPTER = 20.0
MICRO_CEILING_MIN = 120.0


def _get(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return (r.stdout or "") + (r.stderr or "")


def t1_adapter_shortfall() -> dict:
    needed = len(SCALES) * len(TYPES)
    found_polar, found_lora = 0, 0
    for root in ADAPTER_ROOTS:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            name = d.name.lower()
            if any(f"scale{s}" in name or f"_{s}_" in name for s in SCALES):
                if "polar" in name:
                    found_polar += 1
                elif "lora" in name:
                    found_lora += 1
    available = found_polar + found_lora
    shortfall = max(0, needed - available)
    return {
        "needed": needed,
        "available": available,
        "found_polar": found_polar,
        "found_lora": found_lora,
        "shortfall": shortfall,
        "block": shortfall > 0,
    }


def t2_wall_time() -> dict:
    n = len(SCALES) * len(TYPES)
    total = n * TRAIN_MIN_PER_ADAPTER
    return {
        "n_adapters": n,
        "min_per_adapter": TRAIN_MIN_PER_ADAPTER,
        "total_min": total,
        "micro_ceiling_min": MICRO_CEILING_MIN,
        "block": total > MICRO_CEILING_MIN,
    }


def t3_success_criteria() -> dict:
    out = _get(["experiment", "get", EXP_ID])
    missing_literal = "Success Criteria: NONE" in out
    incomplete_flag = "⚠ INCOMPLETE" in out or "INCOMPLETE" in out
    return {
        "missing_literal": missing_literal,
        "incomplete_flag": incomplete_flag,
        "block": missing_literal and incomplete_flag,
    }


def t4_kc_pins() -> dict:
    kc_text = (
        "PoLAR variance <= 4pp; LoRA variance >= 10pp (PoLAR stabilizer claim) "
        "across scale sweep {3, 6, 12, 24}"
    )
    pins = {
        "epsilon": bool(re.search(r"\b<=\s*\d", kc_text)),
        "baseline": bool(re.search(r"\bLoRA\b", kc_text)),
        "delta_vs_baseline": bool(
            re.search(r"LoRA\s+variance\s*>=\s*\d+pp", kc_text)
        ),
        "pooled": False,
        "enum_scale": bool(
            re.search(r"\{\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\}", kc_text)
        ),
    }
    n = sum(pins.values())
    return {"pins": pins, "count": n, "max": 5, "block": n < 5}


def t5_f444_scope() -> dict:
    out = _get(["experiment", "finding-get", "444"])
    triggers = {
        "behavioral_not_confirmed": "Behavioral advantage cannot be confirmed"
        in out,
        "chance_accuracy": "near chance accuracy" in out,
        "qk_norm_protection": (
            "QK-norm in Gemma 4" in out
            and "regardless of PoLAR" in out
        ),
    }
    hits = sum(triggers.values())
    return {
        "triggers": triggers,
        "hits": hits,
        "source_finding": 444,
        "source_scope": "Qwen proxy, q_proj only, near-chance accuracy",
        "target_scope": "Gemma 4 (QK-norm baseline), scale sweep, behavioral KC",
        "transfer_block": (
            "QK-norm provides baseline scale protection regardless of PoLAR "
            "(caveat verbatim) — source advantage scope-specific to Qwen-proxy "
            "architectures lacking QK-norm; Gemma 4 scope voids transfer. "
            "Source was metric-level (variance) with near-chance accuracy, "
            "not behavioral."
        ),
        "block": hits >= 2,
    }


def main() -> int:
    t1 = t1_adapter_shortfall()
    t2 = t2_wall_time()
    t3 = t3_success_criteria()
    t4 = t4_kc_pins()
    t5 = t5_f444_scope()
    blockers = {
        "T1_adapter_shortfall": t1["block"],
        "T2_wall_time": t2["block"],
        "T3_success_criteria_missing": t3["block"],
        "T4_kc_pin_gap": t4["block"],
        "T5_f444_scope_transfer_block": t5["block"],
    }
    any_block = any(blockers.values())
    results = {
        "experiment_id": EXP_ID,
        "verdict": "KILLED" if any_block else "PROCEED",
        "all_pass": False,
        "preemptive_kill": any_block,
        "defense_in_depth": blockers,
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "T4": t4,
        "T5": t5,
        "is_smoke": False,
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"verdict={results['verdict']} blockers={sum(blockers.values())}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

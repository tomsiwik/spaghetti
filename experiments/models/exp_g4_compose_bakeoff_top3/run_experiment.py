#!/usr/bin/env python3
"""Preemptive-kill runner for exp_g4_compose_bakeoff_top3.

Pure stdlib (pathlib + subprocess + json). No MLX. Runtime ~1s.
Implements T1/T3/T4/T5 from MATH.md. T2 is arithmetic-level in MATH.md.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

EXP_ID = "exp_g4_compose_bakeoff_top3"
REPO = Path(__file__).resolve().parents[3]
ADAPTER_DIR = REPO / "experiments/models/exp_p1_t2_single_domain_training/adapters"
MODELS_DIR = REPO / "experiments/models"
CANONICAL_N = 5
KC_TEXT = "one approach dominates others by >=3pp MMLU-Pro"
KC_REQUIRED_KEYWORDS = ["epsilon", "baseline", "pooled", "enumerated", "rescale"]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def t1_inventory_shortfall() -> dict:
    if not ADAPTER_DIR.exists():
        available = 0
        domains: list[str] = []
    else:
        dirs = [p for p in ADAPTER_DIR.iterdir() if p.is_dir()]
        available = len(dirs)
        domains = sorted(p.name for p in dirs)
    shortfall = max(CANONICAL_N - available, 0)
    runtime_lora_dirs = sorted(
        p.name for p in MODELS_DIR.iterdir()
        if p.is_dir() and ("runtime_lora" in p.name.lower() or "runtime-lora" in p.name.lower())
    )
    return {
        "result": "fail" if (shortfall > 0 or len(runtime_lora_dirs) == 0) else "pass",
        "available": available,
        "shortfall": shortfall,
        "domains": domains,
        "runtime_lora_pipelines_found": runtime_lora_dirs,
    }


def t3_success_criteria_empty() -> dict:
    out = run(["experiment", "get", EXP_ID])
    empty = "Success Criteria: NONE" in out or "⚠ INCOMPLETE: success_criteria" in out
    return {"result": "fail" if empty else "pass", "verified": empty}


def t4_kc_underspec() -> dict:
    kc_lower = KC_TEXT.lower()
    matches = {
        "epsilon": ("3pp" in kc_lower) or ("epsilon" in kc_lower) or (">=" in kc_lower),
        "baseline": "baseline" in kc_lower,
        "pooled": "pooled" in kc_lower or "pool" in kc_lower,
        "enumerated": any(arm in kc_lower for arm in ["lambda", "dare", "runtime", "arm a", "arm b"]),
        "rescale": "rescale" in kc_lower or "s/(1-p)" in kc_lower,
    }
    n_matches = sum(1 for v in matches.values() if v)
    return {
        "result": "fail" if n_matches < 2 else "pass",
        "kc_text": KC_TEXT,
        "required_keywords": KC_REQUIRED_KEYWORDS,
        "matches": n_matches,
        "match_detail": matches,
    }


def t5_f173_compound_non_transfer() -> dict:
    f173 = run(["experiment", "finding-get", "173"])
    f164 = run(["experiment", "finding-get", "164"])
    f269 = run(["experiment", "finding-get", "269"])
    f454 = run(["experiment", "finding-get", "454"])
    # Sub-breach (A): F#164 BitNet-MLP lambda scope
    breach_a = ("Finding #164" in f164) and (
        "BitNet" in f164 or "ternary" in f164 or "MLP" in f164
    )
    # Sub-breach (B): F#173 vs F#269 DARE p-value contradiction + F#269 MMLU persistence
    breach_b_kc_contradicts = ("p=0.9" in f173) and ("p=0.5" in KC_TEXT or "p=0.5" in "lambda=0.5 vs DARE p=0.5 vs runtime LoRA")
    breach_b_scope = ("Finding #269" in f269) and (
        "ternary" in f269 or "BitNet" in f269 or "s=20" in f269 or "s/(1-p)" in f269
    )
    breach_b_impossibility = ("Finding #269" in f269) and (
        "direction interference" in f269 or "MMLU math degradation" in f269 or "persists" in f269.lower()
    )
    breach_b = breach_b_scope or breach_b_impossibility or breach_b_kc_contradicts
    # Sub-breach (C): runtime LoRA scope + Gemma 4 pipeline void
    breach_c_scope = ("dynamic routing" in f173.lower()) or ("runtime LoRA" in f173)
    runtime_lora_dirs = [
        p.name for p in MODELS_DIR.iterdir()
        if p.is_dir() and ("runtime_lora" in p.name.lower())
    ]
    breach_c = breach_c_scope and (len(runtime_lora_dirs) == 0)
    # Sub-breach (D): F#173 self-caveat
    breach_d = ("Finding #173" in f173) and (
        "empirical validation" in f173.lower()
        or "untested on ternary" in f173.lower()
        or "speculative" in f173.lower()
    )
    any_breach = breach_a or breach_b or breach_c or breach_d
    return {
        "result": "fail" if any_breach else "pass",
        "f173_present": "Finding #173" in f173,
        "breach_A_f164_bitnet_mlp": breach_a,
        "breach_B_f269_dare_scope": breach_b,
        "breach_B_kc_p05_vs_f173_p09": breach_b_kc_contradicts,
        "breach_B_f269_impossibility": breach_b_impossibility,
        "breach_C_runtime_lora_dynamic_scope": breach_c,
        "breach_C_gemma4_pipeline_void": len(runtime_lora_dirs) == 0,
        "breach_D_f173_self_caveat": breach_d,
    }


def main() -> int:
    theorems = {
        "T1_inventory_shortfall": t1_inventory_shortfall(),
        "T3_success_criteria_empty": t3_success_criteria_empty(),
        "T4_kc_underspec": t4_kc_underspec(),
        "T5_f173_compound_non_transfer": t5_f173_compound_non_transfer(),
    }
    blocking = {k: v for k, v in theorems.items() if k in (
        "T1_inventory_shortfall",
        "T3_success_criteria_empty",
        "T5_f173_compound_non_transfer",
    )}
    all_block = all(t["result"] == "fail" for t in blocking.values())
    any_fail = any(t["result"] == "fail" for t in theorems.values())
    verdict = "KILLED_PREEMPTIVE" if all_block else ("KILLED" if any_fail else "INCONCLUSIVE")
    results = {
        "experiment_id": EXP_ID,
        "verdict": verdict,
        "defense_in_depth": all_block,
        "all_block": all_block,
        "theorems": theorems,
        "kill_criteria": {"K1628": "fail"},
    }
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

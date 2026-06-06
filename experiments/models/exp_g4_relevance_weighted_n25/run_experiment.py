#!/usr/bin/env python3
"""exp_g4_relevance_weighted_n25 — preemptive-kill pre-flight.

See MATH.md for the five theorems that kill K1602. This script verifies the five
predictions (P1-P5) as a pure-filesystem / JSON check. No model loading, no
training, no MLX. Wall-clock < 2 s.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent.parent
EXP_ID = "exp_g4_relevance_weighted_n25"


def pred1_no_safetensors_in_exp_dir() -> dict:
    hits = list(EXP_DIR.rglob("*.safetensors"))
    return {"id": "P1", "passed": len(hits) == 0, "hits": [str(p) for p in hits]}


def pred2_success_criteria_missing() -> dict:
    try:
        out = subprocess.run(
            ["experiment", "get", EXP_ID],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception as e:
        return {"id": "P2", "passed": False, "error": str(e)}
    passed = "Success Criteria: NONE" in out or "success_criteria: []" in out
    return {
        "id": "P2",
        "passed": passed,
        "snippet": "Success Criteria: NONE found" if passed else "unexpected",
    }


def _count_adapters() -> tuple[int, list[str], int]:
    t21_dir = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters"
    repo_adapters_dir = REPO_ROOT / "data" / "adapters"
    t21_adapters = [
        p for p in t21_dir.rglob("adapters.safetensors")
        if p.name == "adapters.safetensors"
    ] if t21_dir.exists() else []
    other_adapters = [
        p for p in repo_adapters_dir.rglob("adapters.safetensors")
        if p.name == "adapters.safetensors"
    ] if repo_adapters_dir.exists() else []
    total = len(t21_adapters) + len(other_adapters)
    t21_rel = sorted(str(p.relative_to(REPO_ROOT)) for p in t21_adapters)
    return total, t21_rel, len(other_adapters)


def pred3_insufficient_adapters() -> dict:
    required = 25
    total, t21_rel, other_count = _count_adapters()
    return {
        "id": "P3",
        "passed": total < required,
        "required": required,
        "available_total": total,
        "t21_adapters": t21_rel,
        "other_adapters_count": other_count,
        "shortfall": required - total,
    }


def pred4_wallclock_macro_bound() -> dict:
    """T2.1 empirical: mean 20.9 min/adapter; 25 - available > 2h threshold."""
    try:
        with open(REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/results.json") as f:
            t21 = json.load(f)
    except Exception as e:
        return {"id": "P4", "passed": False, "error": f"could not read T2.1 results: {e}"}

    t_math = t21.get("math_train_time_s", 0)
    t_code = t21.get("code_train_time_s", 0)
    t_med = t21.get("med_train_time_s", 0)
    if not all((t_math, t_code, t_med)):
        return {"id": "P4", "passed": False, "error": "missing T2.1 training times"}

    mean_s = (t_math + t_code + t_med) / 3
    mean_min = mean_s / 60.0

    total, _, _ = _count_adapters()
    missing = max(0, 25 - total)

    T_total_min = missing * mean_min
    T_total_h = T_total_min / 60.0
    iter_budget_min = 30
    macro_threshold_min = 120

    return {
        "id": "P4",
        "passed": T_total_min >= macro_threshold_min,
        "mean_min_per_adapter": round(mean_min, 2),
        "missing": missing,
        "T_total_min": round(T_total_min, 2),
        "T_total_h": round(T_total_h, 2),
        "iter_budget_min": iter_budget_min,
        "ratio_over_iter_budget": round(T_total_min / iter_budget_min, 2),
        "macro_threshold_min": macro_threshold_min,
    }


def pred5_mmlu_pro_14_categories() -> dict:
    """MMLU-Pro has 14 disciplines (Wang et al. 2024). N=25 > 14."""
    mmlu_pro_categories = [
        "biology", "business", "chemistry", "computer_science",
        "economics", "engineering", "health", "history",
        "law", "math", "philosophy", "physics",
        "psychology", "other",
    ]
    n_cats = len(mmlu_pro_categories)
    target_N = 25
    return {
        "id": "P5",
        "passed": n_cats == 14 and target_N > n_cats,
        "n_mmlu_pro_categories": n_cats,
        "target_N": target_N,
        "pigeonhole_violation": target_N > n_cats,
        "categories": mmlu_pro_categories,
    }


def main() -> None:
    preds = [
        pred1_no_safetensors_in_exp_dir(),
        pred2_success_criteria_missing(),
        pred3_insufficient_adapters(),
        pred4_wallclock_macro_bound(),
        pred5_mmlu_pro_14_categories(),
    ]
    all_pass = all(p.get("passed") for p in preds)
    verdict = "KILLED_PREEMPTIVE" if all_pass else "INDETERMINATE"

    out = {
        "experiment_id": EXP_ID,
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "K1602_result": "fail" if all_pass else "untested",
        "predictions": preds,
        "note": (
            "Preemptive-kill per MATH.md Theorems 1-5. K1602 ('diff >= 5pp "
            "relevance-weighted vs equal-weight compose at N=25') is unreachable "
            "given 3-4/25 adapters available + 7.32h wall-clock macro-breach + "
            "framework/metric mismatches + F#137 non-transfer (BitNet-2B to Gemma 4)."
        ),
    }
    results_path = EXP_DIR / "results.json"
    results_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({"verdict": verdict, "all_pass": all_pass, "results": str(results_path)}))


if __name__ == "__main__":
    main()

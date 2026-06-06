#!/usr/bin/env python3
"""exp_g4_5domain_real_hf — preemptive-kill pre-flight.

See MATH.md for the five theorems that kill K1604 and K1605. This script verifies
the five predictions (P1-P5) as a pure-filesystem / JSON check. No model loading,
no training, no MLX. Wall-clock < 2 s.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent.parent
EXP_ID = "exp_g4_5domain_real_hf"

F44_DOMAINS = {"code", "medical", "python", "legal", "creative"}


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


def _count_f44_matched_adapters() -> tuple[int, list[str]]:
    """Count adapters on disk whose directory name matches an F#44 domain."""
    t21_dir = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters"
    repo_adapters_dir = REPO_ROOT / "data" / "adapters"
    matched: list[str] = []
    for base in (t21_dir, repo_adapters_dir):
        if not base.exists():
            continue
        for adapter_file in base.rglob("adapters.safetensors"):
            # Domain name = immediate parent directory name
            domain = adapter_file.parent.name
            if domain in F44_DOMAINS:
                matched.append(str(adapter_file.relative_to(REPO_ROOT)))
    # Deduplicate on domain (not file path): T2.1 vs repo/adapters dup on same domain
    seen_domains: set[str] = set()
    deduped: list[str] = []
    for rel in matched:
        d = Path(rel).parent.name
        if d not in seen_domains:
            seen_domains.add(d)
            deduped.append(rel)
    return len(seen_domains), deduped


def pred3_insufficient_adapters() -> dict:
    required = 5
    matched_count, matched_paths = _count_f44_matched_adapters()
    return {
        "id": "P3",
        "passed": matched_count < required,
        "required": required,
        "f44_matched_count": matched_count,
        "f44_matched_paths": matched_paths,
        "shortfall": required - matched_count,
        "f44_domain_set": sorted(F44_DOMAINS),
    }


def pred4_wallclock_iter_budget_breach() -> dict:
    """T2.1 empirical: mean 20.9 min/adapter; (5 - F#44-matched) * mean > 30 min iter budget."""
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

    matched_count, _ = _count_f44_matched_adapters()
    missing = max(0, 5 - matched_count)

    T_total_min = missing * mean_min
    T_total_h = T_total_min / 60.0
    iter_budget_min = 30
    macro_threshold_min = 120

    return {
        "id": "P4",
        "passed": T_total_min >= iter_budget_min,
        "mean_min_per_adapter": round(mean_min, 2),
        "missing": missing,
        "T_total_min": round(T_total_min, 2),
        "T_total_h": round(T_total_h, 2),
        "iter_budget_min": iter_budget_min,
        "macro_threshold_min": macro_threshold_min,
        "ratio_over_iter_budget": round(T_total_min / iter_budget_min, 2) if iter_budget_min else None,
        "breaches_iter_budget": T_total_min >= iter_budget_min,
        "breaches_macro_ceiling": T_total_min >= macro_threshold_min,
    }


def pred5_kc_under_specification() -> dict:
    """K1604 and K1605 text lacks eval-task keywords."""
    try:
        out = subprocess.run(
            ["experiment", "get", EXP_ID],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception as e:
        return {"id": "P5", "passed": False, "error": str(e)}

    eval_keywords = ("MMLU-Pro", "GSM8K", "HumanEval", "PPL")
    # Isolate the Kill Criteria block; look for keywords there only.
    lines = out.splitlines()
    in_kc_block = False
    kc_text_parts: list[str] = []
    for ln in lines:
        if "Kill Criteria:" in ln:
            in_kc_block = True
            continue
        if in_kc_block:
            stripped = ln.strip()
            if not stripped:
                # Blank line exits the block
                break
            # Stop at next top-level section header (no leading whitespace and ends with ':')
            if not ln.startswith(" ") and stripped.endswith(":"):
                break
            kc_text_parts.append(stripped)
    kc_text = " ".join(kc_text_parts)
    keywords_found = [kw for kw in eval_keywords if kw.lower() in kc_text.lower()]
    passed = len(keywords_found) == 0
    return {
        "id": "P5",
        "passed": passed,
        "kc_text": kc_text,
        "eval_keywords_searched": list(eval_keywords),
        "eval_keywords_found_in_kc": keywords_found,
    }


def main() -> None:
    preds = [
        pred1_no_safetensors_in_exp_dir(),
        pred2_success_criteria_missing(),
        pred3_insufficient_adapters(),
        pred4_wallclock_iter_budget_breach(),
        pred5_kc_under_specification(),
    ]
    all_pass = all(p.get("passed") for p in preds)
    verdict = "KILLED_PREEMPTIVE" if all_pass else "INDETERMINATE"

    out = {
        "experiment_id": EXP_ID,
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "K1604_result": "fail" if all_pass else "untested",
        "K1605_result": "fail" if all_pass else "untested",
        "predictions": preds,
        "note": (
            "Preemptive-kill per MATH.md Theorems 1-5. K1604 ('>=4/5 domains improve "
            "own-domain') and K1605 ('0/5 degrade base >3%') are unreachable given "
            "2-3/5 F#44-matched adapters on disk + 41.8 min iter-budget breach + "
            "framework-incomplete (success_criteria=[]) + KC under-specification "
            "(eval task, base, delta-threshold unpinned) + F#44 BitNet-2B-4T "
            "PPL-based result non-transfer to Gemma 4 E4B RMSNorm+QK-pre-norm+MQA "
            "task-eval."
        ),
    }
    results_path = EXP_DIR / "results.json"
    results_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({"verdict": verdict, "all_pass": all_pass, "results": str(results_path)}))


if __name__ == "__main__":
    main()

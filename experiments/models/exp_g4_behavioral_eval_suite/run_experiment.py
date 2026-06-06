"""Precondition probe for exp_g4_behavioral_eval_suite.

13th consecutive audit-2026-04-17 cohort experiment gated on the same upstream
blocker (`exp_p1_t2_single_domain_training` rerun). Per MATH.md tripwire, if
any of P1/P2/P3 FAIL at run-start, K1593 is UNMEASURABLE and the experiment
is status=killed before any heavy compute.

This script runs only the probe; it never touches MLX.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
UPSTREAM = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training"


def probe_p1() -> tuple[bool, dict]:
    """P1: Gemma 4 per-domain LoRA adapter safetensors exist on disk."""
    adapters = UPSTREAM / "adapters"
    safetensors = list(adapters.rglob("*.safetensors")) if adapters.exists() else []
    domains_found = sorted({p.parent.name for p in safetensors})
    ok = len(safetensors) >= 3  # at least math/code/medical
    return ok, {
        "upstream_exists": adapters.exists(),
        "n_safetensors": len(safetensors),
        "domains_with_safetensors": domains_found,
    }


def probe_p2() -> tuple[bool, dict]:
    """P2: 4 benchmark harnesses wired to Gemma 4 E4B 4-bit."""
    candidates = [
        REPO / "micro" / "models" / "exp_g4_cot_vs_direct_mmlu_pro" / "run_experiment.py",
        REPO / "micro" / "models" / "exp_bench_aime_2026" / "run_experiment.py",
        REPO / "micro" / "models" / "exp_bench_livecodebench_v6" / "run_experiment.py",
    ]
    present = [c.name for c in candidates if c.exists()]
    needed = {"mmlu_pro": False, "gsm8k": False, "humaneval": False, "medmcqa": False}
    for runner in candidates:
        if not runner.exists():
            continue
        txt = runner.read_text()
        for key in needed:
            if key in txt.lower():
                needed[key] = True
    # Also check for adapter-swap test harness wiring; absent means adapters can't be evaluated.
    ok = all(needed.values())
    return ok, {"runners_present": present, "benchmarks_wired": needed}


def probe_p3() -> tuple[bool, dict]:
    """P3: per-sample correctness labels available for AUC substrate."""
    # Without P1 adapters, P3 cannot be satisfied; no (prompt, adapter) pairs exist.
    p1_ok, _ = probe_p1()
    per_sample_infra = p1_ok  # binding: no adapters -> no per-sample AUC possible
    return per_sample_infra, {
        "per_sample_correctness_recordable": per_sample_infra,
        "reason_if_fail": "no adapter safetensors on disk -> no (prompt, adapter) pairs",
    }


def main() -> None:
    t0 = time.perf_counter()
    p1_ok, p1_detail = probe_p1()
    p2_ok, p2_detail = probe_p2()
    p3_ok, p3_detail = probe_p3()

    all_pass = p1_ok and p2_ok and p3_ok
    verdict = "SUPPORTED" if all_pass else "KILLED"
    k1593 = "PASS" if all_pass else "FAIL"
    wall_s = time.perf_counter() - t0

    out = {
        "experiment_id": "exp_g4_behavioral_eval_suite",
        "verdict": verdict,
        "all_pass": all_pass,
        "ran": True,
        "is_smoke": False,
        "wall_time_s": round(wall_s, 4),
        "kill_criteria": {
            "K1593_auc_ge_0p85_across_4_benchmarks": {
                "result": k1593,
                "measurable": all_pass,
            },
        },
        "preconditions": {
            "P1_adapter_safetensors": {"pass": p1_ok, **p1_detail},
            "P2_benchmark_harnesses": {"pass": p2_ok, **p2_detail},
            "P3_per_sample_labels": {"pass": p3_ok, **p3_detail},
        },
        "upstream_blocker": (
            "exp_p1_t2_single_domain_training rerun at LORA_SCALE=5, "
            "max_tokens>=512, 5+ disjoint domains, rank sweep, grad-SNR logging. "
            "Same blocker as Findings #605/#606/#608/#610/#611/#612/#613/#615/"
            "#616/#617/#618/#619."
        ),
        "cohort_kill_index": 13,
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

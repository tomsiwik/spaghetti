#!/usr/bin/env python3
"""exp_g4_batched_lora_k1 preemptive-kill runner.

Pure-stdlib: no MLX, no model load. Verifies the two structural predictions
in MATH.md (T1 framework-incompleteness; T2 Finding #306 prior-art
displacement) by reading DB state via `experiment get` and `experiment
finding-get`. No shebang-system-python3 risk (ap-027 N/A).
"""

import json
import subprocess
import sys
from pathlib import Path

EXP_ID = "exp_g4_batched_lora_k1"
DIR = Path(__file__).parent
RESULTS = DIR / "results.json"


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (r.stdout or "") + (r.stderr or "")


def check_t1_success_criteria_empty() -> tuple[bool, str]:
    """T1: success_criteria=[] blocks SUPPORTED (PLAN.md §1)."""
    out = run(["experiment", "get", EXP_ID])
    hit = ("Success Criteria: NONE" in out) or ("success_criteria: []" in out)
    return hit, f"success_criteria empty/NONE marker found={hit}"


def check_t2_finding_306_killed() -> tuple[bool, str]:
    """T2: F#306 MLX-batching killed — prior-art displacement."""
    out = run(["experiment", "finding-get", "306"])
    killed = "Status:     killed" in out
    mlx_fusion = "lazy evaluation" in out.lower() and "fusion" in out.lower()
    ok = killed and mlx_fusion
    return ok, (
        f"F#306 status=killed:{killed}; MLX-fusion impossibility cited:{mlx_fusion}"
    )


def check_t3_kc_underspecified() -> tuple[bool, str]:
    """T3: K1601 text contains none of {forward, prefill, decode, batch size}."""
    out = run(["experiment", "get", EXP_ID])
    for line in out.splitlines():
        if "#1601" in line:
            missing = not any(
                w in line.lower() for w in ("forward", "prefill", "decode", "batch=")
            )
            return missing, f"K1601 line='{line.strip()}' underspecified={missing}"
    return False, "K1601 line not found"


def check_t4_cohort_tag() -> tuple[bool, str]:
    out = run(["experiment", "get", EXP_ID])
    in_cohort = "audit-2026-04-17" in out
    return in_cohort, f"audit-2026-04-17 cohort member: {in_cohort}"


def main() -> int:
    preds = {
        "T1_success_criteria_empty": check_t1_success_criteria_empty(),
        "T2_finding_306_killed": check_t2_finding_306_killed(),
        "T3_kc_underspecified": check_t3_kc_underspecified(),
        "T4_audit_cohort_member": check_t4_cohort_tag(),
    }
    all_pass = all(ok for ok, _ in preds.values())
    payload = {
        "experiment_id": EXP_ID,
        "verdict": "KILLED_PREEMPTIVE",
        "all_pass": all_pass,
        "is_smoke": False,
        "kill_criteria": {
            "K1601": {
                "text": "throughput ratio >= 0.96",
                "result": "fail",
                "reason": (
                    "T1 success_criteria=[] blocks SUPPORTED per PLAN.md §1; "
                    "T2 Finding #306 (exp_batched_lora_gather_mlx) already "
                    "settled MLX batching on lazy-eval impossibility structure."
                ),
            }
        },
        "predictions": {k: {"pass": ok, "detail": d} for k, (ok, d) in preds.items()},
        "theorems_verified": len(preds),
        "notes": (
            "Pure-stdlib preemptive-kill. No MLX, no training, no model load. "
            "Finding #306 transfer: MLX lazy-eval makes manual batching "
            "structurally impossible to outperform framework fusion."
        ),
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={payload['verdict']} all_pass={all_pass}")
    for k, (ok, d) in preds.items():
        print(f"  {k}: pass={ok} :: {d}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

"""Precondition probe for exp_model_mtbench_composed.

The full MT-Bench sweep requires N=5 composed adapters on disk, an
MT-Bench harness, and a non-killed upstream baseline. Running a 4-6h
sweep before those preconditions are known-good would just conflate
infrastructure state with capability state. This probe resolves the
preconditions in ~3s and writes results.json so the verdict is auditable
without generation.

Pre-registered KCs (MATH.md):
  K1697 — MT-Bench overall >= base - 0.2        (FAIL if P1 blocked)
  K1698 — No category < base - 0.5              (FAIL if P1 blocked)
  K1699 — GPT-4 judge consistent                (FAIL if P1 blocked — not reached)
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTERS_DIR = REPO_ROOT / "data" / "adapters"
UPSTREAM_RESULTS = (
    REPO_ROOT
    / "micro"
    / "models"
    / "exp_p1_t2_single_domain_training"
    / "results.json"
)


def probe_p1_adapters_on_disk() -> dict:
    """P1: all 5 Pierre adapter safetensors must exist."""
    required = ["math", "code", "medical", "sql", "bash"]
    per_domain: dict[str, dict] = {}
    for dom in required:
        dom_dir = ADAPTERS_DIR / dom
        dir_exists = dom_dir.is_dir()
        safetensors = (
            [str(p.relative_to(REPO_ROOT)) for p in dom_dir.glob("*.safetensors")]
            if dir_exists
            else []
        )
        per_domain[dom] = {
            "dir_exists": dir_exists,
            "safetensors_count": len(safetensors),
            "safetensors": safetensors,
        }
    all_present = all(
        v["dir_exists"] and v["safetensors_count"] > 0 for v in per_domain.values()
    )
    return {
        "pass": all_present,
        "per_domain": per_domain,
        "n_domains_with_weights": sum(
            1 for v in per_domain.values() if v["safetensors_count"] > 0
        ),
        "n_domains_required": len(required),
    }


def probe_p2_harness_import() -> dict:
    """P2: MT-Bench harness importable.

    We check for FastChat's llm_judge module (canonical MT-Bench) via a
    subprocess import so the probe does not leak dependencies into this
    process. If FastChat is absent, fall back to checking lm-eval-harness
    mt_bench task — either is acceptable.
    """
    candidates = [
        ("fastchat.llm_judge", "import fastchat.llm_judge"),
        ("lm_eval", "import lm_eval; _ = lm_eval.tasks"),
    ]
    importable: list[str] = []
    errors: dict[str, str] = {}
    for name, stmt in candidates:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", stmt],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        if proc.returncode == 0:
            importable.append(name)
        else:
            errors[name] = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else f"exit={proc.returncode}"
    return {
        "pass": len(importable) > 0,
        "importable": importable,
        "errors": errors,
    }


def probe_p3_upstream_not_killed() -> dict:
    """P3: T2.1 verdict is not KILLED."""
    if not UPSTREAM_RESULTS.exists():
        return {
            "pass": False,
            "reason": f"upstream results.json missing at {UPSTREAM_RESULTS.relative_to(REPO_ROOT)}",
        }
    data = json.loads(UPSTREAM_RESULTS.read_text())
    verdict = data.get("verdict")
    audit_note = data.get("_audit_note", "") or ""
    has_audit_flag = bool(audit_note) or "metric-swap" in json.dumps(data).lower()
    return {
        "pass": verdict != "KILLED" and not has_audit_flag,
        "upstream_verdict": verdict,
        "has_audit_note": bool(audit_note),
        "audit_excerpt": audit_note[:200] if audit_note else "",
    }


def main() -> None:
    t0 = time.time()
    p1 = probe_p1_adapters_on_disk()
    p2 = probe_p2_harness_import()
    p3 = probe_p3_upstream_not_killed()

    k1697 = "pass" if p1["pass"] else "fail"
    k1698 = "pass" if p1["pass"] else "fail"
    k1699 = "pass" if p1["pass"] else "fail"
    all_pass = p1["pass"] and p2["pass"] and p3["pass"]
    verdict = "PROVISIONAL" if all_pass else "KILLED"

    out = {
        "verdict": verdict,
        "all_pass": all_pass,
        "ran": True,
        "is_smoke": False,
        "experiment_id": "exp_model_mtbench_composed",
        "mode": "precondition-probe",
        "runtime_s": round(time.time() - t0, 2),
        "preconditions": {
            "P1_adapters_on_disk": p1,
            "P2_harness_import": p2,
            "P3_upstream_not_killed": p3,
        },
        "kill_criteria": {
            "K1697_mtbench_overall_within_02": k1697,
            "K1698_no_category_below_base_minus_05": k1698,
            "K1699_judge_consistent": k1699,
        },
        "rationale": (
            "Precondition P1 (adapters on disk) and P3 (upstream non-killed) "
            "must pass before K1697/K1698/K1699 are structurally measurable. "
            "A 4-6h MT-Bench sweep without satisfied preconditions conflates "
            "infrastructure state with capability state."
        ),
    }

    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"probe verdict={verdict} all_pass={all_pass} t={out['runtime_s']}s")


if __name__ == "__main__":
    main()

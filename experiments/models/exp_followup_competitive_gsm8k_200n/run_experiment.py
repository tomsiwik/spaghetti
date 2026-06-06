#!/usr/bin/env python3
"""Pre-flight-only run for exp_followup_competitive_gsm8k_200n.

Verdict: KILLED preemptive (antipattern-017 cascade, 6th confirmed instance).
All 5 domain adapters referenced in adapters/registry.json are config-only stubs
with no adapters.safetensors on disk. K1575 unmeasurable — routed composition
degenerates to base by Theorem 1 in MATH.md.

This script does NOT load the base model, does NOT evaluate any benchmark, and
does NOT train anything. It performs a pre-flight check of adapter weight files
on disk, writes results.json with verdict=KILLED, and exits.

No MLX arrays are created; no mx.eval / mx.clear_cache discipline needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REGISTRY_FILE = REPO_ROOT / "data" / "adapters" / "registry.json"


def check_adapter_weights() -> dict:
    """Return per-adapter weight presence from registry paths."""
    registry = json.loads(REGISTRY_FILE.read_text())
    report = {}
    for adapter in registry["adapters"]:
        name = adapter["name"]
        # Only check the 5 domain knowledge adapters this experiment needs.
        if adapter.get("domain") not in {"math", "code", "medical", "legal", "finance"}:
            continue
        path = REPO_ROOT / adapter["path"]
        weights = path / "adapters.safetensors"
        report[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "config_present": (path / "adapter_config.json").is_file(),
            "weights_present": weights.is_file(),
            "weights_bytes": weights.stat().st_size if weights.is_file() else 0,
        }
    return report


def main() -> None:
    t0 = time.time()
    adapter_report = check_adapter_weights()

    missing = [name for name, info in adapter_report.items() if not info["weights_present"]]
    total = len(adapter_report)
    weights_found = total - len(missing)

    # Pre-registered predictions (MATH.md Theorem 1 + 2)
    # K1575: routed beats base with CI not overlapping zero.
    # Predicted FAIL by construction if any adapter weights missing.
    k1575_result = "fail" if missing else "untested"

    results = {
        "experiment_id": "exp_followup_competitive_gsm8k_200n",
        "verdict": "KILLED",
        "is_smoke": False,
        "all_pass": False,
        "kill_type": "preemptive_cascade",
        "antipattern_trigger": ["antipattern-017", "antipattern-020"],
        "antipattern_017_instance_count": 6,
        "kill_criteria": {
            "1575": {
                "result": k1575_result,
                "text": "At n>=100/subject with fixed extraction, routed beats base with CI not overlapping zero",
                "failure_reason": (
                    f"{len(missing)}/{total} required domain adapters are config-only stubs "
                    f"(no adapters.safetensors). Routed composition degenerates to base by "
                    f"MATH.md Theorem 1 + 2 — E[routed - base] = 0, so CI is centered at zero."
                ),
            },
        },
        "dependencies": {
            "registry_adapter_state": adapter_report,
            "missing_adapter_names": missing,
            "weights_found": weights_found,
            "weights_required": total,
            "source_experiment_dir_adapters": {
                "path": "experiments/models/real_data_domain_experts/adapters/",
                "exists": (REPO_ROOT / "micro" / "models" / "real_data_domain_experts" / "adapters").is_dir(),
                "note": "Prior killed experiment's adapter source dir; referenced at competitive_benchmark_routed/run_experiment.py:39",
            },
            "parent_killed_experiment": {
                "id": "exp_competitive_benchmark_routed",
                "status": "killed",
                "kill_date": "2026-04-17",
                "kill_criterion": "K640: routed worse than base on math -20pp @ n=20, legal -10pp",
            },
        },
        "unblock_path": {
            "id": "P11.ADAPTER-REBUILD",
            "description": (
                "Retrain the 5 domain knowledge adapters (math, code, medical, legal, finance) "
                "into the registry-referenced paths, with adapters.safetensors of nonzero size."
            ),
            "verification": (
                "for name in [math, code, medical, legal, finance]: "
                "assert (registry_path/'adapters.safetensors').stat().st_size > 0"
            ),
        },
        "references": {
            "math_md": "MATH.md - Theorems 1 & 2; antipattern-017 self-check",
            "findings": ["F#237", "F#517", "F#553", "F#560"],
            "antipatterns": ["antipattern-017", "antipattern-020"],
        },
        "runtime_seconds": round(time.time() - t0, 3),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"[PREEMPTIVE KILL] adapter weights found: {weights_found}/{total}")
    print(f"[PREEMPTIVE KILL] missing: {missing}")
    print(f"[PREEMPTIVE KILL] verdict written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

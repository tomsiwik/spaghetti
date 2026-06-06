#!/usr/bin/env python3
"""Pre-flight-only run for exp_g4_routed_beats_base_think.

Verdict: KILLED preemptive (antipattern-017 cascade, 7th confirmed instance).

All 5 domain knowledge adapters referenced in adapters/registry.json are
config-only stubs (no adapters.safetensors on disk). K1592 unmeasurable:
routed composition degenerates to base by MATH.md Theorems 1-3. Thinking
mode does not rescue (Thm 3, prompt-level change inert to absent adapter ops).

This script does NOT load the base model, does NOT evaluate any benchmark,
and does NOT train anything. It performs a pre-flight check of adapter
weight files on disk, writes results.json with verdict=KILLED, and exits.

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

REQUIRED_DOMAINS = {"math", "code", "medical", "legal", "finance"}


def check_adapter_weights() -> dict:
    """Return per-adapter weight presence from registry paths."""
    registry = json.loads(REGISTRY_FILE.read_text())
    report = {}
    for adapter in registry["adapters"]:
        if adapter.get("domain") not in REQUIRED_DOMAINS:
            continue
        name = adapter["name"]
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

    # Pre-registered predictions (MATH.md Theorems 1-3)
    # K1592: GSM8K +5pp AND MMLU-Pro >= base (with thinking mode enabled).
    # Predicted FAIL by construction if any adapter weights missing.
    k1592_result = "fail" if missing else "untested"

    results = {
        "experiment_id": "exp_g4_routed_beats_base_think",
        "verdict": "KILLED",
        "is_smoke": False,
        "all_pass": False,
        "kill_type": "preemptive_cascade",
        "antipattern_trigger": ["antipattern-017", "antipattern-020"],
        "antipattern_017_instance_count": 7,
        "kill_criteria": {
            "1592": {
                "result": k1592_result,
                "text": "GSM8K +5pp AND MMLU-Pro >= base (with thinking mode enabled)",
                "failure_reason": (
                    f"{len(missing)}/{total} required domain adapters are config-only stubs "
                    f"(no adapters.safetensors). Routed composition degenerates to base by "
                    f"MATH.md Theorems 1-2. Thinking mode does not rescue (Theorem 3: "
                    f"prompt-level change is inert to absent adapter operators)."
                ),
            },
        },
        "dependencies": {
            "registry_adapter_state": adapter_report,
            "missing_adapter_names": missing,
            "weights_found": weights_found,
            "weights_required": total,
            "thinking_adapter_note": (
                "adapters/thinking-openthoughts-universal-v0/ has real weights (151 MB) but is "
                "unrelated to this experiment — it is a universal thinking adapter, not one of "
                "the 5 domain experts K1592 routes over."
            ),
        },
        "unblock_path": {
            "id": "P11.ADAPTER-REBUILD",
            "description": (
                "Retrain the 5 domain knowledge adapters (math, code, medical, legal, finance) "
                "into the registry-referenced paths, with adapters.safetensors of nonzero size. "
                "Same unblock path as M0, L0, J0, followup_composition_correct_delta, "
                "followup_routing_multi_sample_ppl, followup_competitive_gsm8k_200n."
            ),
            "verification": (
                "for name in [math, code, medical, legal, finance]: "
                "assert (registry_path/'adapters.safetensors').stat().st_size > 0"
            ),
        },
        "references": {
            "math_md": "MATH.md - Theorems 1, 2, 3; antipattern-017 self-check",
            "findings": ["F#236", "F#237", "F#517", "F#553", "F#560"],
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

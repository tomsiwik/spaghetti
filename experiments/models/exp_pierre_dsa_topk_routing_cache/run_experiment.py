"""DSA top-k key selection as joint sparse-attention + routing prior.

STATUS: SPEC ONLY. Implementation pending.

This is the highest-cost SSA experiment — depends on lightning-indexer
infrastructure that doesn't exist yet. Prerequisite: HiP-style sparse attention
+ routing cache infrastructure.
"""
from __future__ import annotations
import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent


def main():
    out_path = EXP_DIR / "results.json"
    results = {
        "verdict": "INCONCLUSIVE",
        "decision": "implementation_pending",
        "config": {
            "method_description": "DSA top-k indices dual-used as sparse attention + routing prior",
            "paper": "DeepSeek V3.2 (DSA)",
            "implementation_status": "SPEC ONLY",
            "blockers": [
                "MLX lightning indexer (~80M params, scorer per layer)",
                "Top-k selection integration with attention",
                "Routing-cache plumbing (depends on exp_pierre_kv_cached_layer_routing_1m infrastructure)",
                "Long-context eval rig at 128K",
                "Prerequisite: HiP-style sparse attention working + routing cache built",
            ],
            "estimated_implementation_time": "8-12 hours (highest of the SSA set)",
        },
        "kill_criteria_pre_registered": {
            "K1_behavioral": "Routing-via-DSA ≥ Fisher-Rao K=7 baseline",
            "K2_memory": "128K routing+attn cache ≤ 50% of independent caches",
            "K3_cache_hit": "≥60% routing-cache reuse rate",
            "K4_no_collapse": "Per-benchmark within 5pp of K=7 baseline",
        },
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"=== INCONCLUSIVE: implementation pending ===")
    print(f"  See MATH.md for pre-registered KCs")


if __name__ == "__main__":
    main()

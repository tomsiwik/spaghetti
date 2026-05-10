"""ASEntmax dropin — α-entmax replacement for softmax in attention.

STATUS: SPEC ONLY. Implementation pending.

Required to make this experiment runnable:
1. MLX α-entmax kernel (~80 LOC) — closed-form sort-based for α=1.5
2. Patch Gemma 4 attention module's softmax call (~30 LOC) via Finding #831 setattr pattern
3. Long-context eval rig (RULER@32K) for K2

This stub exits with INCONCLUSIVE + implementation_pending verdict to
preserve queue integrity. KCs are pre-registered in MATH.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent


def main():
    out_path = EXP_DIR / "results.json"
    results = {
        "verdict": "INCONCLUSIVE",
        "decision": "implementation_pending",
        "config": {
            "method_description": "α-entmax(α=1.5) drop-in replacement for softmax",
            "paper": "arxiv 2506.16640",
            "implementation_status": "SPEC ONLY",
            "blockers": [
                "MLX α-entmax kernel needs writing (~80 LOC closed-form sort-based)",
                "Gemma 4 attention module's softmax call needs patching (~30 LOC, setattr pattern)",
                "Long-context eval rig (RULER@32K) needs to be added for K2",
            ],
            "estimated_implementation_time": "1-2 hours of focused MLX code + 1-2 hours testing",
        },
        "kill_criteria_pre_registered": {
            "K1_adapter_arithmetic_preserved": "Within ±0.5pp of softmax baseline on Fisher-Rao K=7",
            "K2_retrieval_at_32K": "RULER@32K ≥ softmax baseline + 1pp",
            "K3_decode_latency": "Decode tok/s drops < 5%",
            "K4_sparsity_nontrivial": "≥50% of heads show non-trivial sparsity",
        },
        "next_steps": [
            "Implement alpha_entmax_15(scores, mask=None) → MLX",
            "Locate Gemma 4 attention class in mlx_lm.models.gemma4",
            "Write subclass override with alpha_entmax in place of softmax",
            "Self-test: α→1.001 reproduces softmax within float-precision",
            "Re-run this experiment script (no MATH.md change permitted — KCs are locked)",
        ],
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"=== INCONCLUSIVE: implementation pending ===")
    print(f"  See MATH.md for pre-registered KCs and full spec")
    print(f"  Results: {out_path}")


if __name__ == "__main__":
    main()

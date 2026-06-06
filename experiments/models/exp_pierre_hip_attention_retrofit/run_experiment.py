"""HiP Attention retrofit (training-free hierarchical pruning).

STATUS: SPEC ONLY. Implementation pending.
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
            "method_description": "HiP — training-free hierarchical attention pruning",
            "paper": "arxiv 2406.09827",
            "implementation_status": "SPEC ONLY",
            "blockers": [
                "MLX HiP kernel (~150 LOC) — hierarchical scorer + block-sparse attention",
                "Drop into Gemma 4 full-attention layer(s) only (1/6 layers)",
                "Self-test: at full-density, must reproduce dense attention exactly",
            ],
            "estimated_implementation_time": "3-4 hours",
        },
        "kill_criteria_pre_registered": {
            "K1_behavioral": "≤1.0pp drop vs softmax baseline on 3-bench",
            "K2_speedup_32K": "Decode tok/s @32K ≥ 1.2× softmax baseline",
            "K3_adapter_math_preserved": "DARE recomposition within ±0.5pp",
            "K4_full_attention_layers_preserved": "PPL on 1/6 full-attn layers rises ≤ 5%",
        },
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"=== INCONCLUSIVE: implementation pending ===")
    print(f"  See MATH.md for pre-registered KCs")


if __name__ == "__main__":
    main()

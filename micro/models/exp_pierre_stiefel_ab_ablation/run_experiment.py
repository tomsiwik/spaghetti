"""Stiefel-A vs Stiefel-B vs both — ablation. SPEC — depends on prior
Stiefel arc experiments completing first.
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
            "method_description": "4-way ablation: {no constraint, A only, B only, A+B} × 7 adapters trained from scratch",
            "implementation_status": "SPEC",
            "blockers": [
                "All prior Stiefel arc experiments must complete (infrastructure)",
                "4 full training runs at ~3.5h each = 14h training",
                "Configurable A and B retraction toggles in polar_train",
            ],
            "estimated_total_time": "16h compute",
        },
        "kill_criteria_pre_registered": {
            "K1_current_is_useful": "Stiefel-A alone (Pierre current) > unconstrained by >= 2pp avg",
            "K2_b_adds_value": "Stiefel-A+B > Stiefel-A alone by >= 2pp composed avg",
            "K3_b_alone_sufficient": "Stiefel-B alone >= Stiefel-A alone (could drop A constraint)",
            "K4_diminishing_returns": "Joint gain (A+B - none) vs sum of individual gains — additive or redundant?",
        },
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"=== INCONCLUSIVE: implementation pending ===")


if __name__ == "__main__":
    main()

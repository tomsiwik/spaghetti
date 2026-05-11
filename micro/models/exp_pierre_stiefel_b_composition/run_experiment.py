"""Composition behavior of joint-Stiefel-trained adapters. SPEC — depends on
joint-Stiefel training (sibling experiment) completing first.
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
            "method_description": "4 composition methods on joint-Stiefel-trained K=7 adapters",
            "methods": ["simple_mean (uniform 1/K, no rescale)", "fisher_rao", "ties_b", "karcher_stiefel (Riemannian Frechet)"],
            "implementation_status": "SPEC",
            "blockers": [
                "Depends on exp_pierre_joint_stiefel_b_train weights",
                "Implement Karcher mean on Stiefel(K·r, d_out) (~80 LoC iterative)",
            ],
            "estimated_total_time": "1h eval runtime once weights exist",
        },
        "kill_criteria_pre_registered": {
            "K1_simple_mean_wins": "simple_mean on Stiefel-trained adapters >= TIES-B baseline (71.3%)",
            "K2_no_heuristics_needed": "simple_mean >= best of {fisher_rao, ties_b} by >= 1pp",
            "K3_karcher_upside": "karcher_stiefel >= simple_mean by >= 1pp (FAIL is OK)",
            "K4_robustness": "Under 5% weight noise, composition accuracy drops <= 2pp",
        },
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"=== INCONCLUSIVE: implementation pending ===")


if __name__ == "__main__":
    main()

"""Stiefel-KAN hybrid — hard-orthogonality + additive composition.

STATUS: SPEC + ALGORITHM. Implementation pending.

Required engineering:
1. MLX Stiefel retraction (~80 LoC, QR-based).
2. Riemannian gradient projection (~30 LoC).
3. Stiefel-constrained KAN block (extends exp_pierre_kan_adapter_lagrangian).
4. Multi-task training script (~250 LoC, 3.5h training time).

Total: ~6-8h MLX implementation + 3.5h training. KCs pre-registered in MATH.md.
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
            "method_description": "Stiefel-constrained KAN coefficient vectors per edge",
            "papers": [
                "arxiv 2404.19756 (KAN)",
                "arxiv 2508.17901 (Riemannian Stiefel LoRA)",
                "arxiv 2510.01938 (StelLA Stiefel subspace)",
            ],
            "implementation_status": "SPEC + ALGORITHM",
            "blockers": [
                "MLX Stiefel retraction kernel (~80 LoC, QR-based)",
                "Riemannian gradient projection (~30 LoC)",
                "Stiefel-constrained KAN block (extends KAN baseline)",
                "Multi-task training script (~250 LoC)",
                "Training time: ~3.5h on M5 Pro for 7 adapters",
            ],
            "estimated_total_time": "6-8h MLX dev + 3.5h training",
            "thesis": (
                "Composition is interference iff parameters are interfering. "
                "Stiefel constrains parameters to be non-interfering. "
                "Therefore composition is non-interfering — by theorem, not heuristic."
            ),
        },
        "kill_criteria_pre_registered": {
            "K1_feasibility": "Single Stiefel-KAN adapter ≥ standard PoLAR - 5pp on native benchmark",
            "K2_orthogonality": "⟨ϕ_k, ϕ_j⟩ averaged over input distribution ≤ 0.05",
            "K3_composition": "K=7 composed avg ≥ TIES-B (71.3%)",
            "K4_cross_contribution": "Adding K-1 adapters perturbs adapter k's output ≤ 2% on its task",
        },
        "next_steps_when_implementing": [
            "Verify QR-based Stiefel retraction with M @ M.T ≈ I post-step (1e-5 tolerance)",
            "Smoke test: 1 adapter + 1 task + 100 steps reproduces single-PoLAR within 5pp",
            "Then: 7 adapters + multi-task training + composition eval",
            "K2 measurement requires sampling input distribution — use cached activations from sibling exp_pierre_kan_compositional_orthogonality",
        ],
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"=== INCONCLUSIVE: implementation pending ===")
    print(f"  See MATH.md for full architecture spec and KCs")
    print(f"  Sibling exp_pierre_kan_compositional_orthogonality must run first")


if __name__ == "__main__":
    main()

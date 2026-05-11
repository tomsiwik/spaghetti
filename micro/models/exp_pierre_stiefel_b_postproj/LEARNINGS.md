# LEARNINGS — Stiefel B Post-Hoc Projection

## Core Finding

Post-hoc QR projection onto the Stiefel manifold catastrophically destroys adapter expressivity (-21pp single, -39pp composed). The failure is structural: unconstrained SGD entangles magnitude and direction in B, and orthogonal projection cannot disentangle them. Rescaling does not help because the damage is per-row, not global scale.

## Why It Failed

MedQA collapsing to 0% is the sharpest signal — the medical adapter's learned representation lives entirely in the non-orthogonal structure that QR discards. The projection is mathematically correct (off-diagonal < 1e-4), but correctness of the projection ≠ preservation of the learned function.

## Implication

Stiefel constraints must be imposed *during training*, not after. This validates `exp_pierre_stiefel_b_train_single` as the critical next step: if B is trained on the Stiefel manifold from initialization, magnitude-direction entanglement never forms, and orthogonality is free rather than destructive.

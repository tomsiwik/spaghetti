# LEARNINGS — exp_pierre_joint_stiefel_b_train

## Core Finding

Joint Stiefel constraints on B matrices achieve near-perfect geometric orthogonality
(‖B_all B_all^T − I‖ = 1.89e-07) and preserve individual adapter quality (+0–4pp),
but do NOT deliver functional non-interference. Cross-contribution ranges 3–40%
(threshold 1%) and composed accuracy is 59.3% (threshold 71.3%).

## Why

The full LoRA contribution is A_k @ B_k, not just B_k. Orthogonal B row spaces
guarantee that adapters project to disjoint output subspaces, but the A matrices
are unconstrained — cross-adapter signal flows through A_j @ B_j into directions
that affect downstream computation (attention, LayerNorm, softmax) even when
geometrically orthogonal to B_k. B-only Stiefel is necessary but not sufficient.

## Implication for Next Experiment

The composition guarantee requires joint constraints on BOTH A and B — the product
manifold St(r, d_in) × St(r, d_out) — so that adapters operate on disjoint
input→output channels. Alternatively, a composition operator beyond simple averaging
could compensate for A-B interaction. The `exp_pierre_stiefel_b_composition` and
`exp_pierre_stiefel_ab_ablation` experiments should test these directions.

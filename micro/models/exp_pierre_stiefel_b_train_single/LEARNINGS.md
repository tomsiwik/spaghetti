# LEARNINGS.md — Stiefel-B Training (Single Adapter)

## Core Finding

Training-time Stiefel retraction on B works: +4pp GSM8K over standard, zero overhead, near-perfect orthogonality (5.5e-9). This is the mirror result to postproj's -21pp catastrophe. The optimizer must see the manifold constraint from step 1 — post-hoc projection destroys learned magnitude-direction entanglement that unconstrained SGD creates.

## Why

Unconstrained SGD encodes task information in both the direction AND magnitude of B's rows. Post-hoc QR/SVD preserves direction but destroys magnitude, losing up to 39pp. Training-time retraction prevents the optimizer from ever using magnitude as a degree of freedom, so all task information lives in the subspace orientation — which Stiefel preserves by definition.

The +4pp bonus and -6.2% speed improvement suggest the Stiefel constraint acts as implicit regularization and improves numerical conditioning. At rank=6, SVD retraction cost is negligible.

## Implication

Single-adapter Stiefel-B is validated. The critical next question is whether the joint constraint (K=7 adapters, B_all B_all^T = I_42) scales — `exp_pierre_joint_stiefel_b_train`. The zero-overhead finding at r=6 may not hold at r=42; K3 (step cost) becomes the decisive kill criterion there.

## Finding ID

F-stiefel-b-train-001: Stiefel constraints must be imposed during training, not post-hoc. Training-time retraction preserves expressivity; post-hoc projection destroys it.

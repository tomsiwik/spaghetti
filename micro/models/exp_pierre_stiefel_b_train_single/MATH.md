# MATH.md — Stiefel-aware single-adapter training

## Position in the arc

This experiment runs AFTER `exp_pierre_stiefel_b_postproj` reveals whether
the post-hoc projection works. Two scenarios trigger this:

1. **postproj KILLED or PARTIAL** (existing B's drift too far from Stiefel
   to be projected without accuracy loss): we need to train ON Stiefel
   from the start. This experiment validates that's even possible.
2. **postproj SUPPORTED but not at TIES-B floor**: trained-on-Stiefel
   adapters might reach further than post-hoc projection, closing the gap.

> **Q1**: Can a single PoLAR adapter be trained with Stiefel constraint
> on B from the start without losing per-task accuracy vs unconstrained
> training?
>
> **Q2**: How does Stiefel-trained adapter compare to projected-from-
> unconstrained on the same task?

## Architecture

Single PoLAR adapter on q_proj per layer. Standard rank=6, scale=6.0.
Frozen Grassmannian A (Pierre invariant). The change:

```
Standard training:
   for batch in data:
       loss = forward + cross_entropy
       grad = grad(loss, B)
       B = B - lr * grad
       periodically: retract A to Stiefel  # already in polar_train.py

Stiefel-aware training:
   for batch in data:
       loss = forward + cross_entropy
       grad = grad(loss, B)
       B_euclid = B - lr * grad
       # Riemannian step on B:
       B = retract_to_stiefel(B_euclid)    # row-orthonormalize via QR
```

We use **row-orthonormality per-adapter** (B B^T = I_r). Joint
constraints across K adapters come in the next experiment
(`exp_pierre_joint_stiefel_b_train`).

## Riemannian SGD on Stiefel(r, d_out)

For a single B ∈ ℝ^(r × d_out):

1. Compute Euclidean gradient `g = ∂L/∂B`.
2. Project to tangent space at B:
       `T = g - B sym(B^T g)`
3. Apply step: `B_new = B - lr * T`.
4. Retract to manifold via QR:
       `Q, R = QR(B_new^T)` (transpose because we want rows orthonormal)
       `B_new = Q^T` (or `Q^T · sign(diag(R))` to fix QR sign ambiguity)

Alternative: **Cayley retraction**, smoother but more expensive. Test
both if QR has stability issues.

## Pre-registered Kill Criteria

- **K1 (CONVERGENCE)** Stiefel-trained adapter reaches its native
  benchmark within 5pp of unconstrained training.
  PASS → Stiefel constraint doesn't kill the optimization.
  FAIL → either constraint is too restrictive, or our retraction is
  unstable. Compare QR vs Cayley.

- **K2 (ORTHOGONALITY PRESERVED)** After training, `B B^T - I` Frobenius
  norm ≤ 1e-3 averaged over layers.
  PASS → retraction works.
  FAIL → numerical drift; investigate retraction frequency.

- **K3 (STEP COST)** Per-step training time increase ≤ 25% over
  unconstrained.
  PASS → QR-per-step is cheap enough.
  FAIL → batch the retractions or use a cheaper approximation.

- **K4 (VS POSTPROJ)** Stiefel-trained single-adapter accuracy ≥
  post-hoc-projected single-adapter accuracy from sibling experiment.
  PASS → trained-on-manifold wins, as expected.
  FAIL → existing training was already close enough to Stiefel that
  projection is sufficient. Mathematical interest, no architectural impact.

## Implementation status

**SPEC — implementation pending.**

Required engineering:
1. Modify `scripts/polar_train.py` to add Stiefel retraction step after B's
   optimizer update (~30 LoC).
2. Add `Stiefel_RETRACTION_EVERY_N=1` config knob (retract after every step
   or every N steps).
3. Smoke test: train 50 steps on tiny data, verify B B^T ≈ I post-step.
4. Full train: math adapter, standard hyperparameters, ~30 min on M5 Pro.
5. Eval on GSM8K + compare to existing math_polar baseline.

Estimated: **2-3h MLX impl** + ~30 min training.

## What this experiment is NOT

- Not multi-adapter or compositional. That's
  `exp_pierre_joint_stiefel_b_train` (joint constraint across K=7).
- Not Stiefel-A. That's already in Pierre via Grassmannian-A retraction.
- Not Cayley vs QR retraction benchmark (that's a follow-up if K3 fails).

## References

- Riemannian Optimization for LoRA on Stiefel (arxiv 2508.17901) — prior
  art for single-adapter Stiefel-LoRA.
- StelLA (arxiv 2510.01938).
- `scripts/polar_train.py::retract_to_stiefel` — Pierre's existing
  Stiefel retraction (applied to A only).
- Sibling: `exp_pierre_stiefel_b_postproj` (P1) — must complete first.
- Sibling: `exp_pierre_joint_stiefel_b_train` (P2) — multi-adapter joint
  constraint, runs after this validates feasibility.

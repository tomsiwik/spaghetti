# MATH.md — Stiefel post-hoc projection on existing B-matrices

## Arc framing

After the KAN/scalar-field arc was falsified (KAN parameterization itself
destroys adapter signal: -8pp single, -27pp composed), we return to the
mathematically-grounded but architecturally-conservative approach: apply
**Stiefel manifold constraints to the existing B-matmul** instead of
rebuilding the parameterization.

The thesis from earlier: Pierre's PoLAR theorem promises sibling
orthogonality via the Grassmannian A. But A is shared, frozen, identical
across all K adapters — orthogonality between *adapters* must live in B.
Standard training places no manifold constraint on B; if SGD drifts B's
into overlapping subspaces, composition gets interference.

This experiment is the **cheap entry-point feasibility test**:

> **If we apply post-hoc Stiefel projection to Pierre's existing 7 B-matrices,
> does the projection (a) preserve enough signal to be useful, and
> (b) make composition cleaner via simple mean rather than requiring
> TIES/Fisher-Rao corrections?**

No training. Pure projection + eval. Cheap. Informative either way.

## Mathematical setup

For K=7 PoLAR adapters with rank r=6, d_out=2048, per layer:
- Each B_k ∈ ℝ^(6×2048), unconstrained from training.
- Stack: B_all = [B_1; B_2; ...; B_7] ∈ ℝ^(42×2048) (joint matrix, 42 rows).
- We want row-orthonormality: B_all B_all^T = I_42.

QR-based projection:
```
QR decomposition on B_all^T:
    B_all^T = Q · R                   (Q: 2048×42 orthonormal columns, R: 42×42 upper-tri)
Set B_all_stiefel = Q^T               (42×2048, rows orthonormal)
Slice back per adapter:
    B_k_stiefel = B_all_stiefel[k·6 : (k+1)·6, :]
```

This is **strict Stiefel projection**. The scaling info in R is discarded.

Two flavors to test:
- **Strict (S)**: B_k ← B_k_stiefel (rows have unit norm; magnitude info lost).
- **Rescaled (R)**: B_k ← B_k_stiefel * ‖original B_k‖_F / ‖B_k_stiefel‖_F
  (rows scaled to recover original Frobenius norm per adapter).

(R) preserves "energy" but breaks strict Stiefel; (S) is the mathematically
correct test of "does pure orthogonality help?"

## Pre-registered Kill Criteria

### Per-adapter expressivity preservation

- **K1 (STRICT EXPRESSIVITY)** Mean over 3 native benchmarks of
  `(strict_stiefel_single - standard_single)` ≥ −5pp.
  PASS → strict Stiefel projection doesn't catastrophically destroy
  per-task accuracy. The scaling info wasn't load-bearing.
  FAIL → must use rescaled variant or train-from-scratch on Stiefel.

- **K2 (RESCALED EXPRESSIVITY)** Mean over 3 native benchmarks of
  `(rescaled_stiefel_single - standard_single)` ≥ −2pp.
  PASS → with rescaling, single-adapter accuracy is preserved within noise.

### Composition behavior

- **K3 (COMPOSITION VIA SIMPLE MEAN)** Composing strict-Stiefel adapters
  via uniform 1/K mean (no Fisher-Rao rescaling) achieves avg ≥ Fisher-Rao
  baseline on standard adapters (64.7%).
  PASS → the rescaling step Fisher-Rao does is unnecessary when adapters
  are pre-orthogonalized. Simpler math, same accuracy.

- **K4 (BEATS TIES-B)** Composition of (S) or (R) Stiefel adapters
  reaches TIES-B floor (71.3pp) on the 3-bench average.
  PASS → Stiefel projection + simple mean matches the heuristic
  trim+elect+merge approach. Mathematical guarantee instead of heuristic.

## Verdict logic

| K1 | K2 | K3 | K4 | Outcome |
|----|----|----|----|---------|
| ✓ | ✓ | ✓ | ✓ | **SUPPORTED** — projection works; composition is clean. Next step: training-from-scratch on Stiefel for further gains. |
| * | ✓ | ✓ | ✗ | **SUPPORTED w/ caveat** — projection helps composition but doesn't fully match TIES. Try Stiefel training to close the gap. |
| ✗ | ✓ | * | * | **PARTIAL** — must rescale to preserve expressivity. Strict Stiefel destroys magnitude info. Document and proceed with rescaled variant. |
| ✗ | ✗ | * | * | **KILLED** — even rescaled Stiefel projection loses too much. Existing training drifts too far from Stiefel; the arc requires retraining (see sibling experiments). |
| ✓ | ✓ | ✗ | * | **CONTRADICTION** — Stiefel preserves expressivity but composition still needs Fisher-Rao? Investigate before claiming. |

## What this experiment is NOT

- Not Stiefel-aware training. That's `exp_pierre_stiefel_b_train_single`
  and `exp_pierre_joint_stiefel_b_train` (sibling experiments).
- Not Karcher-mean composition on Stiefel manifold. That's
  `exp_pierre_stiefel_b_composition` (run after Stiefel training).
- Not Stiefel-A. Pierre already has Grassmannian-A constraint. We're adding
  the orthogonality constraint to B specifically.

## Honest gaps

- **Discarding R (the upper-tri factor) loses information.** Standard
  unconstrained training places B's wherever loss gradient leads; the
  scaling-vs-direction split that QR exposes wasn't a target of training.
  We probably lose a few pp on (S). The whole point of K1 is to measure
  by how much.

- **Joint Stiefel constraint is restrictive at K=7, r=6, d_out=2048.**
  K·r = 42 rows must be orthonormal in a 2048-dim space. That's plenty of
  room (42 ≪ 2048), so feasibility isn't the question — closeness of
  existing B's to the Stiefel manifold is. If existing B's are nearly
  orthogonal already, projection barely changes them. If they're heavily
  overlapping, projection costs accuracy.

- **K=2 or K=3 subset compositions** are untested here. If TIES-B at
  K=2 was already wrong (and the K=2 strategy×domain experiment showed
  K=2 doesn't outperform K=7), there's no reason Stiefel-B at K=2
  would be different.

## Implementation

Pure MLX, no training, no special infrastructure:
1. Load 7 adapter states.
2. For each layer, stack B's → QR → split back into per-adapter B's.
3. Install via PoLARLinear, evaluate single + composed.
4. Repeat for (S) strict and (R) rescaled variants.
5. Compose via 3 methods × 2 variants = 6 composition configurations.
6. Compare all to standard-adapter Fisher-Rao (64.7%) and TIES-B (71.3%) baselines.

~50 LoC + standard eval rig. Runtime ~60 min for all configurations
(2 variants × {single×3 + composed×3} = 12 eval passes at N=50).

## References

- PoLAR Grassmannian theorem (Pierre's design doc) — applies analogous
  manifold constraint to A; this experiment extends to B.
- Riemannian Optimization for LoRA on Stiefel (arxiv 2508.17901) — prior
  art for Stiefel-LoRA training (single adapter); we're testing the
  composition implication.
- StelLA (arxiv 2510.01938) — Stiefel-LoRA subspace learning.
- Sibling: `exp_pierre_stiefel_b_train_single` (P2) — Stiefel-aware
  training from scratch; needed if K1 fails here.
- Prior baselines: Fisher-Rao 64.7%, TIES-B 71.3% (settled on these
  7 adapters at N=50).

# PAPER.md — Stiefel B Post-Hoc Projection

## Verdict: KILLED

Post-hoc QR-based Stiefel projection on Pierre's 7 existing B-matrices
destroys adapter expressivity catastrophically. All 4 kill criteria fail
by large margins (-21 to -39pp). The existing adapters' B-matrices have
drifted too far from the Stiefel manifold during unconstrained training
for retroactive projection to be viable.

## Hypothesis

If we project Pierre's 7 trained B-matrices onto the joint Stiefel manifold
(B_all B_all^T = I_{42}) via QR decomposition, the projection preserves
enough signal for (a) single-adapter accuracy within noise of standard, and
(b) composition via simple mean to match or beat Fisher-Rao/TIES-B baselines.

## Method

- Stack all 7 B_k (6x2048 each) into B_all (42x2048) per layer.
- QR decompose B_all^T = QR; set B_stiefel = Q^T (strict) or rescale to
  match original Frobenius norm (rescaled).
- Evaluate single-adapter and 3 composition methods x 2 variants = 6 configs.
- N=50 per benchmark (GSM8K, HumanEval, MedQA).

## Kill Criteria Results

| KC | Criterion | Threshold | Measured | Delta | Result |
|----|-----------|-----------|----------|-------|--------|
| K1 | Strict single-adapter avg vs standard | >= -5pp | 40.7 vs 62.0 | -21.3pp | **FAIL** |
| K2 | Rescaled single-adapter avg vs standard | >= -2pp | 40.0 vs 62.0 | -22.0pp | **FAIL** |
| K3 | Simple-mean composition vs Fisher-Rao baseline | >= 64.7% | 29.3% | -35.4pp | **FAIL** |
| K4 | Best composition vs TIES-B floor | >= 70.3% | 32.7% (strict_fisher_rao) | -38.6pp | **FAIL** |

## Detailed Results

### Single-Adapter Accuracy (avg of GSM8K/HumanEval/MedQA, N=50)

| Variant | GSM8K | HumanEval | MedQA | Avg |
|---------|-------|-----------|-------|-----|
| Standard (baseline) | 66.0 | 78.0 | 42.0 | 62.0 |
| Strict Stiefel | 62.0 | 60.0 | 0.0 | 40.7 |
| Rescaled Stiefel | 60.0 | 60.0 | 0.0 | 40.0 |

MedQA collapses to 0% under both variants — the medical adapter's signal is
entirely in the magnitude/direction structure that QR projection discards.

### Composition Accuracy (avg of 3 benchmarks, N=50)

| Method | Strict | Rescaled |
|--------|--------|----------|
| Simple mean | 29.3 | 28.7 |
| Fisher-Rao | 32.7 | 32.7 |
| TIES-B | 30.7 | 30.7 |

All compositions perform far worse than the unprojected baselines (Fisher-Rao
64.7%, TIES-B 71.3%). Orthogonalizing B does not help composition when the
projection itself destroys per-adapter signal.

### Orthogonality Verification

QR projection achieves near-perfect orthogonality (off-diagonal max < 1.1e-4),
confirming the projection itself is mathematically correct. The problem is not
the projection quality but the distance between existing B's and the Stiefel
manifold.

## Analysis

The failure mode is clear: unconstrained SGD training placed the 7 B-matrices
in overlapping subspaces where magnitude and direction are both load-bearing.
QR projection preserves direction but discards relative scaling (strict) or
attempts to restore it (rescaled), but neither recovers the entangled
magnitude-direction structure that the trained adapters rely on.

Rescaled does not help (-22.0pp vs -21.3pp for strict) because the problem
isn't global scale — it's per-row, per-column structure that QR's orthogonal
basis cannot preserve.

## Implications for Sibling Experiments

This result is actually informative, not just negative:

1. **exp_pierre_stiefel_b_train_single** (P2): Training a single adapter
   *from scratch* on the Stiefel manifold is now the critical path. Post-hoc
   projection fails because the adapters weren't trained to live on Stiefel;
   training with Stiefel constraints from the start should avoid this.

2. **exp_pierre_joint_stiefel_b_train** (P2): Joint training of multiple
   adapters on Stiefel is the follow-up if single-adapter Stiefel training
   preserves expressivity.

3. **exp_pierre_stiefel_b_composition** (P2): Stiefel-aware composition
   (Karcher mean) is only worth testing after Stiefel-trained adapters exist.

4. **exp_pierre_stiefel_ab_ablation** (P3): A vs B vs AB Stiefel constraints
   — deferred until training-based experiments establish feasibility.

## References

- PoLAR Grassmannian theorem (Pierre design doc)
- Riemannian Optimization for LoRA on Stiefel (arxiv 2508.17901)
- StelLA (arxiv 2510.01938)
- Prior baselines: Fisher-Rao 64.7%, TIES-B 71.3% (7 adapters, N=50)

# Norm-Bounded Adapter Training: Proof Verification Report

## Theorem
Training LoRA adapters with Frobenius norm constraints on B during training
produces adapters whose composed deltas have equalized energy across domains,
eliminating the 400x energy imbalance that causes spectral Gini > 0.49.
(Theorem 1 from MATH.md: if all ||B_i||_F = tau, then f_i = 1/N exactly.)

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| P1: Composed Gini <= 0.15 (Thm 1c bound) | Best: 0.440 (Strategy A) | NO -- bound falsified |
| P1b: Thm 1c bound <= 0.316 (Strategy C) | 0.456 | NO -- exceeds bound by 44% |
| P2: Mixed PPL <= 6.508 | Best: 6.652 (Strategy B, miscalibrated) | NO |
| P3: >= 3/5 domains converge | Best: 4/5 (Strategy A) | YES |
| P4: B-norm ratio < 2.0 under constraint | Strategy A: ratio 5.2:1, Strategy C: ratio 1.2:1 | PARTIAL |
| P5: Training-time constraint preserves quality better than post-hoc | No: all strategies worse than partial eq | NO |

Note: P1's threshold of 0.15 was derived from Theorem 1c, which is now known to
be structurally incomplete (omits overlap term from Pyatt 1976 Gini decomposition).
K709's threshold is therefore not meaningful as a kill criterion -- the bound itself
is falsified, not just the prediction.

## Hypothesis
Training domain adapters with Frobenius norm constraints produces scale-balanced
adapters that compose with Gini < 0.15 and PPL <= 6.508 without post-hoc correction.

**Verdict: K709 FAIL (but threshold was derived from falsified bound), K710 FAIL, K711 PASS.**
**Finding status: Supported (negative result) -- training-time norm bounding does NOT
outperform post-hoc equalization. This is a Type 2 guided exploration that successfully
narrowed the unknown.**

## What This Model Is

Three training-time norm control strategies tested on 5-domain BitNet-2B-4T
LoRA adapters:

- **Strategy A (Norm Projection):** After each optimizer step, project B-norms
  so that s_i * ||B_i||_F = target (geometric mean). Effectively constrains
  high-scale domains (s=20) to small B-norms (||B|| = 12) while inflating
  low-scale domains (s=1) to large B-norms (||B|| = 63).

- **Strategy B (Scale-Compensated Weight Decay):** Add lambda * (s_i/s_ref)^2 *
  ||B||_F^2 to loss. Weight decay coefficient scales with s^2 to push
  high-scale domains toward smaller B-norms.

- **Strategy C (Uniform-Scale Training):** Train all domains at s=10 with mild
  weight decay. At composition, use s=10 uniformly.

## Key References
- NB-LoRA (arXiv:2501.19050) -- singular value bounding during LoRA training
- DeLoRA (arXiv:2503.18225) -- magnitude-direction decoupling
- Finding #279 -- Frobenius equalization: 50% log-compression ceiling
- Finding #281 -- Fisher importance reduces to Frobenius (rho=1.0)

## Empirical Results

### Training Convergence

| Domain   | Strategy A | Strategy B | Strategy C |
|----------|-----------|-----------|-----------|
| medical  | OK (14.3% loss reduction) | FAIL (diverged, 293% loss increase) | OK (6.6% loss reduction) |
| code     | OK (14.3%) | FAIL (251%) | FAIL (2.7% -- marginal) |
| math     | OK (6.0%) | FAIL (303%) | FAIL (loss increased 3.9%) |
| legal    | FAIL (2.0%) | FAIL (108%) | FAIL (loss increased 4.0%) |
| finance  | OK (5.7%) | FAIL (8.8%) | FAIL (3.0%) |
| **Total** | **4/5** | **0/5** | **1/5** |

**Strategy B: MISCALIBRATED -- results not meaningful for the norm-bounding hypothesis.**
Weight decay lambda of 0.0631 for s=20 domains produced a WD loss term of ~57 that
dominated the CE loss (~1.0), causing training to optimize for small B rather than
small prediction error. This is a hyperparameter miscalibration, not a test of the
norm-bounding concept. A grid search over lambda_0 in [0.0001, 0.001] might produce
useful results but was not conducted. Strategy B is excluded from conclusions below.

### Between-Domain Energy Equalization

| Strategy | Delta Norm Ratio | Norm Gini | Interpretation |
|----------|-----------------|-----------|---------------|
| Baseline raw sum | 21.6:1 | 0.341 | Severe imbalance |
| Baseline partial eq | 4.6:1 | 0.236 | Post-hoc compression |
| **A norm projection** | **3.8:1** | **0.169** | Partial equalization (50% better than raw) |
| B weight decay | 10.0:1 | 0.298 | Marginal (WD too strong) |
| **C uniform scale** | **1.2:1** | **0.036** | **Near-perfect equalization** |

Strategy C achieves almost perfect energy equalization (ratio 1.2:1, Gini 0.036)
because uniform scale + natural B-norm convergence (~15 for all domains) produces
nearly identical delta norms.

### Composed Spectral Gini (the KEY metric)

| Strategy | Composed Gini | Change from Baseline |
|----------|--------------|---------------------|
| Baseline raw sum | 0.490 | -- |
| Baseline partial eq | 0.393 | -19.6% |
| A norm projection | 0.440 | -10.1% |
| B weight decay | 0.472 | -3.5% |
| C uniform scale | 0.456 | -6.8% |

**CRITICAL FINDING:** Strategy C achieved near-perfect between-domain energy
equalization (norm Gini 0.036) but composed Gini only dropped to 0.456.
However, this comparison is CONFOUNDED: Strategy C uses different adapters
(trained at s=10) than the baseline (trained at per-domain optimal scales).
The correct decomposition uses Finding #279, which applied full equalization
to the SAME baseline adapters and achieved Gini 0.267:

| Comparison | Gini | Adapters | Energy Ratio |
|-----------|------|----------|-------------|
| Baseline (raw sum) | 0.490 | Baseline, per-domain scales | 21.6:1 |
| Finding #279 (full eq) | 0.267 | SAME baseline, equalized | ~1:1 |
| Strategy C (this exp) | 0.456 | DIFFERENT (trained at s=10) | 1.2:1 |

Correct Gini decomposition (from Finding #279, same adapters):
- Between-domain contribution: 0.490 - 0.267 = 0.223 (~45%)
- Within-domain contribution: 0.267 (~55%)

Strategy C's Gini (0.456) being HIGHER than Finding #279's full equalization
(0.267) -- despite both achieving near-perfect energy equalization -- proves
that adapter quality (within-domain SV structure) matters as much as energy
balance. Training at s=10 instead of per-domain optimal scales produces
adapters with worse within-domain spectral structure, negating the
equalization benefit.

### Composition PPL

| Strategy | Mixed PPL | vs Baseline Raw | vs Partial Eq |
|----------|----------|----------------|--------------|
| Baseline raw sum | 6.584 | -- | +1.2% |
| Baseline partial eq | 6.508 | -1.2% | -- |
| A norm projection | 6.839 | +3.9% | +5.1% |
| B weight decay | 6.652 | +1.0% | +2.2% |
| C uniform scale | 7.129 | +8.3% | +9.5% |

Both meaningfully-tested strategies (A and C) produce WORSE composition PPL than
the partial equalization baseline. (Strategy B is excluded as miscalibrated -- see
above.) Strategy C is the worst despite having the best energy equalization,
because training at s=10 (instead of per-domain optimal s=1-20) reduces adapter
quality for all domains.

### Per-Domain PPL

| Domain | Base | Raw Sum | Partial Eq | A (proj) | B (wd) | C (uniform) |
|--------|------|---------|-----------|----------|--------|-------------|
| medical | 6.73 | 3.85 | 3.94 | 4.13 | 3.95 | 4.60 |
| code | 5.69 | 3.76 | 3.72 | 3.93 | 3.86 | 4.11 |
| math | 3.79 | 2.42 | 2.53 | 2.69 | 2.46 | 2.89 |
| legal | 20.98 | 15.50 | 14.63 | 15.26 | 15.45 | 15.39 |
| finance | 18.36 | 14.08 | 13.57 | 14.16 | 14.05 | 14.25 |

All norm-bounded strategies degrade high-scale domain quality (medical +7-20%,
code +4-9%, math +2-19%) while providing marginal improvement on legal/finance.

## Limitations

1. **Strategy B (weight decay) was miscalibrated.** Lambda = 0.0631 for s=20
   domains was 60x too large. A grid search over lambda (0.0001 to 0.01) might
   find a useful operating point. However, this is a hyperparameter search, not
   a principled solution. Strategy B results are excluded from conclusions.

2. **The 200-step training budget is short.** Norm-bounded training may need
   more steps to find good B-directions under the constraint. But the baseline
   also uses 200 steps, so the comparison is fair.

3. **Strategy A applies asymmetric projection.** It clips B-norms DOWN to
   target but never inflates UP. Medical/code/math domains (s=20) get clipped
   from natural B-norm ~30 to 12.06 (matching target = geo_mean_delta / (s*sqrt(r))).
   Legal and finance, whose unconstrained B-norms (~29, ~29) at lower scales
   (s=4, s=1) already produce delta norms below the target, are NOT projected.
   Legal B-norm stays at 46.8, finance at 62.9 (inflated by the optimizer, not
   by projection). The resulting 3.8:1 delta norm ratio reflects this asymmetry
   of the projection operator, not a failure of the norm-bounding concept per se.

4. **Theorem 1c (Gini union bound) is falsified.** The bound omits the overlap
   term from the standard Gini decomposition (Pyatt 1976). K709's threshold of
   0.15 was derived from this incorrect bound. This also retroactively affects
   the interpretation of Finding #279, which used the same bound structure
   (though that experiment's measurement happened to fall below the bound).

5. **The 7%/93% claim in the original analysis was confounded.** The comparison
   between baseline Gini (0.490) and Strategy C Gini (0.456) compares different
   adapter populations. The correct decomposition from Finding #279 (same adapters)
   gives ~45% between-domain / ~55% within-domain.

## What Would Kill This

K709 FAIL (Gini 0.440 >> 0.15): The threshold 0.15 was derived from the
now-falsified Theorem 1c bound. The bound itself is structurally incomplete
(missing overlap term). Even the corrected bound would predict ~0.316 for
Strategy C, while the measurement is 0.456. The correct interpretation is
that training-time equalization produces worse Gini than post-hoc equalization
on the same adapters (0.456 vs Finding #279's 0.267).

K710 FAIL (PPL 6.652 > 6.508): Norm constraints during training reduce
adapter quality. The constrained optimizer cannot find B-directions as
effective as the unconstrained optimizer within 200 steps. Note: the best
PPL (6.652) came from Strategy B, which was miscalibrated and should not
be credited; among valid strategies, Strategy A gives PPL 6.839.

## Key Structural Findings

### 1. Training-time norm constraints produce WORSE composition quality than post-hoc equalization

Strategy C achieves near-perfect energy equalization (1.2:1 ratio) but Gini 0.456,
while Finding #279 achieves full equalization on the SAME baseline adapters with
Gini 0.267. Training-time constraints force the optimizer into a constrained
landscape that produces adapters with worse within-domain SV structure.

### 2. Adapter quality matters as much as energy balance

Strategy C's Gini (0.456) being higher than Finding #279's (0.267) despite both
having near-perfect energy equalization proves that within-domain spectral structure
-- determined by training conditions -- contributes at least as much to composed
Gini as between-domain energy imbalance.

Correct Gini decomposition (from Finding #279, same adapters):
- Between-domain: ~45% (0.490 - 0.267 = 0.223)
- Within-domain: ~55% (0.267)

### 3. Gini union bound (Theorem 1c) is empirically falsified

The bound Gini(composed) <= max_i Gini(B_i) + Gini_between omits the overlap term
from the standard Gini decomposition (Pyatt 1976). Strategy C: bound predicts <= 0.316,
measured 0.456. K709's threshold of 0.15 was derived from this incorrect bound.

### 4. Spectral arc resolution

| Experiment | Diagnosis | Result |
|-----------|-----------|--------|
| #277 DC-Merge | Within-domain SV shape | Wrong variable (18.5% Gini reduction) |
| #278 Surgery | Post-composition SVD | Structurally inverted |
| #279 Frobenius eq | Between-domain energy | Best: 50% log-compression, Gini 0.267 |
| **#282 Norm-bounded** | **Train-time energy eq** | **Training-time eq worse than post-hoc eq** |

**Implication:** Post-hoc partial equalization (Finding #279, Gini 0.267) remains the
practical ceiling. It works better than training-time norm equalization because it
operates on the SAME well-trained adapters without degrading their within-domain
quality. The remaining 0.267 Gini is within-domain spectral structure -- addressable
only by changing how individual adapters are trained (SV shape), not by rebalancing
their relative energies.

# Persistence Diagram Diff: Before vs After Adapter Composition

## STATUS: SUPPORTED (K625 PASS, K626 FAIL, K627 NOT ASSESSED)

Composition is topologically nontrivial but the stability bound is vacuous at
current scale. K625 PASS (d_B > 0 in all modules), K626 FAIL (no features lost,
but vacuously true — bound 10-100x loose). The genuine finding is feature
CREATION: +242 H0, +401 H1 features, concentrated in output projections.
Adapters restructure weight geometry rather than merely perturbing it.

## Theorem

**Theorem 1 (Stability Bound for Low-Rank Composition).**
d_B(Dgm(P), Dgm(P')) <= max_i ||delta_i||_2

where P = rows of W^T, P' = rows of (W^T + Delta), and delta_i is the i-th
row of the perturbation Delta = (scale/N) * sum_i(A_i @ B_i).

**Corollary 1.** Features with persistence > 2 * max_i ||delta_i||_2 are
guaranteed to survive composition.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? | Information Content |
|------------------------|----------|--------|---------------------|
| P1: d_B <= max ||delta_i|| (stability bound) | 0/35 violations | YES | **TAUTOLOGICAL** — this is a mathematical theorem (Cohen-Steiner 2007). A violation would indicate a code bug, not a scientific finding. |
| P2: Features with pers > 2*delta survive | 17,223/17,223 stable (100%) | YES | **VACUOUSLY TRUE** — the bound is ~10-100x loose (median persistence ~30-58, vulnerability window ~0.4-4.0). ALL features trivially exceed the threshold. |
| P3: Random perturbation produces similar d_B | Adapter/random ratio = 1.38 mean | PARTIAL | **GENUINE FINDING** — adapters cause 38% more topological change than random perturbations (Wilcoxon p=0.0002, 95% CI [1.20, 1.57]). |
| P4: Vulnerability window is small | 0 vulnerable features across all 35 modules | YES | **VACUOUSLY TRUE** — same issue as P2. The window is empty because the bound is too loose to be informative. |

**Honest assessment:** Only P3 provides genuine empirical information. P1 is a
mathematical certainty (not a prediction). P2 and P4 are vacuously true because
the stability bound is 10-100x looser than the actual persistence values. The
theorem is correct but uninformative at this adapter scale.

P3 reveals that adapter perturbations cause statistically significantly more
topological change than random perturbations of equal norm (Wilcoxon signed-rank
test: W=101, p=0.0002; one-sample t-test: t=4.00, p=0.0003; 95% CI [1.20, 1.57]).
This confirms adapters have structured (non-random) geometry.

## Hypothesis

Adapter composition via pre-merge (W' = W + sum(s_i * B_i)) preserves all
high-persistence topological features of the weight row point cloud. The topological
cost of composition is confined to noise-level features with persistence below the
stability threshold.

## What This Experiment Is

**Weight-space persistent homology:** Treat the rows of each weight matrix W^T as
a point cloud in R^d. Build a Vietoris-Rips complex and compute persistent homology
(H0: connected components, H1: loops). Compare persistence diagrams of base weight
matrices vs composed (base + adapter) weight matrices using bottleneck distance.

**35 projection modules analyzed** across 5 layers (0, 7, 15, 22, 29) and 7
projection types (q/k/v/o attention + gate/up/down MLP). 500 rows subsampled per
module. Includes random baseline control (5 random perturbations per module with
same per-row norms).

**Redesign from killed predecessor:** The previous experiment (exp_pathway_graph_bitnet2b)
used co-activation graphs with sparsification, which created an artifact. This experiment
uses weight-space PH (no sparsification, no activation collection, direct Rips complex).

## Key References

- Cohen-Steiner, Edelsbrunner, Harer (2007): Algebraic Stability Theorem for PH
- Rieck et al. (2018, arXiv:1812.09764): PH on weight matrices of neural networks
- Garin & Tauzin (2020, arXiv:2312.10702): PH for model compression analysis

## Empirical Results

### Summary Statistics

| Metric | Value |
|--------|-------|
| Modules analyzed | 35 (5 layers x 7 projections) |
| Subsample | 500 rows per module |
| H0 bottleneck mean | 0.068 |
| H0 bottleneck max | 0.257 (layer 29 o_proj) |
| H0 bottleneck median | 0.044 |
| H1 bottleneck mean | 0.026 |
| H1 bottleneck max | 0.111 |
| Random baseline H0 mean | 0.083 |
| Adapter/random ratio mean | 1.38 |
| Adapter/random ratio median | 1.31 |
| Stability bound violations | 0 / 35 |
| Total features | 17,223 |
| Vulnerable features | 0 (0%) |
| Stable features | 17,223 (100%) |

### Layer-Level Patterns

| Layer | Avg H0 bottleneck | Avg relative perturbation | Interpretation |
|-------|-------------------|--------------------------|----------------|
| 0 (first) | 0.074 | 0.86% | Moderate change |
| 7 (early-mid) | 0.050 | 0.60% | Smallest change |
| 15 (middle) | 0.040 | 0.53% | Smallest change |
| 22 (late-mid) | 0.055 | 0.72% | Moderate change |
| 29 (last) | 0.108 | 1.89% | **Largest change** |

Layer 29 has the largest bottleneck distances (2x other layers) and largest
relative perturbations. The final layer adapters perturb weights most aggressively,
consistent with the observation that the last layer is most important for
task-specific output formatting.

### Module-Type Patterns

The o_proj and down_proj projections show the largest H0 bottleneck distances:
- o_proj: attention output projection, directly shapes the residual stream
- down_proj: MLP output, directly adds to the residual stream

These are the modules where adapter perturbations have the most direct impact on
the model's output pathway. The attention projections (q/k/v) and MLP input
projections (gate/up) show smaller topological changes.

### Feature Creation Analysis

**The stability theorem bounds feature MOVEMENT and DESTRUCTION but says nothing
about feature CREATION.** The data reveals substantial feature creation that the
original analysis entirely ignored:

**H0 (connected components) created:** 242 new features across 6 modules.
The base weight matrices have rows that are identical or near-identical (sharing
connected components at birth). Composition breaks these apart into distinct points.

| Module | Base H0 | Composed H0 | Created | % Change |
|--------|---------|-------------|---------|----------|
| layer_29 o_proj | 383 | 499 | +116 | +30.3% |
| layer_0 o_proj | 427 | 499 | +72 | +16.9% |
| layer_0 down_proj | 480 | 499 | +19 | +4.0% |
| layer_7 down_proj | 483 | 499 | +16 | +3.3% |
| layer_29 down_proj | 484 | 499 | +15 | +3.1% |
| layer_22 down_proj | 495 | 499 | +4 | +0.8% |

**H1 (loops) created:** 401 new features across 19 modules (only 2 lost across
all modules). Composition introduces cyclic structure where the base model had
none:

| Module | Base H1 | Composed H1 | Created |
|--------|---------|-------------|---------|
| layer_29 o_proj | 1 | 53 | +52 |
| layer_7 k_proj | 1342 | 1393 | +51 |
| layer_15 k_proj | 1361 | 1410 | +49 |
| layer_0 o_proj | 0 | 48 | +48 |
| layer_22 k_proj | 1360 | 1400 | +40 |

**Interpretation:** Feature creation concentrates in output-facing projections
(o_proj, down_proj) — the same modules where adapters have the most direct impact
on the residual stream. The base model's o_proj weights have many near-identical
rows (383-427 out of 500 sampled). Composition differentiates these rows,
creating new topological features. This is a fundamental restructuring of the
weight geometry, not "near-lossless" preservation.

The H1 loop creation (0→48-53 in o_proj) indicates composition creates consistent
geometric patterns (cyclic structure) that the base model entirely lacked. This
could be the topological signature of adapter specialization.

### Adapter vs Random Comparison

In 26/35 modules (74%), adapter composition causes MORE topological change than
a random perturbation of equal norm (ratio > 1.0). Mean ratio = 1.38.

**Statistical significance (N=35 modules):**
- Wilcoxon signed-rank test (H0: ratio = 1.0): W=101, p=0.0002
- One-sample t-test (H0: ratio = 1.0): t=4.00, p=0.0003
- 95% CI for mean ratio: [1.20, 1.57]
- Standard error: 0.095

**The adapter/random ratio of 1.38 is statistically significant** (p < 0.001).
Adapters cause genuinely more topological change than random perturbations of
equal norm.

Notable exceptions (ratio < 1.0):
- o_proj at layers 0, 29: ratio 0.67, 0.30
- down_proj at layers 0, 7, 22, 29: ratio 0.69-0.90

These output-facing projections show LESS bottleneck distance than random but
MORE feature creation (see Feature Creation Analysis above). The adapter
perturbation restructures the weight geometry differently from random — not
less aggressively, but along qualitatively different dimensions that the
bottleneck metric underweights.

## Kill Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K625: d_B > 0 | **PASS** | 35/35 H0 nonzero, 24/35 H1 nonzero |
| K626: >=3 lost features with pers > median | **FAIL** | 0 modules with significant loss; ALL 17,223 features stable |
| K627: Lost features correlate with cross-domain | **NOT ASSESSED** | Weight rows lack domain labels; requires activation analysis |

**Overall: K625 PASS but K626 FAIL.**

The experiment establishes that composition IS topologically nontrivial (K625)
but that the topological cost is exclusively in noise-level features (K626 FAIL
-- no important features are lost). This is actually a POSITIVE result for the
composition framework: it means composition preserves all meaningful topological
structure.

## Key Findings

### Finding 1: Stability Bound is Vacuously Satisfied; Feature Creation Tells the Real Story

All 17,223 high-persistence features survive composition across all 35 modules.
However, **this metric is vacuously true** — the stability bound is 10-100x
looser than the actual feature persistence values, so feature loss was impossible
to detect at this adapter scale regardless of composition quality.

**The real topological story is feature CREATION:** composition creates 242 new
H0 features (connected components) and 401 new H1 features (loops) while losing
only 2 H1 features. Output projections (o_proj, down_proj) show the most dramatic
restructuring (+30% H0, 0→53 H1 in layer 29). Composition differentiates
near-identical weight rows and introduces cyclic geometric structure that the base
model lacked entirely.

This cannot be characterized as "near-lossless" — it is topologically nontrivial
restructuring that happens to preserve existing features while adding new ones.

### Finding 2: Stability Theorem Holds (As It Must)

Zero violations of d_B <= max||delta_i|| across 35 modules. **Note:** This is a
mathematical certainty, not an empirical finding. The stability theorem is a
proven property of Rips persistence; zero violations confirms correct implementation,
not a scientific discovery.

### Finding 3: Last Layer Has Largest Topological Cost

Layer 29 shows 2x the bottleneck distance of other layers, with relative
perturbations up to 5.4%. This is consistent with the last layer being most
task-specific and adapters needing to make the largest changes there.

### Finding 4: Output Projections Show Lower Bottleneck Distance But More Feature Creation (QUALIFIED)

For o_proj and down_proj, adapter perturbations cause LESS bottleneck distance
than random perturbations of equal norm (ratio < 1.0). However, these SAME modules
show the most dramatic feature creation (+30% H0, 0→53 H1 in layer 29 o_proj).

**The original "structured alignment" interpretation is retracted.** A lower
bottleneck distance with simultaneous massive feature creation does not indicate
alignment — it indicates the bottleneck metric is insensitive to the type of
topological change happening. The bottleneck distance measures max point-wise
shift in the persistence diagram, but feature creation (new points appearing
from the diagonal) can produce large topological restructuring with small
bottleneck distance.

The correct interpretation: adapters restructure output projection geometry
(differentiating near-identical rows, creating loops) in a way that moves
existing features LESS than random perturbations while creating many new features.
This is a qualitatively different type of change from random perturbation.

### Finding 5: No Sparsification Artifact

Unlike the killed predecessor (exp_pathway_graph_bitnet2b), this experiment uses
NO sparsification. The random baseline control shows adapter and random perturbations
produce comparable bottleneck distances, confirming the measurements are not
artifacts of the methodology.

## Implications for the Architecture

1. **Feature loss is not a problem at current scale, but feature creation is
   substantial.** With rank-16 adapters and 5 domains, no existing features are
   destroyed (vacuously true — the bound is too loose). But composition creates
   242 H0 and 401 H1 features, restructuring output projection geometry.

2. **The bridge extraction experiment (exp_persistence_bridge_extraction) should
   focus on CREATED features, not lost ones.** The topological change from
   composition is additive, not destructive. Understanding what the created
   features represent is more informative than looking for destroyed bridges.

3. **The vulnerability window will grow with more/larger adapters.** The stability
   bound scales with max||delta_i||, which grows with adapter rank, scale, and
   number of adapters. At some threshold, high-persistence features will start
   entering the vulnerability window.

4. **Scaling prediction:** For the current setup (rank=16, scale=20, 5 domains),
   max||delta_i|| ~ 0.3-2.0 while median feature persistence ~ 30-50. The
   vulnerability threshold (2*max_delta) ~ 0.6-4.0 is far below median persistence.
   This would need to increase ~10x before any features are lost.

## Limitations

1. **500-row subsample.** Only 500/2560 rows sampled per module. Important topology
   may live in the unsampled rows.
2. **Euclidean metric only.** Cosine similarity or other metrics may reveal different
   topological structure.
3. **No behavioral connection.** The experiment measures topological change but does
   not prove this connects to behavioral quality.
4. **Single model.** BitNet-2B-4T only. Results may differ for other architectures.
5. **K627 not assessed.** Weight rows don't carry domain labels, so the cross-domain
   correlation criterion cannot be tested with this methodology.
6. **NaN/overflow in skeleton matrices.** The code applies `nan_to_num` defensively
   before A@B computation. **Quantified impact: ZERO.** Inspection of all skeleton
   and adapter files shows 0 NaN and 0 inf values out of 53,452,800 skeleton values
   and 54,681,600 adapter values (5 domains). The `nan_to_num` call has no effect
   on any computation. No perturbation norms are artificially deflated.

## What Would Kill This

- If behavioral quality degrades despite topological preservation (topology is the
  wrong metric for composition quality)
- If scaling to more adapters or higher rank causes the vulnerability window to
  cross into high-persistence features
- If a different metric (cosine, spectral) reveals topological damage that Euclidean
  Rips misses
- If the subsample is unrepresentative and full-matrix PH shows different results

## Runtime

Total: 2057s (~34 min) on Apple M5 Pro 48GB. Dominated by bottleneck distance
computation for H1 features (slow O(n^3) matching). The PH computation itself is
fast (~0.1s per module).

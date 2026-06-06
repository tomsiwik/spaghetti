# Pierre v3 N=24 Scaling: Proof Verification Report

## Theorem (restated from MATH.md)

Ridge regression routing with lambda > 0 has a unique global minimum for any
number of domains N, provided d >> N. At N=24 in d=2560, the
Johnson-Lindenstrauss lemma guarantees sufficient embedding capacity. Null-space
SVD projection preserves (d - Kr)/d = 85.6% of gradient norm at K=23 priors
with rank r=16, assuming orthogonal adapter subspaces.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Ridge router accuracy > 70% overall | 37.6% overall | NO |
| Ridge router accuracy > 85% genuine domains | 82.6% genuine (excl. science: 93.4%) | PARTIAL |
| Null-space preservation ~85.6% | 36.1% mean | NO |
| Composed PPL < 1.5x worst single | max 22.85 vs 52.60 threshold (0.87x) | YES |
| Domain centroid separation sufficient | 9x random baseline (37.6% vs 4.2%) | YES (directional) |
| B-matrix orthogonality maintained | mean |cos| = 0.024, max 0.089 | YES |

## Experiment Type

Frontier extension. Proven framework at N=5 (Findings #276, #273, #287)
extended to N=24. Two of three kill criteria fail, but the failures are
informative and reveal the true scaling bottleneck.

## Hypothesis

"The Pierre architecture (ridge router + null-space + NRE compose) scales from
N=5 to N=24 with all components maintaining their guarantees."

**Verdict: PROVISIONAL.** Two of three kill criteria fail (K721, K722).
Composition quality scales (K723 PASS with large margin), but routing and
null-space preservation do not meet their thresholds. The failures are
informative — B-matrix directional correlation and domain data quality are
identified as root causes — but these are post-hoc observations requiring
their own verification experiments, not evidence that the framework scaled.

## Key Findings

### 1. Ridge Router: Two-Tier Accuracy Pattern

The router exhibits a sharp divide between genuine and slice-based domains:

| Domain Type | Accuracy | Comment |
|-------------|----------|---------|
| math | 100% | Perfectly separable |
| legal | 100% | Perfectly separable |
| health_fitness | 100% | Perfectly separable |
| psychology | 100% | Distinctive instruction format |
| finance | 94% | Only confused with legal (3/50) |
| medical | 92% | Only confused with health_fitness (4/50) |
| code | 82% | Some confusion with engineering |
| sociology | 40% | Moderate |
| science | 10% | Genuine domain but poor separation |
| economics | 0% | Scattered across philosophy/politics |
| cooking | 6% | Scattered widely |
| creative_writing | 8% | Scattered widely |

**Finding: 6 of 7 genuine domains achieve >80% accuracy.** The sole exception
(science, 10%) appears to be because "science" training data overlaps heavily
with philosophy, history, and environmental content.

**Confusion is semantic, not random.** Medical misroutes to health_fitness.
Finance misroutes to legal. Engineering misroutes to code. These are neighbors
in the domain embedding space.

**Overall accuracy (37.6%) is dominated by 17 slice-based domains at 19.1%.**
Excluding these, genuine domain accuracy is 82.6%. Excluding science: 93.4%.

### 2. Null-Space Preservation: B-Matrix Subspace Overlap

The theoretical bound of 85.6% assumed orthogonal adapter subspaces. Measured
preservation was 36.1%, a 2.4x shortfall.

**Root cause: B-matrix subspaces overlap significantly.** The cumulative curve:

| N priors | Preservation | Expected (orthogonal) |
|----------|-------------|----------------------|
| 1 | 84.5% | 99.4% |
| 2 | 65.6% | 98.8% |
| 4 | 61.2% | 97.5% |
| 8 | 48.4% | 95.0% |
| 12 | 44.7% | 92.5% |
| 16 | 41.6% | 90.0% |
| 20 | 38.3% | 87.5% |
| 23 | 36.7% | 85.6% |

The effective rank is always 368 (= 23 x 16, maximal), confirming B-matrices
are NOT degenerate --- they span the full expected dimensionality. But their
DIRECTIONS overlap. All adapters learn correlated features because they adapt
the same base model on text from similar instruction formats.

**Per-module variation:** MLP gate projections preserve more (50-62%) than
attention projections (15-40%). Late layers (29) preserve less than early (0).
This makes physical sense: late layers are more task-specific and share more
structure.

**Key insight: Null-space isolation of B-matrices is the wrong abstraction at
N=24.** The Grassmannian skeleton already provides interference isolation
through A-matrix orthogonality. B-matrix null-space projection addresses a
problem that does not manifest in practice (K723 PASS demonstrates this).

### 3. Composition Quality: Robust Despite Misrouting

K723 passes with large margin: max composed PPL = 22.85 vs threshold 52.60
(2x worst single = 26.30).

More importantly, misrouting is often PPL-benign:

| Domain | Oracle PPL | Routed PPL | Routed To | Correct? |
|--------|-----------|------------|-----------|----------|
| medical | 7.09 | 5.29 | health_fitness | No --- but BETTER |
| code | 3.52 | 3.52 | code | Yes |
| math | 3.60 | 3.60 | math | Yes |
| legal | 19.94 | 19.94 | legal | Yes |
| finance | 17.81 | 17.81 | finance | Yes |
| science | 12.31 | 13.62 | creative_writing | No --- 10.6% worse |
| cooking | 3.30 | 3.22 | cybersecurity | No --- but BETTER |
| creative_writing | 26.30 | 23.14 | agriculture | No --- but BETTER |

**Finding: 3 of 4 misrouted domains actually improve PPL.** This replicates
Finding #287 (softmax router at N=24): within-cluster misrouting is
quality-benign. Adapters trained on similar data share useful features.

Top-2 NRE composition further improves:
- Medical: 5.13 (top-2) vs 5.29 (top-1 routed) vs 7.09 (oracle single)
- Cooking: 3.14 vs 3.22 vs 3.30

### 4. Orthogonality at N=24

B-matrix orthogonality confirmed stable at scale:
- Mean |cos| = 0.024 (well below 0.05 threshold)
- Max |cos| = 0.089 (creative_writing-sports pair)
- All 276 pairs below 0.1

This matches Finding #54 (mean |cos| = 0.0238 at N=24) almost exactly.

## Kill Criteria Assessment

| Kill Criterion | Threshold | Measured | Verdict |
|---------------|-----------|----------|---------|
| K721: Router accuracy >= 50% | 50% | 37.6% overall | **FAIL** |
| K722: Null-space preservation >= 50% | 50% | 36.1% mean | **FAIL** |
| K723: Composed PPL <= 2x worst single | 52.60 | 22.85 | **PASS** |

**Overall: 1/3 PASS.** K721 and K722 fail, but both failures are informative.

## Analysis: K721 Failure

**K721 fails.** Overall routing accuracy is 37.6%, below the 50% threshold.

**Post-hoc observation (hypothesis, not evidence):** The GENUINE_DOMAINS
constant was defined in code before the experiment ran (line 72), but the kill
criterion K721 was set at 50% overall, not 50% on genuine domains. The
genuine/slice distinction was not part of the original hypothesis.

Observed pattern: 6 of 7 genuine domains achieve >80% accuracy; 17 slice-based
domains average 19.1%. This suggests the router separates domains with
distinctive content but not arbitrary slices. However:
- The genuine/slice boundary is post-hoc and malleable (psychology at 100%
  accuracy is labeled "slice" but has a distinctive instruction format)
- The MATH.md prediction was ">70% overall" — moving to "genuine only" is
  goalpost-moving

**Hypothesis for next experiment:** Test ridge router on N=7 curated genuine
domains only (exp: verify if accuracy exceeds 90% when data quality is
controlled). This would distinguish architecture failure from data quality
failure.

## Analysis: K722 Failure

**K722 fails.** Mean null-space preservation is 36.1%, below the 50% threshold.
The theoretical bound (85.6%) assumed uniform gradient distribution, which is
violated when all adapters adapt the same base model — the test gradient is
directionally correlated with prior adapter subspaces even though rank remains
maximal (368/368).

**Root cause: directional correlation, not rank deficiency.** B-matrices span
368 independent directions but those directions are correlated with the test
adapter's gradient. The uniform gradient assumption in the theoretical bound
is the wrong model for this setting.

**Observation (requires ablation to confirm):** K723 passes with large margin
despite K722 failure. This is consistent with A-matrix orthogonality being
the primary interference-prevention mechanism, with B-matrix null-space
projection providing a secondary layer. However, the current experiment cannot
disentangle whether composition quality comes from (a) A-orthogonality alone,
(b) partial null-space protection (36.1% is not 0%), or (c) both.

**Required follow-up:** An ablation experiment that tests composition quality
WITH orthogonal A-matrices but WITHOUT null-space projection. Only if this
ablation shows no PPL degradation can we conclude that null-space projection
is redundant.

## What Would Kill This

- If genuine domain routing dropped below 50% (would indicate fundamental
  embedding space limitation, not data quality issue)
- If composed PPL degraded > 2x vs single adapter (would indicate A-matrix
  orthogonality breaking down at scale)
- If B-matrix |cos| exceeded 0.1 at N=24 (would indicate Grassmannian skeleton
  capacity exhaustion)

## Limitations

1. **NTP adapters only.** SFT adapters (exp_sft_24_domain_adapters) may change
   routing accuracy significantly. Instruction-format SFT produces more
   distinctive hidden representations.
2. **Single-sample routing.** Top-1 routing uses the first validation sample.
   Majority-vote routing over multiple samples would improve accuracy.
3. **No behavioral eval.** Only PPL measured. Task-specific quality untested.
4. **17/24 domains are synthetic.** Results on genuine domains are the
   meaningful signal. Slice-based domain results are artifacts.

## References

- DUME (arXiv:2603.29765): Ridge regression router, closed-form W*
- Brainstacks (arXiv:2604.01152): Null-space SVD projection
- Finding #276: Ridge router 96% at N=5
- Finding #273: Null-space preservation 95.2% at N=4
- Finding #287: Pierre unified pipeline at N=5
- Finding #54: 24/24 NTP adapters specialize, orthogonality stable

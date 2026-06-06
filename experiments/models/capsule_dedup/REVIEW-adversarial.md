# Peer Review: Capsule Deduplication (Exp 8)

## NotebookLM Findings

Skipped -- NotebookLM deep review was not performed for this experiment
due to the experiment being a clear negative result with well-identified
kill criteria. The mathematical and experimental analysis below was
conducted directly from source materials.

---

## Mathematical Soundness

### 2.1 Cosine Similarity as Redundancy Detector: CORRECT with Caveats

The core claim (MATH.md Section 2.1) is that for rank-1 capsules with
ReLU gating, cos(a_i, a_j) > tau implies near-identical activation
patterns. The proof sketch is sound:

- Capsule i fires iff a_i^T x > 0, defining half-space H_i.
- If cos(a_i, a_j) = 1, then H_i = H_j exactly.
- The disagreement fraction is proportional to arccos(tau) / 180 for
  isotropically distributed inputs.

**Verified**: The angle calculations are correct:
```
tau = 0.90 => arccos(0.90) = 25.84 deg => disagreement ~ 14.4%
tau = 0.95 => arccos(0.95) = 18.19 deg => disagreement ~ 10.1%
tau = 0.99 => arccos(0.99) =  8.11 deg => disagreement ~  4.5%
```

The paper claims "at most 10% of inputs" at tau=0.95. The bound is
theta/180 * 100% = 10.1%. Correct, but this is for the FULL sphere.
For d=64, two nearly-aligned half-spaces have an even smaller
disagreement region (the ratio of spherical cap area shrinks with
dimension). The stated bound is loose (conservative), which is fine
for a sufficiency argument.

**One hidden assumption worth noting**: The claim that cosine similarity
IS sufficient for rank-1 capsules (Section 2.2) is correct only when
both a_i vectors have similar norms. If ||a_i|| >> ||a_j||, capsule i
will produce much larger activations even when pointing in the same
direction, making the b-sum rule inexact. The paper acknowledges this
in Section 3.2 but then dismisses it ("norms tend to be similar after
identical training procedures"). This dismissal is empirically
plausible but unverified -- no norm statistics are reported.

### 3.1 The a-Average, b-Sum Merging Rule: CORRECT

The derivation is clean and well-justified:

```
output_unmerged = b_i * (a_i^T x) + b_j * (a_j^T x)
               ~ (b_i + b_j) * (a^T x)   [when a_i ~ a_j]
```

Therefore b_merged = b_i + b_j preserves the additive contribution.
The error bound ||a_merged - a_i|| = sqrt((1-tau)/2) is correctly
derived from the cosine law:

```
||a_j - a_i||^2 = ||a_i||^2 + ||a_j||^2 - 2*a_i^T*a_j
               = 1 + 1 - 2*tau = 2(1-tau)     [unit vectors]
||a_merged - a_i|| = ||(a_j - a_i)/2|| = sqrt((1-tau)/2)
```

At tau=0.95: sqrt(0.05/2) = sqrt(0.025) = 0.158. Confirmed.

**Important subtlety the paper handles correctly**: averaging a but
summing b. If you averaged b, the merged capsule would produce half the
expected output magnitude. This is a common mistake in expert merging
literature, and the paper explains it clearly.

### 4.2 Cluster-Based Merging via Connected Components: MINOR CONCERN

The connected-components approach can create transitivity chains:
if cos(a_i, a_j) > tau and cos(a_j, a_k) > tau but cos(a_i, a_k) < tau,
all three get merged. For tau = 0.90, two hops could accumulate up to
~52 degrees of angular deviation (2 * 25.8), meaning the merged capsule
could fire on a very different input region than one of its constituent
capsules.

At tau = 0.95, two hops give ~36 degrees max, and at tau = 0.99, ~16
degrees. In practice the clusters found are tiny (mostly pairs), so
this risk did not materialize. But the algorithm does not guard against
it, and at larger P with more capsules, long chains could form. This
is a known issue with single-linkage clustering. The paper should
acknowledge it.

### Section 6: Expected Redundancy Estimate: HONEST

The paper correctly identifies the key unknown: "whether 54% shared
knowledge corresponds to 54% shared capsules." The experimental result
(1.9%) decisively answers "no." The paper's pessimistic estimate of
5% was close to the actual 1.9%.

### Section 7: Worked Numerical Example: VERIFIED

I spot-checked the cosine similarity values in the 4-dimensional
toy example:

```
a_0 = [0.8, 0.2, 0.0, 0.1], a_4 = [0.79, 0.22, 0.01, 0.09]
cos = (0.8*0.79 + 0.2*0.22 + 0.0*0.01 + 0.1*0.09) /
      (sqrt(0.69) * sqrt(0.6726))
    = (0.632 + 0.044 + 0.0 + 0.009) / (0.8307 * 0.8201)
    = 0.685 / 0.6812 = 1.006...
```

Wait -- this gives >1.0, which is impossible. Let me recheck:
||a_0||^2 = 0.64 + 0.04 + 0 + 0.01 = 0.69, ||a_0|| = 0.8307
||a_4||^2 = 0.6241 + 0.0484 + 0.0001 + 0.0081 = 0.6807, ||a_4|| = 0.8251
dot = 0.632 + 0.044 + 0 + 0.009 = 0.685
cos = 0.685 / (0.8307 * 0.8251) = 0.685 / 0.6854 = 0.9994

The paper claims 0.999, which is consistent (rounding). The example
is essentially correct.

---

## Novelty Assessment

### Prior Art

1. **PuzzleMoE (2024)**: Uses cos > 0.95 threshold for expert component
   merging. The paper cites this correctly. The deduplication threshold
   choice is not novel.

2. **SERE (2024)**: Uses Frobenius norm of weight differences for expert
   re-routing. Related but uses a different distance metric.

3. **Sub-MoE (2024)**: SVD-based expert merging with activation-based
   clustering. More sophisticated than cosine on raw weights.

4. **BuddyMoE (2024)**: Co-activation profiling. Behavioral rather than
   weight-space similarity.

### Novelty Delta

The novelty is modest but appropriate for a micro-experiment:
- Applying cosine dedup specifically to rank-1 ReLU capsules (where cosine
  IS functionally exact) rather than multi-layer experts (where it's a
  proxy).
- The negative finding is the contribution: shared knowledge is distributed,
  not concentrated at the capsule level.
- The dead capsule discovery (60%) is a genuine insight that reframes the
  problem.

**Assessment**: The experiment's value is primarily in the negative result
and the "distributed vs concentrated" insight, not in the dedup mechanism
itself. This is appropriate for micro-scale research.

---

## Experimental Design

### Does the experiment test the stated hypothesis? YES

The hypothesis is: "redundant capsules (cos > 0.95) can be identified
and merged to reduce parameter count without quality loss."

The experiment:
1. Correctly composes models by concatenation (validated in relu_router)
2. Sweeps three thresholds (0.90, 0.95, 0.99)
3. Tests both cross-pool and all-pairs modes
4. Includes proper controls (joint, concat, weight averaging)
5. Runs 3 seeds with aggregate statistics
6. Reports both quality impact AND redundancy statistics

### Controls: ADEQUATE

- Joint training as upper bound
- Unmerged concatenation as lower bound for dedup to beat
- Weight averaging as the practical alternative
- Dedup + calibration to test whether the gap is fixable

### Potential Confound: Dead Capsule Measurement

The 60% dead capsule finding deserves scrutiny. The measurement method
(`check_capsule_death` in `test_dedup.py`) runs 10 batches of 32 tokens
each through the model, checking if each capsule fires at least once.

**10 batches * 32 samples * 32 tokens = 10,240 activation checks per
capsule.** For a 256-capsule model, a capsule that fires with 1% base
rate would fire ~102 times across the test set. A capsule that fires
with 0.01% base rate would fire ~1 time. So 10 batches can reliably
detect capsules with >0.1% activation rate but may miss very-low-frequency
capsules.

The paper acknowledges this in Limitation 5: "Some 'dead' capsules may
fire on rare inputs not in the sample." This is fair. However, the
magnitude (60%) is so large that even if 10% of "dead" capsules are
false negatives, the conclusion holds: the vast majority of composed
capsules are inactive.

**More important question**: Is the 60% dead capsule rate an artifact
of the composition method or a real property? In a composed model with
2 domains and P=128 per domain (256 total), domain A's capsules were
trained on a-m names and domain B's on n-z names. When evaluating on
the joint dataset (both domains), you would expect ~50% of capsules
to be inactive for any given input (the "wrong domain" capsules do not
fire). But 60% > 50%, suggesting some capsules are dead even for their
own domain.

This could be:
1. Natural ReLU dead neuron phenomenon (some neurons never get positive
   pre-activation for any input in the training distribution)
2. Composition artifact (concatenation changes the input distribution
   to later layers via residual connections from earlier composed layers)
3. Overfitting during fine-tuning (200 steps may push some detectors
   to very narrow input regions)

The paper identifies sources 1-3 but does not disentangle them. A
simple ablation would be: check dead capsule rate in the individual
domain models BEFORE composition. If each domain model already has
10-15% dead capsules, then composition would produce ~55-60% dead
(10-15% native dead + ~50% wrong-domain). This ablation is missing
but would strengthen the analysis.

### Code Quality

**`capsule_dedup.py`** (277 lines): Clean, well-documented. The
`CapsuleDedupGPT` class is an empty subclass of `ReLURouterGPT`, which
is the right design -- dedup is a post-processing step, not an
architectural change. The utility functions (`cosine_similarity_matrix`,
`find_redundant_clusters`, `merge_capsules`, `deduplicate_composed_model`)
are well-separated and independently testable.

**Minor issue**: `cosine_similarity_matrix` uses `1e-8` epsilon in the
norm computation, which could produce values slightly above 1.0 when
two vectors are exactly parallel. The `find_redundant_clusters` function
uses `>` (strict greater than) for thresholding, so cos = 1.0000001
would still pass. Not a bug in practice but could be tightened with
`mx.clip(S, -1.0, 1.0)`.

**`test_mechanism.py`** (208 lines): Good unit test coverage. Tests:
- Cosine similarity correctness (orthogonal and parallel cases)
- Cluster detection (within-pool vs cross-pool)
- Exact output preservation (identical a vectors)
- Approximate output preservation (similar a vectors)
- Forward pass survivability after dedup

Missing: no test for the transitivity chain issue (multi-hop clusters
where endpoints are far apart).

**`test_dedup.py`** (403 lines): Comprehensive experiment runner.
Well-structured with 7 stages, 3-seed aggregation, and automatic
kill-threshold analysis. The dead capsule analysis is integrated
properly.

### Test Coverage: GOOD

The tests cover the mechanism thoroughly. The experiment code is
well-organized with clear separation of concerns. One gap: no test
for the `_get_pool_index` helper function, though it is simple enough
to verify by inspection.

---

## The "Distributed vs Concentrated" Insight

This is the most valuable finding. The paper's reconciliation table:

| Metric | Procrustes (Exp 3) | Cosine Dedup (This) |
|--------|-------------------|---------------------|
| Granularity | Matrix-level | Neuron-level |
| Shared fraction | 54% | 1.9% |

The interpretation -- "shared knowledge is distributed across many
neurons as small perturbations, not concentrated in a few neurons as
large changes" -- is well-supported by the data:

1. Mean cross-pool cosine of 0.296 indicates capsules point in
   moderately different directions (not orthogonal, not parallel).
2. If knowledge were concentrated, you would see a bimodal distribution:
   many pairs at cos~0 and a cluster at cos~1. Instead, the distribution
   is unimodal around 0.3.
3. The Procrustes 54% shared delta can be reconstructed as: each of 128
   capsules has ~54% of its weight change in the shared direction, but
   the remaining ~46% is unique, making each capsule's TOTAL direction
   distinct.

**This insight has real implications**: it explains why weight averaging
works (+1.1% vs joint) but bottom-up capsule matching fails. Averaging
implicitly handles the distributed shared component by blending ALL
capsules, while matching only catches the rare concentrated duplicates.

**One weakness**: The paper does not verify this interpretation directly.
A simple verification would be to decompose each capsule's fine-tuning
delta (a_i_finetuned - a_i_base) into shared and unique components
(as Procrustes does at the matrix level) and show that the shared
component is small per-capsule but large in aggregate. This would
close the loop between the two experiments.

---

## Macro-Scale Risks (advisory, not blocking)

1. **More redundancy at larger P**: With P=1024+ capsules per domain,
   the probability of two capsules landing in similar directions
   increases combinatorially. At P=1024, there are 1M cross-pool pairs
   vs 16K at P=128. Even if the per-pair redundancy rate stays at 0.1%,
   that is 1000 redundant pairs -- potentially significant. The micro
   result may not transfer.

2. **Different domains may change the picture**: a-m vs n-z names share
   character distributions extensively. Python vs JavaScript (macro
   domains) have more distinct token vocabularies, which could produce
   either more redundancy (shared programming concepts) or less
   (distinct syntax capsules). The direction is unpredictable.

3. **Dead capsule pruning as the real opportunity**: The 60% dead
   capsule finding is more actionable than deduplication. At macro scale,
   pruning dead experts from a composed MoE is a well-studied problem
   (expert pruning in Mixtral, DeepSeek). The micro experiment should
   pivot to testing this.

4. **Activation-based similarity (BuddyMoE-style)**: Weight-space
   cosine may systematically underestimate functional redundancy. Two
   capsules with cos(a_i, a_j) = 0.5 could have nearly identical
   activation patterns on the actual data distribution (if the data
   lies in a subspace aligned with both). At macro scale, behavioral
   dedup may find much more to merge.

---

## Verdict

**PROCEED**

This is a well-executed negative result. The experiment cleanly
falsifies the hypothesis that shared knowledge at the matrix level
(54% per Procrustes) manifests as shared capsule detectors at the
neuron level (only 1.9%). The kill criteria are honestly applied,
the controls are adequate, and the code is clean.

The "distributed vs concentrated" insight is genuinely valuable for
the project's direction. It explains a pattern across multiple
experiments (Procrustes decomposition fails because shared knowledge
is entangled; capsule dedup fails because it is distributed; weight
averaging succeeds because it handles both). This meta-finding
advances the understanding needed for macro-scale design.

### Minor Recommendations (not blocking)

1. **Add dead-capsule baseline**: Measure dead capsule rate in each
   domain model BEFORE composition. This disentangles natural ReLU
   death from composition artifacts. Estimated effort: 5 minutes,
   ~10 lines of code in `test_dedup.py`.

2. **Report norm statistics**: The a-average merge rule assumes similar
   norms. Report mean, std, and range of ||a_i|| across pools and
   layers. Estimated effort: 3 minutes.

3. **Acknowledge transitivity risk**: Add one sentence to MATH.md
   Section 4.2 noting that connected-component clustering can merge
   capsules with low pairwise similarity through chains.

4. **Verify the distributed interpretation**: Decompose per-capsule
   deltas into shared/unique components and show the per-capsule
   shared fraction is small but aggregates to 54%. This would make
   the "distributed not concentrated" claim quantitative rather than
   qualitative.

None of these are blocking. The experiment has reached its conclusions
and the findings are recorded properly in PAPER.md and FINDINGS.md.

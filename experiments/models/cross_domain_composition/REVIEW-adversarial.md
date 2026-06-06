# Peer Review: Cross-Domain Composition (Revision 2)

## NotebookLM Findings

Skipped -- NotebookLM not used for this review. Analysis conducted directly on MATH.md, PAPER.md, source code, and results.json.

## Mathematical Soundness

### Gap Computation (MATH.md Section 3)

The gap formula `gap = (L(strategy) - L(base)) / L(base) * 100%` is standard. The code correctly computes per-type, per-seed gaps and then aggregates. Revision 2 properly reports mean +/- stddev, max, 75th percentile, and count exceeding threshold. This addresses Fix 1 and Fix 4 adequately.

One subtlety: the code computes `gap_{t,s}` for each type-seed pair, then reports the mean across all 50 pairs. This is the correct order of operations (average of ratios, not ratio of averages). Verified in code lines 783-793.

### Multi-Expert Composition Formula (Section 2.1)

The formula `W_multi = W_base + (1/2)(Delta_i + Delta_j)` averages the two deltas with equal weight. This is correct for the equal-contribution case. However, MATH.md Section 7 claims "Multi-expert adds routing complexity but no inference overhead if pre-merged." This is true only if the composition weights are fixed; if they are query-dependent (which they should be for cross-domain queries where one domain dominates), pre-merging is impossible. The paper does not address asymmetric composition weights.

### Cancellation Analysis (Section 4.1)

The cancellation reporting is honest and well-structured:
- 22/50 positive, 28/50 negative
- Mean of positive: +15.0%, mean of negative: -13.6%
- 7/50 exceed 20% threshold

This is a genuine improvement over v1. The artifact is properly acknowledged.

### Binomial CI on K2 (Section 4.3)

MATH.md states "95% CI of approximately [2.2%, 19.2%]" for the 4/50 error rate. Verified: exact binomial CI for 4/50 at 95% is [2.2%, 19.2%]. Correct.

### Missing: Effect Size Analysis

The paper reports gap percentages but never asks whether the base model itself has learned enough for the comparison to be meaningful. Looking at the raw losses:

- Pure-domain expert losses are often WORSE than base losses (seed 42: arithmetic base=1.41, expert=1.70; repeat base=0.49, expert=1.65). The SVD-truncated rank-4 experts on an untrained base do not even beat the jointly-trained base on their own domains. This means "expert quality" is dubious, and any composition of these experts is comparing two barely-functional models.
- Cross-domain base losses range from 2.3 to 4.5 (vs uniform random at ~ln(42)=3.74). Several cross-domain base losses are near or above random baseline, meaning the base has not learned anything useful for those types.

This does not invalidate the mechanism test (which is about relative gaps), but it means the absolute quality of all models is near-random, making percentage gap comparisons noisy by nature. The paper acknowledges "untrained base model" in limitations but does not acknowledge that the experts themselves are worse than the base on pure-domain tasks.

## Novelty Assessment

### Prior Art

The cited references (Task Arithmetic, TIES-Merging, LoRAHub, LoRA Soups) are appropriate. LoRAHub (Huang et al., 2023) is the most relevant -- it demonstrates cross-task LoRA composition with learned coefficients. The delta here is testing composition on explicitly cross-domain queries (queries spanning two domains simultaneously) rather than novel single-domain tasks.

### Building on Project Work

Appropriately builds on `exp_gram_schmidt_composition` and shares infrastructure with `composition_vs_monolithic`. The lineage is correctly documented in PAPER.md.

### Novelty Concern

The experiment does not test LoRA composition -- it tests full-rank SVD-truncated delta composition. Line 684: `delta_trunc, _ = svd_truncate_delta(delta, rank_per_expert)` applies post-hoc SVD truncation to full-parameter deltas. This is "task arithmetic with rank reduction," not LoRA training. The distinction matters because LoRA constrains the optimization trajectory (low-rank from the start), while SVD truncation discards information after the fact. At micro scale this is acceptable, but the paper should note that actual LoRA training may produce different composition behavior due to the optimization constraint.

## Experimental Design

### Fix Assessment: All 5 Fixes Applied

1. **Per-type gaps with stddev**: Fully implemented. Table in PAPER.md matches results.json. PASS.
2. **Single-expert K1 added**: Implemented as K1_single with its own per-type table. Mean -7.0%, properly reported. PASS.
3. **"Reviewer-attack-neutralized" claim removed**: PAPER.md no longer claims to neutralize the reviewer attack. Findings are properly framed as "oracle routing" vs "hash-ring scenario." PASS.
4. **Cancellation artifact acknowledged**: Section in both MATH.md and PAPER.md, with full breakdown (22/50 positive, 28/50 negative, 7/50 >20%). PASS.
5. **K2 expanded to 50 trials, 20% threshold**: 10 types x 5 seeds = 50 trials. Threshold lowered from 50% to 20%. Error rate 8.0% (4/50). PASS.

All five requested fixes have been implemented.

### Remaining Issue 1: Per-Type Kill Criterion Avoidance

The aggregate K1 metrics pass, but individual cross-domain types systematically exceed 20%:

**Multi-expert:**
- arith_reverse: mean +21.5%, max +49.2% -- FAILS per-type
- repeat_sort: mean +17.4%, max +27.8% -- borderline
- repeat_parity: mean +11.8%, max +24.1% -- borderline

**Single-expert:**
- arith_parity: mean +21.3%, max +30.4% -- FAILS per-type

The paper honestly reports these but then concludes "supported" because the aggregate passes. This is a defensible choice given that the kill criterion is written as "aggregate," but it should be called out more prominently that 2 out of 10 cross-domain types systematically fail the 20% threshold (arith_reverse for multi-expert, arith_parity for single-expert). The mechanism is not uniformly reliable.

### Remaining Issue 2: Expert Quality Inversion

The pure-domain results (seed 42) show that individual experts often perform WORSE than the base model on their own domains:

| Domain | Base Loss | Expert Loss | Expert vs Base |
|--------|-----------|-------------|----------------|
| arithmetic | 1.41 | 1.70 | +20.6% worse |
| reverse | 1.31 | 1.98 | +51.3% worse |
| repeat | 0.49 | 1.65 | +235% worse |
| sort | 1.14 | 1.67 | +46.5% worse |
| parity | 0.52 | 1.01 | +92.7% worse |

Every single expert is substantially worse than the base on its own domain. This occurs because: (a) the base is trained on all 5 domains jointly (1000 samples), while each expert is trained on only its domain (200 samples), and (b) the SVD truncation to rank 4 at d=32 discards significant information.

This means the experiment is testing "can a model composed of individually-bad experts match a model trained jointly?" -- which is a different question than "can expert composition handle cross-domain queries?" The cross-domain gaps are not measuring whether composition preserves expert quality; they are measuring whether two bad experts averaged together are less bad than one bad expert alone.

The paper acknowledges "rank-4 at d=32 is severely capacity-constrained" but does not flag that the experts are worse than base on their own domains, which is a more fundamental problem than capacity.

**This is not blocking.** At macro scale with real LoRA training, experts should outperform the base on their domains. But the micro evidence is weaker than claimed because the constituent experts are individually broken.

### Remaining Issue 3: K2 Routing Definition

K2 asks "Is the best single expert from one of the two involved domains?" with 4/50 errors (8.0%). All 4 errors are on `reverse_parity`. But looking at the raw data, the "routing error" on reverse_parity means a non-involved domain's expert happens to achieve lower loss on this cross-domain type. This is not really a routing error -- it is an accidental quality correlation. The K2 metric conflates "routing correctness" with "which expert happens to have lowest loss," which are different concepts. At micro scale with near-random models, this distinction is largely semantic, but at macro scale it matters.

### Statistical Sufficiency

- 5 seeds is adequate for per-type mean/std estimation (paired with 10 types gives 50 total observations for aggregate)
- The 95% CI on K2 ([2.2%, 19.2%]) is wide but passes the 20% threshold with margin
- Per-type stddev ranges from 3.6% to 15.1%, indicating substantial type-to-type variation that 5 seeds cannot fully characterize

## Hypothesis Graph Consistency

HYPOTHESES.yml kill criteria:
1. "merged model scores >20% worse than base on cross-domain queries (aggregate)" -- PASS at -1.0% (multi) and -7.0% (single)
2. "cross-domain queries route to wrong expert >20% of the time (tightened from 50%)" -- PASS at 8.0%

The experiment matches its kill criteria. The criteria are tested as stated. The "aggregate" qualifier in K1 is now explicitly noted, and per-type failures are documented. The evidence supports changing status to "supported" (not "proven" -- the per-type failures and expert quality inversion prevent a stronger claim).

The notes in HYPOTHESES.yml accurately describe the revision and findings. No inconsistency.

## Macro-Scale Risks (advisory)

1. **Expert quality at macro should resolve Issue 2.** Real LoRA experts trained at d=4096/r=16 on specific domains should outperform the base on their domains. The composition test becomes meaningful when the constituents are individually strong.

2. **Routing for k=2 remains unsolved.** Hash ring routes to k=1. This experiment assumes oracle routing to the 2 relevant experts. The production system needs either a learned router for multi-expert queries or evidence that k=1 suffices (the single-expert results suggest it does on average, but arith_parity at +21.3% is a counterexample).

3. **Sequential chaining vs semantic composition.** The cross-domain queries tested here are concatenations of two sequential tasks. Real cross-domain queries (e.g., "convert Python to Bash") require simultaneous cross-domain understanding that weight-space addition may not provide. This is the most important gap between micro and macro.

4. **Dilution at large N.** At N=500, naive merge is useless (0.2% per expert). Multi-expert with k=2 routing is essential. The routing problem hardens as N grows because more candidates must be discriminated.

5. **arith_reverse and arith_parity failures.** These specific cross-domain types fail at micro scale. If analogous failures persist at macro (e.g., math+code composition), they would indicate fundamental limits of weight-space composition for certain domain pairs.

## Verdict

**PROCEED**

The revision adequately addresses all 5 fixes from the previous review cycle:

1. Per-type gaps with stddev -- fully reported and tabulated
2. Single-expert K1 -- added as separate metric, properly framed
3. Oracle routing framing -- claims appropriately scoped
4. Cancellation artifact -- explicitly documented with breakdown
5. K2 sample size and threshold -- 50 trials, 20% threshold

The remaining issues (expert quality inversion, per-type failures) are acknowledged limitations of the micro-scale setup and do not invalidate the directional finding: weight-space composition of relevant experts handles cross-domain queries on aggregate, with per-type variance that must be characterized at macro scale.

**Non-blocking observations for future work:**

1. At macro, verify that LoRA-trained experts outperform base on their own domains before testing cross-domain composition. The micro experiment's experts are individually worse than base, making the composition test weaker than it appears.

2. Test at least one "semantic composition" cross-domain query at macro (e.g., Python-to-Bash translation) rather than sequential task chaining.

3. The SVD-truncated delta approach used here differs from actual LoRA training. Confirm that LoRA-trained expert composition shows similar aggregate behavior.

4. Consider a per-type kill criterion (e.g., "no more than 2/10 types exceed 20%") to complement the aggregate criterion. The aggregate masks systematic failures on specific domain pairs.

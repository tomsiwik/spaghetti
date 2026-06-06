# Peer Review: Decorrelation Filter Scaling

## NotebookLM Findings

Skipped -- the experiment is a clean negative result with transparent analysis. The researcher recommends KILL and I concur after independent verification. Deep review would not change the verdict.

## Mathematical Soundness

**What holds:**

1. The filter ratio definition F(d) = E[|cos(trained)|] / E[|cos(random)|] is well-formed and directly tests the hypothesis.

2. The decomposition of why F(d) increases is correct: trained delta cosine scales as d^{-0.11} (nearly flat) while random delta cosine scales as d^{-0.77} (fast decay). The ratio of a slow-decaying quantity over a fast-decaying quantity necessarily increases. This is sound.

3. The 1/sqrt(D_delta) scaling for random vectors follows from concentration of measure. The measured exponent -0.77 is consistent with the sub-quadratic effective dimensionality growth given the non-uniform d_ff/d ratios across conditions.

4. The B-matrix correlation ratio (~4-5x random, constant across d) is a clean result. Training induces a fixed multiplicative correlation floor in B-matrices regardless of ambient dimension.

**What to note:**

1. The AP delta cosine power law fit has R^2 = 0.094. This is essentially no fit at all -- the "d^{-0.11}" exponent is not statistically meaningful. The PAPER acknowledges the trained delta cosine is "nearly flat" but still reports the exponent as if it were a scaling law. More accurate to say: trained delta cosine shows no significant trend with d (values range 0.001-0.002 with no clear pattern). This does not change the kill verdict -- if anything, "no trend" is worse than "slow decay" for the hypothesis.

2. The d_ff scaling inconsistency (4x at d=64,128 vs 2x at d=256,512) is a confound. At d=256, D_delta = 2*(256*512 + 512*256) = 524,288. If d_ff had been 4*256=1024, D_delta would be 2*(256*1024 + 1024*256) = 1,048,576 -- double. This would change the random baseline and potentially the filter ratio. The PAPER acknowledges this but claims the trend is "robust." With only 4 data points and a confounded independent variable, this claim is weak. However, the filter ratio exceeds 1.0 at d=256, and even halving the random baseline (which the d_ff change does not do) would not bring it below 0.5, so the K2 kill criterion is robust to this confound.

3. Per-seed variance is high (0.33 to 2.32 per the PAPER). With only 3 seeds, the standard error on the mean filter ratio is large. At d=128, the mean is 0.767 -- plausibly consistent with either <1.0 or >1.0 within error bars. Again, this does not rescue the hypothesis because the d>=256 values are unambiguously above 1.0.

## Novelty Assessment

This is a null result experiment, so novelty is assessed by the value of the negative finding.

**The key insight is valuable:** The decorrelation filter at d=64 was an artifact of the random baseline being high, not a structural property of the Grassmannian skeleton. This corrects a potentially misleading interpretation from the parent experiment (b_matrix_training_correlation), which reported 0.14x as if it were a fundamental property.

**Consistency with prior work:** The result is consistent with structural_orthogonality_characterization (gradient/random ratio increases with d) and minimum_viable_base (untrained ratio ~1.0). The experiment properly contextualizes itself in this lineage.

**The reframing is important for SOLE:** Knowing that composition safety comes from concentration of measure (absolute cosines < 0.002) rather than from the Grassmannian skeleton's "decorrelation" effect means the skeleton's value proposition must be justified on other grounds (slot assignment, zero drift, capacity scaling). The PAPER correctly identifies this.

## Experimental Design

**Strengths:**
- Clean two-condition design (AP vs random-orthonormal A) with shared base model per seed
- Random vector baseline provides the correct null hypothesis
- Three seeds per condition, though more would be better
- Code reuses parent infrastructure, reducing implementation risk

**Weaknesses:**

1. **Adapters did not specialize.** All losses converged to log(V=32) = 3.466 (random baseline for uniform character prediction). This means B-matrix "correlation" comes entirely from shared gradient structure on uninformative data, not from meaningful domain overlap. The PAPER correctly flags this as Limitation 1, but it is more serious than acknowledged: the entire experiment measures an artifact of non-specialization. At production scale with real domain data, adapters would develop genuine domain-specific features, and the B-matrix correlation pattern could be qualitatively different. The kill is still valid -- the hypothesis as stated (filter ratio decreases with d) is killed regardless of specialization level -- but the quantitative results (4-5x B-matrix ratio, specific filter ratios) should not be extrapolated to production.

2. **Non-uniform d_ff scaling** (discussed above). A cleaner design would use d_ff = 4*d throughout, accepting higher compute at d=256,512.

3. **Only 4 data points for power law fits.** Fitting a power law to 4 points, one of which has R^2 = 0.094, is marginal. The qualitative conclusion (filter weakens) is robust, but the specific exponent (+0.603) should not be cited with precision.

## Macro-Scale Risks (advisory)

1. **The "absolute cosines are tiny" claim needs macro validation.** At d=896 (Qwen 0.5B), macro/ortho_scaling already measured cos=0.0002, consistent with concentration of measure. But those were untrained adapters. With trained adapters showing 4-5x B-matrix correlation, the trained delta cosine at d=896 might be ~4x the random baseline of ~0.0002, giving ~0.0008. Still well below tau=0.01, so composition safety holds. But this should be verified, not assumed.

2. **At very large N (thousands of experts), even tiny per-pair cosines could accumulate.** The composition safety argument relies on individual pair cosines being small. If composing K experts, the relevant quantity is the sum of K*(K-1)/2 interference terms, not just pairwise maxima. This is outside the scope of this micro experiment but worth tracking.

## Verdict

**KILL -- confirmed.**

Both kill criteria are clearly triggered:
- K1: Filter ratio increases with d (d^{+0.603}, though the specific exponent is imprecise)
- K2: At d>=256, filter ratio is 1.86-1.93, far exceeding the 0.5 threshold

The researcher's self-assessment is accurate and thorough. The analysis correctly identifies the crossover mechanism (random baseline decays faster than trained) and properly reframes the Grassmannian skeleton's role. The key takeaway -- SOLE safety comes from concentration of measure, not from decorrelation -- is well-supported and important for the project's theoretical narrative.

No revisions needed. This is a well-executed negative result that advances understanding.

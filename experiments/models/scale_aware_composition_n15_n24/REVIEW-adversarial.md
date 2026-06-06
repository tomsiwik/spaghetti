# Peer Review: Scale-Aware Composition at N=15

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (per-domain scale selection, inherited from N=5). No stacking. CLEAN.
- Is MATH.md a proof or a description? **Description dressed in equations.** There is no Theorem/Proof/QED block. The SNR derivation in Section C is a mechanism analysis with reasonable math, but it never states a theorem or concludes with QED. It is an informal argument for why optimal scales should be N-independent under oracle top-1.
- Metric used as evidence: Format-correctness score. NOT proven to predict behavioral outcome. PAPER.md acknowledges this ("scoring metric captures format compliance but not content quality").
- Kill criteria source: K636 threshold "<=1/N domains" is derived from self-stabilization theory. K637 threshold "within 2x for >=3/5" is a reasonable tolerance but not formally derived from the SNR argument. MIXED.

## Self-Test Audit

1. **One-sentence impossibility property:** "Under oracle top-1, N has zero effect on the active adapter's output." This is a clear, single property. PASS.
2. **Cited theorems:** Weyl's inequality (real theorem, correctly cited). Finding #233 (internal finding, not a published theorem -- acceptable for micro). LIMA citation is tangential. PASS with note.
3. **Predicted numbers:** "Under oracle top-1: optimal scales identical to N=5 (shift = 0x)." This is specific and falsifiable. PASS.
4. **Falsification condition:** "If under oracle top-1, optimal scales shift by >2x between N=5 and N=15." Correctly targets the theory. PASS.
5. **Hyperparameter count:** 0. Per-domain scales are the existing mechanism. PASS.
6. **Hack check:** No new fixes. Validation experiment. PASS.

Self-Test is complete and honest. PASS.

## Mathematical Soundness

### What holds:
- The SNR derivation in Section C is algebraically correct for the multi-adapter 1/N averaging case. The sqrt(N/(N-1)) -> 1 limit is valid.
- The key insight in Section F (lines 150-161) is correct and self-critical: under oracle top-1, there IS no background noise; the only effect of N is the 1/N normalization on the single active adapter, meaning effective scale = s/N.

### What does not hold:
1. **The prediction contradicts the implementation.** MATH.md correctly derives that oracle top-1 WITH 1/N averaging should see effective scale = s/N, requiring s to scale linearly with N. But the Self-Test (line 188-189) and prediction P1 claim "optimal scales identical to N=5 (shift = 0x)." This is only true if oracle top-1 does NOT apply 1/N averaging. The code in `phase_oracle_top1_sweep` applies only the single adapter with no 1/N normalization -- so the prediction of "identical scales" is correct FOR THE CODE, but the MATH.md's own Section F contradicts it by showing s_eff = s/N under oracle + averaging.

2. **No formal theorem.** The SNR analysis is a back-of-envelope calculation, not a proof. For Type 2 guided exploration, this is acceptable IF the framework (Finding #233, Weyl) is correctly cited. The framework IS cited. Acceptable for Type 2.

3. **The N-independence claim is trivially true by construction.** If oracle top-1 applies exactly one adapter with no normalization, then N literally cannot affect the output -- it is not even a parameter of the computation. The "prediction" that scales are identical is not testing a theorem; it is testing whether the implementation correctly ignores N. This makes K637 a software correctness test, not a science experiment.

### BLOCKING ISSUE: No Theorem/Proof/QED structure
MATH.md contains a mechanism description with equations. For Type 2, this requires stating the PROVEN framework and identifying the UNKNOWN. The framework (self-stabilization) is stated. The unknown (do N=5 scales generalize?) is stated. This passes Type 2 requirements, though barely.

## Prediction vs Measurement

### The prediction table exists. PASS on form.

### The measurements are uninterpretable. FAIL on substance.

**Critical finding: Identical scores across scales reveal broken instrumentation.**

From results.json, oracle top-1 results:
- Medical s=1 and s=2: IDENTICAL score arrays [0.097, 0.146, 0.143]
- Code s=1, s=2, s=4: IDENTICAL score arrays [0.047, 0.0, 0.0]
- Legal s=1, s=2: IDENTICAL score arrays [0.022, 0.028, 0.055]
- Finance s=1, s=2: IDENTICAL score arrays [0.218, 0.06, 0.0]

From averaging results:
- Finance: ALL FIVE SCALES produce IDENTICAL score arrays [0.223, 0.069, 0.006]
- Legal s=1 through s=8: IDENTICAL score arrays
- Medical s=1, s=2, s=4: IDENTICAL; s=8, s=20: IDENTICAL (different group)
- Code s=1 through s=8: IDENTICAL

**This means the adapter is having NO EFFECT at low scales.** Likely explanation: the ternary quantization (STE with `lora_b` initialized to zeros, then overwritten) combined with small scale values produces perturbations below the model's representational resolution. When lora_out = (x @ lora_a) @ b_ste * scale, if scale * ||b_ste|| is below floating-point significance relative to base_out, the output is identical to base.

**This has a devastating consequence for the entire experiment:** When multiple scales produce identical outputs, the "optimal scale" is determined by whichever of the identical scores happens to be listed first (or last, depending on argmax tie-breaking). The N=5 "optimal" scales of s=20 for math/code/medical were presumably measured with different data or more prompts. Now at N=15 with 3 prompts, the scale sweep is measuring noise in the 2-3 scales that actually produce different outputs, and reporting ties as "optimal = lowest tied scale."

**Math domain: 0.000 at ALL scales, ALL compositions, INCLUDING base.** This domain provides zero information. It should be excluded from kill criteria assessment. With math excluded, K636 becomes 0/4 degraded (still passes), but K637 becomes 1/4 within 2x (still fails). The paper correctly notes this but does not exclude math from the denominators.

### Statistical power of 3 prompts

With n=3, the standard error on a proportion is sqrt(p(1-p)/3). For p=0.5, SE=0.29. The 95% CI on any score is approximately +/-0.57 -- wider than the entire scale range for most domains. For medical oracle: best=0.136, worst=0.120, delta=0.016. The SE is ~0.013. The "optimal" scale at s=8 vs s=20 is within 1 SE. This is noise.

For code oracle: the story is binary. s<=4 produces ~0.015 (base-level), s>=8 produces ~0.52. This is a genuine signal -- the adapter "turns on" at some threshold. But it tells us nothing about optimal scale; it tells us about minimum effective scale.

## NotebookLM Findings
[Skipped -- proceeding with manual review given clear findings above.]

## Novelty Assessment

The MATH.md Section F insight -- that oracle top-1 under 1/N averaging requires scale to grow linearly with N -- is genuinely useful for deployment design. However, this insight was derived analytically BEFORE the experiment and is trivially true. No experiment was needed to validate it.

The empirical finding that "composition at N=15 still works (4/5 domains beat base)" is a useful data point but is not novel -- it follows directly from oracle top-1 not involving N in the computation.

## Macro-Scale Risks (advisory)

1. The binary on/off behavior of code adapter (works at s>=8, dead at s<=4) may indicate a minimum perturbation threshold for ternary adapters. Worth investigating at macro scale.
2. The format-correctness metric provides no signal about content quality. Macro experiments need behavioral evaluation.
3. Math scoring 0 everywhere suggests the evaluation pipeline or the math adapter itself is broken. This must be diagnosed before any scale-related conclusions about math domain.

## Verdict

**REVISE**

The experiment is not interpretable in its current form due to critical instrumentation issues. SUPPORTED is not justified.

### Required fixes:

1. **Diagnose identical-score-across-scales bug (BLOCKING).** When oracle medical s=1 and s=2 produce byte-identical outputs, either: (a) the adapter is not loading, (b) the ternary quantization snaps small perturbations to zero, or (c) there is a caching/seed issue. Until this is diagnosed, no scale-sensitivity claim is valid in either direction.

2. **Increase to minimum 10 prompts per domain (BLOCKING for any scale-sensitivity claim).** With 3 prompts, the experiment can detect binary on/off transitions (code adapter) but cannot distinguish between scales within the "working" regime. The paper correctly identifies this but then claims SUPPORTED status anyway.

3. **Exclude math domain from kill criteria denominators or diagnose the 0-score issue.** Math provides zero signal. Including it inflates both pass and fail counts depending on the criterion.

4. **Downgrade status to PROVISIONAL at best.** One kill criterion failed. The passing criterion (K636) is trivially satisfied because oracle top-1 by construction does not involve N. The failing criterion (K637) failed due to measurement noise, not because the theory is wrong -- but "failed due to noise" is not "supported." The honest status is: **the experiment was underpowered to test its own predictions.**

5. **Add a Theorem/Proof/QED block or reclassify.** The MATH.md contains good analytical reasoning but no formal proof. Either add a formal statement or explicitly classify as "mechanism analysis" rather than proof.

6. **Report the actual useful finding.** The binary on/off behavior of ternary adapters at different scales (code dead at s<=4, alive at s>=8) is more interesting than the scale-stability question. This suggests a minimum effective scale threshold that depends on adapter norm, not on N. This should be the headline finding.

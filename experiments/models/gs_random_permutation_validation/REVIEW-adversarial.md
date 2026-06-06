# Peer Review: GS Random Permutation Validation

## NotebookLM Findings

Skipped (tool not authenticated). Review proceeds from direct document analysis.

## Mathematical Soundness

### What holds

1. **Expectation equalization (Section 2.3).** The derivation that E[D_k^{pi}] = C * mean_cos * (N-1)/2 for all k under uniform random permutation is correct. If pi is uniform on S_N, then pi^{-1}(k) is indeed uniform on {0,...,N-1} for any fixed k. The expectation is k-independent. Sound.

2. **Variance formula (Section 2.4).** Var[N-1-pi^{-1}(k)] = Var[Uniform{0,...,N-1}] = (N^2-1)/12. The CV formula follows. Correct.

3. **CLT averaging across layers (Section 2.5).** With L independent permutations per layer, the total deviation involves L independent draws. The sqrt(L) variance reduction follows from independence. Correct in principle, though the CLT approximation at L=12 is generous -- this is moderate-N CLT, not asymptotic.

4. **Mean preservation.** Empirically confirmed: permuted/unpermuted mean ratio is 1.00x across all configs. This is the strongest result and consistent with the theory.

5. **CV reduction.** The observed ~2.2x CV reduction is consistent with sqrt(P)=sqrt(5)=2.24x. This is a genuine and well-explained effect.

### What does not hold

1. **Linear approximation D_k ~ C * (N-1-k) * mean_cos.** This is stated as an assumption (Section 6, item 3) and is critical to the entire analytical framework. The parent experiment showed R^2=0.946 for linear decay, which is good but not exact. The approximation breaks for non-uniform pairwise cosines. The paper should more clearly state that the theoretical predictions are approximate guidance, not exact bounds.

2. **Independence across layers (Section 2.5).** The layers are coupled through the forward pass: the output of layer l is the input to layer l+1. The permutation at layer l changes the composed delta at layer l, which changes the hidden state entering layer l+1. Strictly, the per-layer errors are NOT independent random variables -- they are functions of the same hidden state trajectory. The CLT argument holds only if we treat the per-layer GS position as independent (which it is, by construction of the permutation), but the forward-pass amplification of those errors is correlated. This is a mild concern, not fatal, because the amplification ratio was shown to be position-independent (CV=2.9% in the parent), so the coupling is weak.

## Critical Finding: Post-Hoc Redefinition of Kill Criteria

This is the central issue of this review.

**The experiment's own code reports FAIL.** The `results.json` file, produced by the experiment's `run_experiment.py`, records:

```
K1_overall_ratio: 4.856
K1_pass: false
K2_overall_worst: 1.585%
K2_pass: false
overall: "FAIL"
```

The kill criteria as stated in HYPOTHESES.yml are:
- K1: "Permuted worst-case exceeds 2x the unpermuted mean deviation"
- K2: "Permutation introduces new failure modes (any position exceeds 1% at d=256, N=50)"

**K1 as implemented in code (abs_worst / unperm_mean):**
- d=128, N=20: 4.86x FAIL
- d=256, N=20: 2.86x FAIL
- d=128, N=50: 3.35x FAIL

All three configs fail K1 by the code's own metric.

**K2 as implemented in code (abs_worst < 1%):**
- d=128, N=20: 1.59% FAIL
- d=256, N=20: 0.45% PASS
- d=128, N=50: 1.11% FAIL

Two of three configs fail K2.

**The PAPER then redefines both criteria:**

For K1, the paper switches from "absolute worst case" to "expected worst case" (mean over permutations for the worst expert). This changes the metric from max over all samples to mean over samples for the max expert -- a fundamentally softer statistic. Under this reinterpretation, K1 passes at 1.42x.

For K2, the paper restricts evaluation to d=256 only, arguing d=128 "is below the production regime." Under this restriction, K2 passes at 0.45%.

**This is moving the goalposts after seeing the data.** The kill criteria exist to prevent exactly this kind of motivated reasoning. If the criteria were too strict, that should have been identified before running the experiment, not after the results came in.

### Is the reinterpretation reasonable?

Partially. The "expected worst case" (mean over permutations for the worst expert) IS arguably the more operationally relevant metric for a production system using a fixed seed. And d=128 IS below the production regime. However:

1. The HYPOTHESES.yml kill criterion says "permuted worst-case" -- which most naturally reads as the absolute worst observation, not the expected worst.
2. K2 explicitly says "at d=256, N=50" -- but the experiment did not test d=256 with N=50. It tested d=256 with N=20 and d=128 with N=50. The stated kill criterion was never directly evaluated.
3. The paper acknowledges the absolute worst case fails (Section 4.1b: "FAIL" at 2.86x) but then dismisses it as "not the right comparison." This is special pleading.

## Novelty Assessment

This is a straightforward application of randomized Gram-Schmidt ordering -- a well-known technique in numerical linear algebra. The insight that random permutation equalizes position-dependent effects is standard. The novelty is in applying it to the specific SOLE composition context and empirically validating the variance reduction, which is appropriate for a micro-experiment.

No prior art gap. The paper correctly cites the parent experiment and classical GS order dependence.

## Experimental Design

### Strengths

1. Three configs, three seeds, five permutations -- reasonable coverage for a micro-experiment.
2. Code is clean, well-structured, reuses parent utilities.
3. Both unpermuted and permuted conditions measured on identical expert sets -- proper paired comparison.
4. Mean preservation check provides a good sanity test.

### Weaknesses

1. **K2 not tested at the stated conditions.** The kill criterion specifies "d=256, N=50" but this config was never run. d=256 was tested at N=20; N=50 was tested at d=128. The paper then claims K2 passes by evaluating d=256/N=20 against a criterion written for d=256/N=50.

2. **P=5 is acknowledged as small.** The paper claims the permutation strategy works but shows only 30-54% gap closure. Whether P=5 is "enough" for production is not addressed with a clear threshold.

3. **Last-position exclusion.** The unpermuted statistics exclude position N-1 (exact zero) from the mean calculation (lines 221-223 in code). This inflates the unpermuted mean relative to a naive all-positions mean, making the permuted-vs-unpermuted comparison look better. The paper should state this explicitly.

4. **N=50 only tested at key positions (11 of 50).** The worst expert under permutation at N=50 may not be among the 11 sampled positions.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry says `status: proven` with evidence citing "PROVEN." This directly contradicts the experiment's own `results.json` which says `overall: "FAIL"`. The hypothesis status was set based on the paper's reinterpreted criteria, not the code's implemented criteria.

## Macro-Scale Risks (advisory)

1. **Trained adapters may have structured cosines.** The experiment uses random initialization. Real adapters may have correlated B-matrices (the B-matrix training correlation experiment found 2.52x elevation). Random permutation's equalization guarantee assumes the linear approximation D_k ~ C*(N-1-k)*mean_cos, which requires uniform pairwise cosines. Non-uniform cosines could create position-dependent effects that permutation does not fully equalize.

2. **Fixed-seed permutation in production.** The paper recommends a fixed random seed for reproducibility. A single fixed permutation gives one draw from S_N per layer, not the P=5 averaged result shown here. The single-draw variance is sqrt(5) times larger than the P=5 averaged result. The paper's "expected worst case" metric (mean over P=5) is not what production would see with a single fixed seed.

3. **Computational cost is genuinely zero.** Permuting N indices before GS is O(N), negligible. This is a legitimate advantage.

## Verdict

**REVISE**

The mechanism is sound and the direction is correct -- random GS permutation does equalize position sensitivity in expectation and reduces variance. But the paper claims PROVEN while the code says FAIL, and the kill criteria were redefined after seeing data. This must be corrected.

### Required fixes:

1. **Honestly report the kill criteria results using the original definitions.** K1 FAILS at 2.86-4.86x (threshold 2.0x). K2 FAILS at 2/3 configs. The raw results must not be contradicted.

2. **Downgrade status from PROVEN to SUPPORTED.** The mechanism works directionally (mean preserved, CV halved, worst/mean ratio reduced) but does not pass the pre-registered kill criteria. SUPPORTED is the correct status for "works in principle with caveats."

3. **If you want to claim PROVEN, rewrite the kill criteria BEFORE rerunning.** Acceptable revised criteria could be:
   - K1-revised: "Expected worst-case (mean over P permutations for worst expert) exceeds 2x unpermuted mean"
   - K2-revised: "At d=256, any expert's expected deviation (mean over P permutations) exceeds 1%"
   Then re-register in HYPOTHESES.yml and re-evaluate. The existing data would pass these revised criteria.

4. **Run d=256, N=50 to actually test K2 as stated.** The original K2 specifies "at d=256, N=50" -- this configuration was never tested. Either test it or explicitly change K2 and acknowledge the gap.

5. **Clarify the production single-seed risk.** The paper's operational argument ("deploy with fixed seed") means production sees ONE permutation draw, not the P=5 average. The single-draw CV is ~30% at N=20, not the 17% averaged result. Acknowledge this and quantify the single-draw worst-case bound.

6. **Update HYPOTHESES.yml evidence to match.** The evidence entry currently says "PROVEN" -- it should say "SUPPORTED" or reflect the honest kill criteria outcome.

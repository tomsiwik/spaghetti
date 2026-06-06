# Peer Review: BitNet Multi-Seed Validation (Re-review after REVISE)

## Prior Review Summary

The first review identified three issues requiring revision:
1. MATH.md described 1/N composition but code implements averaged-factor (1/N^2 effective)
2. Seed-42 PPL discrepancy with prior experiment unexplained
3. Population std used instead of sample std

## Verification of Applied Fixes

### Fix 1: MATH.md rewritten for averaged-factor composition -- APPLIED CORRECTLY

MATH.md now accurately describes the averaged-factor method (lines 17-37). The 1/N^2 effective scaling on diagonal terms is derived step-by-step. The cross-term structure is explicitly shown. The contrast with "textbook 1/N" composition is clear. The explanation of why the composition ratio is ~3.4x follows logically from the 1/N^2 dilution. PAPER.md (lines 80-99) mirrors this with consistent language ("averaged-factor composition," "1/N^2 effective scaling"). Both documents now accurately describe what the code actually does.

### Fix 2: Seed-42 discrepancy clarified -- APPLIED CORRECTLY

PAPER.md lines 104-123 explain the 7.3% code-domain discrepancy via the domain-dependent seeding formula `seed * 1000 + hash(domain_name) % 10000` (line 526 of run_experiment.py, confirmed). The key insight -- that the composition ratio still matches within 0.2% despite per-domain weight differences -- is correctly framed as strengthening the reproducibility claim.

### Fix 3: Sample std (N-1) -- PARTIALLY APPLIED

The code (line 601) now correctly uses `/ (len(ratios) - 1)` for sample std. PAPER.md reports the corrected values (std 0.016, CV 0.5%). However:

**results.json was not regenerated.** It still contains:
- `"composition_ratio_std": 0.0127` (population std, should be ~0.0155)
- `"composition_ratio_cv_pct": 0.4` (should be ~0.5)

**HYPOTHESES.yml evidence entry** (line 2769) still says "CV(composition_ratio) = 0.4%" and "0.013" std.

**FINDINGS.md** still says "CV(composition ratio) = 0.4% across 3 seeds" and "3.440 +/- 0.013".

These are stale artifacts from before the fix. The PAPER.md and code are now correct and mutually consistent, but three other files contain the old values. This is a bookkeeping issue, not a scientific one -- the verdict is unaffected (0.4% vs 0.5% against a 50% threshold).

## Mathematical Soundness

### Derivation: Correct

The expansion of the averaged-factor product (MATH.md line 29) is algebraically correct:

```
(1/N * sum B_i)(1/N * sum A_i) = 1/N^2 [sum_i B_i A_i + sum_{i!=j} B_i A_j]
```

The claim that cross-terms are small because |cos| ~ 0.002 is 40x below sqrt(r/d) = 0.079 is geometrically sound. The cosine similarity of vectorized deltas being small implies the column/row spaces of different adapters have minimal overlap, which bounds the cross-term norms.

### Statistics: Correct (in PAPER.md)

Sample std with N-1=2 divisor is the right choice for estimating population variance from 3 observations. CV = 0.016/3.440 = 0.465%, reported as 0.5% -- consistent.

### Minor concern: composition ratio definition

The ratio uses min_individual_ppl as denominator rather than mean. This is a conservative choice (makes the ratio look worse). Both MATH.md and PAPER.md are consistent on this definition, and the prior experiment uses the same definition, so comparisons are valid.

### Assumption 5 (STE gradient noise does not cause divergent trajectories)

This assumption is now empirically validated by the experiment itself -- that is the point. The CV of 0.5% demonstrates STE does not cause divergent trajectories at this scale/step count.

## Novelty Assessment

This is a validation experiment, not a novelty claim. It correctly positions itself as resolving the single-seed limitation of exp_bitnet_ternary_convergence. No novelty concerns.

The implicit finding that ternary STE training produces near-deterministic composition geometry (CV 0.5%) is interesting. The PAPER.md offers a plausible mechanistic explanation (ternary quantization constrains the solution space). The absence of an FP16 control means this explanation cannot be confirmed -- FP16 might also show similar stability. This is noted in the non-blocking recommendations below but does not affect the verdict.

## Experimental Design

### Strengths

1. **Kill criteria are pre-registered, sensible, and tested with huge margins.** CV 0.5% vs 50% threshold (100x margin). Max ratio 3.45x vs 10x threshold (2.9x margin).
2. **Self-consistency across the codebase.** The same `compose_adapters` function is used in bitnet_2b_real_composition and bitnet_ternary_convergence, so all results are comparable. MATH.md and PAPER.md now correctly describe this shared method.
3. **Training convergence patterns are seed-independent.** 3/5 domains converge by train loss in all 3 seeds; 5/5 improve val PPL in all 3 seeds. The non-convergence of code and creative is correctly attributed to batch_size=1 noise, not a seed-dependent issue.
4. **Per-domain CVs are very low.** Individual PPL CV: 0.02-0.19%. Composed PPL CV: 0.36-0.61%. This rules out the possibility that the aggregate CV is low only due to cancellation of per-domain fluctuations.
5. **Limitations are honest and comprehensive** (lines 127-138).

### Weaknesses (non-blocking)

1. **3 seeds is the statistical minimum.** The confidence interval on CV from 3 observations is very wide. The paper acknowledges this (Limitation 1). Sufficient for the 100x margin but would not survive a tighter threshold.
2. **No FP16 multi-seed control.** Cannot attribute the low CV to ternary quantization specifically vs. general properties of the model/data. The paper does not overclaim here -- the interpretation section uses "likely because" rather than "proven because."
3. **Thin validation sets (~3,200 tokens/domain).** The measurement noise floor is low enough that the per-domain composed CVs (0.36-0.61%) are distinguishable from zero, but barely. With larger val sets, the true between-seed variance might be even smaller.

### What could explain the result more simply?

The low CV could be entirely due to the 1/N^2 dilution: when adapter effects are scaled to 1/25 of their individual strength, any seed-dependent variation in adapter weights is also scaled to 1/25 of its effect on PPL. This is a valid alternative explanation that does not require invoking ternary quantization constraints. However, the individual-adapter CV is also very low (0.02-0.19%), and those are not subject to 1/N^2 dilution. So the low variance is genuine at the adapter level, not just masked by underscaling.

## Hypothesis Graph Consistency

- **Status "proven"**: Appropriate. Both kill criteria pass with large margins.
- **Kill criteria in HYPOTHESES.yml** match those in MATH.md and PAPER.md.
- **Blocks exp_bitnet_scale_n15 and exp_bitnet_task_eval**: Correct. N=15 scaling inherits the averaged-factor composition, and HYPOTHESES.yml note for N=15 says "1/N scaling" which should be updated to "averaged-factor" for accuracy, but this is advisory.
- **Evidence entry** needs the CV corrected from 0.4% to 0.5% (bookkeeping).

## Macro-Scale Risks (advisory)

1. **1/N^2 scaling at N=15 means 1/225 effective adapter strength.** The adapters may become invisible to the output. The exp_bitnet_scale_n15 experiment should monitor whether composed PPL approaches base PPL (i.e., adapters have no effect).
2. **If composition is ever changed to true 1/N** (pre-computed deltas), the reproducibility must be re-validated. The cross-terms currently masked by 1/N^2 would be absent, but individual adapter contributions would be 5x stronger, potentially amplifying seed-dependent variation.
3. **The near-zero CV is encouraging but may not transfer** to longer training (1000+ steps), different model scales, or true ternary-native composition (LoTA-QAF).

## Verdict

**PROCEED**

The three revision requests have been substantively addressed:

1. MATH.md and PAPER.md now accurately describe the averaged-factor composition with 1/N^2 effective scaling. The math-code-paper chain is internally consistent.
2. The seed-42 discrepancy is explained with the specific seeding formula, and the explanation is verifiable in the code.
3. The code uses sample std (N-1). PAPER.md reflects the corrected CV of 0.5%.

**One bookkeeping task remains (non-blocking):** Update results.json, HYPOTHESES.yml evidence, and FINDINGS.md to reflect the corrected std (0.016) and CV (0.5%). These are stale values from before the code fix, and the experiment was not re-run. Since the PAPER.md and code are authoritative, and the difference is 0.4% vs 0.5% against a 50% threshold, this does not block proceeding.

### Non-blocking Recommendations

1. Re-run the experiment (or just the aggregation) to regenerate results.json with corrected statistics. Alternatively, manually patch the three stale files.
2. Update HYPOTHESES.yml exp_bitnet_scale_n15 notes to say "averaged-factor composition" instead of "1/N scaling" for accuracy.
3. Consider adding a note to exp_bitnet_scale_n15 kill criteria about monitoring composed PPL vs base PPL gap (to detect adapter invisibility from 1/N^2 dilution at large N).

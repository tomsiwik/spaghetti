# Peer Review: Mixing Ratio Significance

## NotebookLM Findings

Skipped -- this is a straightforward statistical follow-up to a parent simulation study. The math is elementary (Wilcoxon signed-rank, bootstrap percentile CI, Cohen's d). Deep review not warranted for this complexity level.

## Mathematical Soundness

### What holds

1. **The Wilcoxon signed-rank test is correctly chosen and applied.** The paired design (same seed generates both ratio=0.0 and ratio=0.2) controls for initialization variance, which is the dominant noise source. Wilcoxon is appropriate for n=20 with potentially non-normal bounded differences. The two-sided alternative is conservative (correct for a follow-up where the parent claimed a positive effect -- one-sided would be defensible but two-sided is more rigorous).

2. **The power analysis is sound.** With n=20, approximately 80% power to detect d=0.65, the parent's apparent effect size of d~0.64 is right at the detection threshold. Observing d=-0.057 (wrong sign) with p=0.571 is strong evidence against the parent's claim. This is a textbook well-powered replication failure.

3. **Cohen's d computation is correct.** `mean(D) / std(D, ddof=1)` is the standard paired-sample effect size formula. The value of -0.057 is unambiguously negligible.

4. **The bootstrap CI for optimal ratio is correctly implemented** in `run_significance.py`. Resampling seeds with replacement, computing mean quality per ratio, taking argmax over mixed ratios, and reporting 2.5th/97.5th percentiles is standard nonparametric bootstrap for argmax stability.

5. **The Holm-Bonferroni correction in `run_experiment.py` is a welcome addition.** Testing 20 ratios against baseline creates a multiple comparisons problem. Even without correction, no ratio is significant (all raw p > 0.26), so the correction is academic but methodologically correct.

### What does not hold or is questionable

1. **RNG state coupling across ratios within a seed.** In both `run_significance.py` and `run_experiment.py`, each seed's `run_seed()` uses a single `rng` object sequentially across all 21 mixing ratios. The RNG state after generating data for ratio=0.00 feeds into the data generation for ratio=0.05, and so on. This means the training data for ratio=0.20 depends on the exact sequence of RNG draws for all previous ratios. This is not a bug per se -- the paired design still controls for initialization -- but it means the "pairing" between ratio=0.0 and ratio=0.2 is mediated by 4 intermediate ratio conditions that consume RNG state. A cleaner design would use independent RNG streams per ratio (e.g., `rng_ratio = np.random.default_rng(seed * 100 + ratio_idx)`). The current design is acceptable but could introduce subtle correlations between adjacent ratios' quality values.

2. **The A matrix is not truly frozen across ratios.** In `train_lora()`, `A` is initialized from the same RNG stream, but because the RNG is consumed differently by each ratio's data generation, the A matrix for ratio=0.0 and ratio=0.2 are *different* within the same seed. This contradicts MATH.md Section 2.1 which states "A: LoRA input projection (frozen)." The A matrix is frozen *during training* (not updated by gradient descent), but it is not the *same* frozen A across ratios. This weakens the paired design because part of the quality difference is attributable to different random projections, not just different training data. This is the same issue as the parent experiment, so it does not invalidate the comparison with the parent's findings, but it increases noise and reduces statistical power.

## Novelty Assessment

This experiment has zero novelty claims and makes none. It is a statistical follow-up to test a specific claim from the parent experiment. This is the correct framing. The contribution is purely methodological: replacing 5-seed unpaired comparison with 20-seed paired Wilcoxon. The conclusion (parent's +11.2% was noise) is the expected outcome given the parent's adversarial review already flagged overlapping CIs.

No prior art search is needed -- this is a replication/power study, not a new mechanism.

## Experimental Design

### Does this test what it claims?

**Yes.** The experiment directly tests K1 (is ratio=0.2 significantly better than ratio=0.0?) and K2 (is the optimal ratio stable?). The answer to both is clearly no.

### Discrepancy between the two scripts and the PAPER

There are two implementation files with **different K2 criteria and conflicting verdicts**:

- `run_experiment.py` defines K2 as `std(optimal_alpha) <= 0.15` and finds std=0.139, verdict: **PASS**
- `run_significance.py` defines K2 as `bootstrap CI width > 0.15` and presumably finds width=0.35, verdict: **KILLED**

The saved `results.json` was produced by `run_experiment.py` (it contains `std_optimal_alpha: 0.139` and `verdict: "PASS"` for K2, and lacks bootstrap CI fields for optimal ratio). But PAPER.md reports K2 as KILLED with bootstrap CI width=0.35, which matches `run_significance.py`.

**This is a material inconsistency.** The PAPER claims "KILLED both criteria" but the actual saved results show K2 PASS. The discrepancy arises because:
- std=0.139 < 0.15 (PASS by run_experiment.py's criterion)
- bootstrap CI width=0.35 > 0.15 (KILLED by run_significance.py's criterion)

These measure different things. The std of per-seed optima measures point dispersion. The bootstrap CI of the population argmax measures estimation uncertainty of the mean optimal ratio. The bootstrap CI is the more appropriate statistic because it answers the operationally relevant question: "if I commit to a single ratio, how confident am I that it is actually optimal?" However, the PAPER should not claim KILLED on K2 while the saved results say PASS without explaining the discrepancy.

Additionally, `run_experiment.py` includes alpha=0.0 in the optimal ratio search (line 208: `best_alpha = max(ALPHA_SWEEP, ...)`), while `run_significance.py` restricts to mixed ratios `0 < r < 1` (line 226). Including alpha=0.0 means two seeds (14 and 16) pick "no mixing" as optimal, which inflates the instability measure.

### Controls

- **Adequate:** 20 seeds, paired design, multiple test ratios, both one-sided and two-sided tests, Holm-Bonferroni correction, effect size reporting, bootstrap CIs
- **Missing but not blocking:** Independent RNG streams per ratio would strengthen the paired design

### Could a simpler mechanism explain the result?

Not applicable -- this is a null result. The experiment correctly identifies that the parent's positive result was noise, which is the simplest explanation.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches well:
- K1: "paired test p-value > 0.05 for ratio=0.2 vs ratio=0.0" -- matches both scripts
- K2: "optimal ratio changes by >0.15 with 20 seeds" -- ambiguous. The yml says "changes by >0.15" which could mean range, std, or CI width. The two scripts interpret it differently.

The status is correctly `killed`. The evidence claim in HYPOTHESES.yml matches the PAPER's narrative (both killed). Given K1 is unambiguously killed (p=0.571), the K2 discrepancy does not change the overall verdict.

## Macro-Scale Risks (advisory)

1. **The null result at micro does not prove mixing is useless at macro.** The parent's adversarial review correctly noted the linear-task limitation. Nonlinear tasks with distribution mismatch between teacher and student might genuinely benefit from data mixing. The PAPER's Limitations section acknowledges this.

2. **The strong finding transfers well:** pure synthetic is ~55% worse than pure real, confirmed at both 5 and 20 seeds. This reinforces the priority of adding real data to the pilot-50 pipeline.

3. **The flat quality landscape at low mixing ratios (0-25%) is practically useful:** it means the exact mixing ratio does not matter, so the pipeline need not be tuned precisely. Any amount of real data supplementation is fine.

## Verdict

**PROCEED**

This is a clean, well-executed statistical follow-up that correctly identifies the parent's +11.2% mixing benefit as noise. The K1 result is unambiguous (p=0.571, d=-0.057). The conclusion that mixing ratio optimization is not worth pursuing is well-supported.

The experiment has one material issue that should be documented but does not change the verdict:

1. **Document the K2 discrepancy between the two scripts.** The saved `results.json` (from `run_experiment.py`) shows K2 PASS (std=0.139), while PAPER.md reports K2 KILLED (bootstrap CI width=0.35, from `run_significance.py`). Add a note to PAPER.md explaining which script produced which result and why bootstrap CI width is the preferred stability metric. Alternatively, regenerate `results.json` from `run_significance.py` so the saved data matches the paper's claims. This is a bookkeeping issue, not a scientific one -- K1 alone is sufficient to kill the hypothesis, and the flat quality landscape makes the K2 verdict moot regardless of which metric is used.

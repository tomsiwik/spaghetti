# Peer Review: DARE Sparsified Composition

## Experiment Type
Guided exploration (Type 2). The proven framework is DARE unbiasedness (Yu et al., 2311.03099). The unknown is the optimal drop rate p for ternary-valued LoRA adapters.

## Hack Detector
- Fix count: 1 (DARE sparsification only). No flag.
- Is MATH.md a proof or a description? **Mixed.** The unbiasedness proof (Section C, first theorem) is a genuine Theorem/Proof/QED. The "OOD Interference Reduction" (Section C, informal bound) is explicitly labeled informal and is a description dressed in equations. This is acceptable for Type 2 -- the proven framework is DARE unbiasedness; the OOD interference argument is the exploration hypothesis.
- Metric used as evidence: Domain count with >=5pp degradation, in-distribution ratio vs no-DARE. These are reasonable proxies for behavioral outcomes (OOD corruption, task retention).
- Kill criteria source: Derived from prior findings (#260, #263) and the unbiasedness property. Adequate.

## Self-Test Audit

1. **One-sentence impossibility property:** "Sparsifying the delta reduces the number of corrupted output dimensions on OOD inputs while the unbiased estimator preserves expected in-distribution effect." This is two properties conjoined. Acceptable for exploration but slightly imprecise -- the impossibility claim (unbiasedness) should be separated from the exploration hypothesis (OOD dimension reduction). Minor issue.

2. **Cited theorems:** DARE unbiasedness (Yu et al. 2311.03099 Section 3.1) and Bernoulli variance. Both real, both correctly applied. Conditions are met: entries are treated as independent (assumption acknowledged in E.1), delta_ij values are deterministic given fixed adapter weights, M_ij are i.i.d. Bernoulli. PASS.

3. **Predicted numbers:** MMLU <= 3pp degradation, GSM8K >= +8pp, in-dist >= 64%. These are specific and falsifiable. PASS.

4. **Falsification condition:** "In-distribution performance drops >50% at p=0.5 would indicate the independence assumption fails." This targets the proof assumption (independence of delta entries), not just the experiment. PASS.

5. **Hyperparameter count:** 1 (drop rate p). Correctly identified as the guided exploration unknown. PASS.

6. **Hack check:** "No. DARE is a single modification." Correct. PASS.

**Self-Test verdict:** PASS (all 6 items addressed).

## Mathematical Soundness

**Theorem 1 (Unbiasedness):** Correct. E[delta_ij * M_ij / (1-p)] = delta_ij. Trivial but valid.

**Theorem 2 (Variance):** Correct. Var = delta_ij^2 * p/(1-p). Derivation is standard.

**Corollary (variance explosion):** Correct. At p=0.9: Var/delta^2 = 9. At p=0.95: 19.

**Informal OOD bound (Section C, "OOD Interference Reduction"):** This is where the math gets loose. The argument claims:

> E[||Delta_W_DARE * x||_inf] ~ (1-p) * (1/(1-p)) * E_active[|delta_ij * x_j|] = E[|delta_ij * x_j|]

This manipulates the infinity norm incorrectly. The infinity norm is a MAX over dimensions, not a SUM. Sparsifying entries does not change the infinity norm in expectation because the max of a subset can be larger (due to rescaling) or smaller (due to zeroing). The argument conflates "fewer corrupted dimensions" (an L0 claim) with "preserved infinity norm" (an L-infinity claim). These are different statements.

However, the qualitative conclusion -- "DARE should reduce the NUMBER of corrupted features, not the amount of corruption per feature" -- is actually reasonable despite the sloppy infinity-norm argument. This is because DARE zeros out entries entirely, and rescaled surviving entries have the same expected value. The L0 of the perturbation vector Delta_W_DARE * x is reduced in expectation. The actual per-feature corruption magnitude has higher variance but unchanged mean.

**Verdict on math:** The proven parts (unbiasedness, variance) are correct. The informal OOD argument has a technical error in the infinity-norm manipulation but the qualitative conclusion it draws is defensible. Acceptable for Type 2 exploration.

## Prediction vs Measurement

PAPER.md contains the required table. Assessment:

| Prediction | Measured | Match |
|-----------|----------|-------|
| P1: MMLU degradation <= 3pp at p=0.9 | MMLU -8pp at best (p=0.5) | FAIL |
| P2: GSM8K >= +8pp at p=0.9 | GSM8K +6pp at p=0.5, +10pp at p=0.7 | PARTIAL |
| P3: Higher p degrades in-dist | In-dist stable 75-80% all p | FAIL (no degradation observed) |
| P4: No degenerate output | All p pass | PASS |

**Critical observation:** Predictions P1-P2 were stated for p=0.9 but the best results come from p=0.5. The MATH.md predictions at p=0.9 are falsified. However, the exploration successfully narrowed the unknown (optimal p is ~0.5 for ternary adapters, vs ~0.9 for FP16 in the original DARE paper). This is a legitimate Type 2 outcome.

**Statistical concern:** MMLU sub-domain evaluations use n=20 (20 questions per domain). At n=20, the 95% CI for a proportion at 0.40 is approximately +/-21pp. This means differences of 5-10pp between conditions are well within noise. The only reliable MMLU signal is MMLU math, which degrades 15-35pp -- large enough to be real even at n=20. Most other MMLU sub-domain comparisons (e.g., medical staying at 40%, code staying at 40%) are consistent with random variation, not evidence of stability.

Similarly, code gen uses n=10. The 95% CI at 80% correct is +/-25pp. The non-monotonic behavior (90%, 80%, 70%, 90% across drop rates) is indistinguishable from noise at this sample size. PAPER.md acknowledges this in Limitations.

**GSM8K (n=50):** More reliable. 95% CI at 44% is about +/-14pp. The +6pp improvement at p=0.5 is within noise relative to base (38%), but the +10pp at p=0.7 is borderline significant. The -14pp at p=0.95 (24% vs 38%) is likely real.

## NotebookLM Findings

Skipping NotebookLM step -- the experiment is straightforward enough that manual analysis is sufficient.

## Novelty Assessment

**DARE itself is not novel** -- it is directly from Yu et al. (2311.03099). The contribution here is applying it to ternary (BitNet) base models with LoRA adapters and discovering that p=0.5 is optimal (vs p=0.9 in the original paper). This is a minor but useful finding for the project's specific architecture.

**Prior art:** TIES-Merging (Yadav et al., 2306.01708) is cited but not compared experimentally. TIES uses magnitude-based pruning + sign resolution, which could be more principled than random dropping for low-rank deltas. This is a gap -- the exploration would be strengthened by comparing DARE vs TIES.

## Macro-Scale Risks (advisory)

1. **Scale s interaction:** The effective rescaling is s/(1-p). At s=20, p=0.5: effective=40. At macro scale, if s changes, the optimal p will shift. The relationship s/(1-p) < threshold should be tracked.

2. **Multi-adapter composition:** This experiment tests single-adapter (oracle top-1 routing). Real composition merges multiple adapters. DARE variance accumulates across adapters: Var_total = sum_i Var_i. At N=5 adapters, variance is 5x higher. The optimal p may need to decrease with adapter count.

3. **MMLU math degradation is architectural.** DARE does not address it. Any macro system needs a separate solution for knowledge preservation under composition.

## Verdict

**PROCEED**

Justification:

1. The experiment correctly identifies itself as Type 2 (guided exploration) and the proven framework (DARE unbiasedness) is sound.

2. The exploration successfully narrowed the unknown: optimal p for ternary adapters is ~0.5, significantly lower than the p=0.9 recommended for FP16 in the original paper. The physical intuition (s=20 rescaling makes high drop rates destructive) is reasonable.

3. Kill criteria K681-K683 all PASS at p=0.5. The criteria were derived from prior findings and the unbiasedness property.

4. The main weakness -- predictions P1 and P2 were falsified -- is acceptable for Type 2 exploration. The predictions were made for p=0.9 (based on the original DARE paper's recommendation), and discovering that ternary adapters need lower p IS the exploration finding.

5. MMLU math degradation (-25pp at p=0.5) is honestly reported as unsolved and attributed to the composition mechanism itself (Finding #263), not DARE.

Caveats to carry forward:

1. The statistical power is low (n=10-20 for most benchmarks). The "1/5 domains degraded" result at p=0.5 vs "2/5 domains degraded" without DARE is a difference of 1 domain (code gen recovering from 80% to 90%), which is within the n=10 confidence interval. This finding should be labeled provisional until replicated with larger n.

2. TIES-Merging was cited but not compared. A follow-up should test whether magnitude-based pruning outperforms random dropping for low-rank structure.

3. The finding status should be **supported** (not conclusive) because predictions P1-P2 were not confirmed and statistical power is insufficient to distinguish signal from noise on most benchmarks.

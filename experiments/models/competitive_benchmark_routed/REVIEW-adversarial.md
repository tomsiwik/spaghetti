# Peer Review: competitive_benchmark_routed

## Experiment Type
Verification (Type 1) -- retesting a killed experiment with a proven fix.

## Hack Detector
- Fix count: 2 (per-domain scales + oracle routing). Below flag threshold.
- Is MATH.md a proof or a description? **Description dressed in equations.** There is no Theorem/Proof/QED block. The "perturbation ratio rho" calculation in Section F is a back-of-envelope estimate, not a bound. No theorem states that rho < 0.007 implies no MMLU degradation. The entire MATH.md is a narrative argument with some numbers, not a formal proof.
- Metric used as evidence: MMLU accuracy (n=20) and GSM8K accuracy (n=50). Neither is proven to predict the behavioral outcome of "competitive composition."
- Kill criteria source: K1 is sensibly derived from the claim (no degradation vs base). K2 threshold (4/6 vs Gemma) is somewhat arbitrary but reasonable for a comparative benchmark.

## Self-Test Audit

1. **One-sentence impossibility property:** "Per-domain scale selection keeps knowledge domains at s<=4 where rho<0.007, ensuring the adapter is in the pure augmentation regime (no overwriting)." This is a mechanism description, not a mathematically proven impossibility. The experiment just disproved it -- legal MMLU degraded at s=4. **FLAG: the claimed impossibility was falsified.**

2. **Cited theorems:** Finding #217, #220, and "Weyl's inequality." Weyl's inequality gives eigenvalue perturbation bounds for Hermitian matrices. Its application here is not shown -- no derivation connects Weyl's inequality to the rho bound or to MMLU accuracy. **FLAG: Weyl's inequality cited but never applied with verifiable steps.**

3. **Predicted numbers:** Specific and falsifiable. GSM8K >= 48% (PASS), Legal MMLU >= 50% (FAIL at 45%), Finance >= 30% (PASS), all 6 >= base (FAIL on 2). The predictions were genuinely testable. Credit here.

4. **Falsification condition:** "If routed composition with per-domain scales STILL degrades ANY benchmark below base." This is exactly what happened. The falsification condition was met. The experiment correctly killed itself.

5. **Hyperparameter count:** 0 new (uses prior findings). Correct.

6. **Hack check:** Legitimate retest of killed experiment. Not a hack.

## Mathematical Soundness

**There is no proof to verify.** MATH.md contains:
- A root cause analysis (narrative)
- A perturbation ratio calculation (rho = (s/1) * 0.034/20, one line, no derivation of why 0.034 or what the units are)
- A prediction table with thresholds

None of this constitutes a Theorem/Proof/QED block. The experiment is labeled "Type 1 (verification)" but there is nothing to verify -- there is no theorem predicting that rho < 0.007 implies MMLU accuracy >= base. The rho bound is an intuition dressed in a formula.

**The core logical gap:** Finding #220 proved 0/5 domains degrade *on generation quality (PPL)*. MATH.md extrapolated this to MMLU accuracy without proving the connection. PPL and MMLU accuracy measure different things. A model can have identical PPL but produce verbose answers that fail MMLU extraction, or produce terse answers that pass MMLU but have worse PPL. The experiment discovered exactly this gap.

## Prediction vs Measurement

The prediction-vs-measurement table is present and honest. Credit for clean reporting.

| Prediction | Expected | Measured | Verdict |
|------------|----------|----------|---------|
| P1: all 6 >= base | delta >= 0 | 2 worse | **FAIL** |
| P2: beat Gemma >= 5/6 | >= 5 | 3 | **FAIL** |
| P3: GSM8K >= 48% | >= 0.48 | 0.48 | PASS |
| P4: Legal MMLU >= 50% | >= 0.50 | 0.45 | **FAIL** |
| P5: Finance MMLU >= 30% | >= 0.30 | 0.35 | PASS |

3/5 predictions failed. The experiment correctly killed itself.

## Critical Issues

### 1. The "format mismatch" diagnosis is plausible but unproven

PAPER.md claims the root cause is format mismatch: SFT adapters produce verbose instruction-following output while MMLU expects single-letter answers. The evidence offered:

- Math adapter helps GSM8K (+10pp, chain-of-thought format matches) but hurts MMLU math (-20pp, single-letter format conflicts)

This is a reasonable hypothesis but it is **not verified**. To confirm format mismatch:
- Examine the actual generated outputs for MMLU: do they contain verbose text instead of "A"/"B"/"C"/"D"?
- Compare answer extraction success rates between base and routed models
- Test with a format-specific prompt (e.g., "Output only the letter")

Without this evidence, the diagnosis could be wrong. Alternative explanations:
- The math adapter at s=20 genuinely destroys the base model's factual math knowledge (overwriting, not format mismatch)
- The legal adapter at s=4 introduces noise that happens to hurt the specific 20 questions sampled
- Answer extraction bugs that differentially affect adapted vs base outputs

### 2. n=20 MMLU is statistically meaningless for the claims made

MATH.md itself acknowledges "+/-22pp CI" at n=20. This is devastating:

- **Legal MMLU:** base=55% (11/20) vs routed=45% (9/20). Delta = -2 correct answers. The 95% CI for a binomial at p=0.55, n=20 is [0.32, 0.77]. The routed result of 0.45 is well within this interval. **This "degradation" is indistinguishable from noise.**
- **Math MMLU:** base=50% (10/20) vs routed=30% (6/20). Delta = -4 correct answers. The 95% CI for p=0.50, n=20 is [0.27, 0.73]. The routed result of 0.30 is barely at the edge but still within. **Not statistically significant at p<0.05.**
- **Medical MMLU:** base=40% (8/20) vs routed=40% (8/20). Identical, but 8/20 could be 4/20 or 12/20 next run.

The experiment acknowledges this limitation but then proceeds to make causal claims ("per-domain scales do NOT fix the MMLU degradation", "format mismatch is the disease") based on differences that cannot be distinguished from random sampling noise. The KILL verdict for K1 may be correct in expectation, but the evidence is too weak to draw the causal conclusions in the root cause analysis.

### 3. Gemma and Qwen comparison numbers are not trustworthy

- **Gemma MMLU math: 5% (1/20).** Gemma-2-2B published MMLU scores are substantially higher. Getting 1/20 correct on math MMLU suggests a severe answer extraction or prompting problem, not genuine model capability. PAPER.md acknowledges this ("same extraction issue flagged in the original experiment") but still uses the numbers for win/loss counts.
- **Qwen GSM8K: 36%.** Qwen2.5-3B-Instruct published GSM8K is 65-70%. Getting 36% means the evaluation harness is broken for Qwen. Again acknowledged but still used.
- **SOLE beating Gemma on math (30% vs 5%)** is not a win -- it is two broken measurements being compared.

The K2 pass ("worse on only 3/6 vs Gemma") is meaningless when Gemma's math score is clearly an extraction artifact. If Gemma's actual math performance is 30-40% (plausible), SOLE loses 4/6 and K2 also kills.

**The comparator evaluation is not fit for purpose.** Using model-specific prompt templates (lines 310-371) is correct, but the answer extraction code is shared across all models. If the extraction regex fails differently for different model output styles, the comparison is invalid. The fact that two published models score dramatically below their known benchmarks is proof that the extraction pipeline has systematic errors.

### 4. MATH.md predictions failed because the theoretical framework has a gap

The framework (Findings #217, #220) proved that per-domain scales prevent *generation quality* degradation (measured by PPL). MATH.md assumed this transfers to *factual recall accuracy* (MMLU). This assumption was never stated as an assumption -- it was treated as obvious. It was wrong.

This matters beyond this experiment: the entire scale-aware composition framework (Finding #221, "minimum viable architecture") was validated only on PPL. If PPL improvements do not transfer to downstream task accuracy, the practical value of the architecture is narrower than claimed.

### 5. Recommendations do not address the root cause

- **"Format-aware adapter training"**: This means training adapters on MMLU-format data. But MMLU is a benchmark, not a use case. Training adapters to produce single-letter answers is benchmark gaming, not a fix for the underlying mismatch between SFT training and evaluation format.
- **"Conditional routing (skip adapter for factual recall)"**: This admits the adapters provide no value for MMLU. An adapter that must be bypassed for half the benchmarks is not a "composable expert" -- it is a task-specific tool with known failure modes.
- **"The positive GSM8K signal is real"**: Agreed, but 1/6 benchmarks with consistent improvement is a narrow win. And GSM8K specifically rewards chain-of-thought, which is exactly what SFT teaches. This may not generalize to other reasoning benchmarks.

## NotebookLM Findings

Skipped -- the experiment is already killed and the key issues are clear from document review.

## Novelty Assessment

This is a retest, not a novel contribution. The value is in the negative result: per-domain scales fix PPL but not MMLU accuracy. This is a useful finding that should be recorded.

## Macro-Scale Risks (advisory)

1. The gap between PPL improvement and benchmark accuracy improvement is a fundamental concern for the architecture. At macro scale, if adapters improve PPL but degrade or do not affect downstream tasks, the entire composition framework loses practical value.
2. The answer extraction problem will compound at scale. Any competitive benchmarking effort needs a validated evaluation harness first.
3. The format mismatch diagnosis, if correct, implies that the adapter training procedure (SFT on instruction data) is fundamentally mismatched with standard benchmarks. This is not fixable by routing or scale tuning -- it requires rethinking what the adapters are trained to do.

## Verdict

**KILL** (agreed with experiment's own assessment)

The experiment correctly killed itself. The KILL verdict is sound even though the evidence is weaker than claimed. Specific issues to record:

1. **MATH.md lacks a formal proof.** The experiment was labeled Type 1 (verification) but contains no theorem to verify. The rho bound is an intuition, not a derivation. Future retests must include a Theorem/Proof/QED block or be labeled Type 2 (guided exploration).

2. **n=20 MMLU is insufficient to support causal claims.** The legal -10pp gap is noise. The math -20pp gap is borderline. The root cause analysis should be recorded as a hypothesis, not a conclusion. Minimum n=100 per domain for any future MMLU-based kill criteria.

3. **The comparator evaluation is broken.** Gemma math at 5% and Qwen GSM8K at 36% are extraction artifacts. Any future competitive benchmark must first validate the extraction pipeline against published numbers (within 5pp) or the comparison is invalid.

4. **The PPL-to-accuracy gap is the real finding.** Finding #220 (0/5 domains degrade on PPL) does not transfer to Finding-this (2/6 degrade on MMLU). This gap should be recorded as a finding in its own right -- it constrains the claims that can be made about scale-aware composition.

5. **The format mismatch hypothesis needs direct evidence.** Inspect generated MMLU outputs. Count how many produce verbose text vs single letters. Compare extraction failure rates between base and adapted models. Without this, the diagnosis remains speculative.

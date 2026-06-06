# MATH.md: Metric-Behavioral Dissociation Under Format-Sensitive Evaluation

## Type: Guided Exploration (narrowing behavioral quality under routed composition)

## A. Failure Mode Identification

**The disease:** Standard benchmarks (MMLU) measure *format compliance* — whether a model outputs a single letter A/B/C/D — not *domain knowledge*. SFT adapters teach instruction-following format (`### Instruction: ... ### Response:`) which conflicts with MMLU's expected single-letter output. This creates a systematic confound: adapters that improve actual knowledge degrade benchmark scores because they produce verbose, explanatory responses instead of single letters.

**Why this is a real risk:** Three consecutive competitive benchmark experiments (exp_competitive_benchmark, exp_competitive_benchmark_retest, exp_competitive_benchmark_routed) were all KILLED because routed composition degraded MMLU scores relative to base. Meanwhile, GSM8K (which accepts free-form numerical answers) showed consistent +10pp improvement. This is not noise — it is a systematic format confound.

**Prior evidence establishing the failure mode:**
- Finding #236: Per-domain scales fix PPL but NOT MMLU accuracy (PPL-accuracy gap)
- Finding #237: GSM8K +10pp is the only consistent competitive advantage
- Competitive benchmark routed: MMLU dropped from 44% to 38% (math), while GSM8K improved from 38% to 48%

## B. The Right Question

**Wrong:** "How do we make routed composition improve MMLU scores?"
**Right:** "Does routed composition improve the actual behavioral quality of generated text, independent of format-sensitive benchmarks?"

If yes, then the architecture works and MMLU was measuring the wrong thing.
If no, then the adapters genuinely degrade knowledge, and the project has an existential problem.

## C. Prior Mathematical Foundations

**Observation (Format-Metric Confound):** Let M(x) be a format-sensitive metric (MMLU) and B(x) be a behavioral metric (execution-based). For model f and adapter perturbation P:

M(f + P) < M(f) does NOT imply B(f + P) < B(f)

when P shifts the response distribution toward a format incompatible with M's scoring function but compatible with B's.

This is not a deep theorem — it is a straightforward observation about measurement validity. The "proof" here is empirical: we have prior evidence (Finding #236, #237) establishing the dissociation, and this experiment tests whether it extends to execution-based behavioral metrics across all 5 domains.

**Grounding:** The distinction between format compliance and knowledge is well-established in the evaluation literature. Evaluation contamination and prompt sensitivity are documented in:
- Liang et al., "Holistic Evaluation of Language Models" (HELM, 2022) — benchmark sensitivity to prompt format
- The SIGReg analysis in this project's Finding #236 — formal identification of the PPL-accuracy gap

## D. Predictions

The following predictions are derived from the format-confound hypothesis:

| # | Prediction | Basis | Kill if |
|---|-----------|-------|---------|
| P1 | Routed composition improves math behavioral score (answer correctness) vs base | GSM8K +10pp (Finding #237) | math behavioral < base |
| P2 | Routed composition improves code behavioral score (syntax validity) vs base | Code adapter is trained on code SFT data | code behavioral < base |
| P3 | Routed composition neutral-to-positive on prose domains (medical, legal, finance) | Domain-matched adapters with calibrated scales | All 3 prose domains worse |
| P4 | Behavioral improvement on >= 1 domain where MMLU degraded (i.e., >= 1 behavioral-MMLU contradiction) | Format confound predicts dissociation | Zero contradictions (K2 KILL) |
| P5 | Routed composition >= base on behavioral for >= 3/5 domains | Oracle routing selects best adapter | K1: < 3/5 domains better |

**Quantitative bounds (from prior data):**
- Math: base behavioral = 0.10 (1/10 correct from prior experiment with code adapter). With math adapter + scale 20, expect >= 0.20 (at least doubling, consistent with GSM8K +10pp).
- Code: base behavioral = 0.42. With code adapter + scale 20, expect >= 0.50 (consistent with prior +0.15 improvement).
- Prose: base behavioral ~ 0.10-0.26. With domain-matched adapter, expect >= base (neutral).

## E. Assumptions & Breaking Conditions

1. **Oracle routing assumption:** We use oracle top-1 routing (best adapter = domain-matched adapter). If the adapters do not specialize by domain, routing gains nothing. Breaking: all adapters produce identical output regardless of domain.

2. **Per-domain scale calibration:** We use Finding #217 scales {medical:20, code:20, math:20, legal:4, finance:1}. If these scales are wrong for behavioral metrics (they were optimized for PPL), behavioral quality could degrade. Breaking: legal/finance adapters at low scale produce worse text than base.

3. **Evaluation function validity:** Finding #210 validated the behavioral framework with Cohen's kappa = 0.800. If the framework is unreliable, results are meaningless. Breaking: kappa < 0.7 on new data.

4. **SFT format compatibility:** The behavioral eval prompts use `### Instruction:` format which matches SFT training format. This is intentional — we test whether the adapters improve response quality in their native format, not in MMLU's format.

## F. Worked Example (not applicable — no novel math)

This experiment is a measurement, not a mathematical derivation. The "proof" is that format-sensitive metrics can dissociate from behavioral quality, which is self-evident once stated. The novel contribution is the empirical measurement across 5 domains with execution-based metrics.

## G. Complexity & Architecture Connection

**Runtime:** 5 domains x 10 prompts x 2 conditions (base, routed) x 128 tokens/generation
= 100 generations at ~1.2s each = ~2 min for generation + model load overhead.
Estimated total: 15-25 min (dominated by model load/unload between adapter swaps).

**Memory:** Peak ~5-15 GB. Base model ~5 GB, adapter merge adds negligible overhead.
Well within M5 Pro 48 GB budget.

## Self-Test

1. **One property:** Format-sensitive metrics (MMLU) conflate format compliance with knowledge, so behavioral (execution-based) metrics are needed to measure actual quality.

2. **Existing theorems:** Not a mathematical theorem — this is measurement methodology. Grounded in HELM evaluation framework (Liang et al., 2022) and Finding #236 (PPL-accuracy gap).

3. **Specific predictions:** Math behavioral >= 0.20 (vs base 0.10). Code behavioral >= 0.50 (vs base 0.42). Routed >= base on >= 3/5 domains.

4. **Falsification:** The hypothesis is wrong if routed composition degrades behavioral quality on >= 3/5 domains (K1 KILL). The format-confound theory is wrong if behavioral and MMLU agree in direction on all MMLU-degraded domains — i.e., zero behavioral-MMLU contradictions (K2 KILL).

5. **Hyperparameters added:** 0 new. Per-domain scales are from Finding #217. Oracle routing is deterministic.

6. **Hack check:** No. This is a measurement experiment, not a fix. We are measuring whether the architecture already works.

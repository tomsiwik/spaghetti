# Adversarial Review: exp_p10_mcq_adapter_training

## Verdict: PROCEED (KILLED)

KILLED status is appropriate. Data is clean, findings are genuine and actionable.

## Prediction-vs-Measurement Audit

PAPER.md table is comprehensive (8 predictions). Results:
- 2 PASS (base no-thinking 41.7%, training time 10.8min)
- 1 EXCEED (base+thinking 62.1% vs predicted 45-55%)
- 5 FAIL (adapter+thinking, MCQ effect, thinking effect, HumanEval, combined)

Kill criteria match results.json exactly:
- K1470: 50.4% = 141/280 ✓
- K1471: 25.0% = 5/20 ✓
- K1472: 10.8min = 650.1s ✓

## Three Findings Assessment

**1. Thinking works on MMLU-Pro (+20.4pp)** — STRONG. The 757,251 thinking chars vs 0 chars
in non-thinking mode is decisive evidence that thinking chains are generated and useful.
The +20.4pp effect (41.7→62.1%) is large and consistent across categories (math +63pp,
business +50pp). This genuinely contradicts the GPQA finding (#528) and the depth-dependent
quantization ceiling explanation is plausible. Sample size: 280 questions (20/category) is
modest but the effect size overwhelms noise.

**2. Adapter suppresses thinking (0 chars)** — DECISIVE. 0 thinking chars vs 757,251 is
not a statistical claim — it's a binary observation. The structural explanation (LoRA trains
direct-answer pathway, blocks thinking channel entry) is mechanistically sound.

**3. MCQ adapter degrades generative quality** — SUPPORTED but caveat: N=20 HumanEval is
small. 5/20 = 25% has wide confidence interval (~10-47% at 95% CI). Still, the direction
is clear (-35pp is too large for sampling noise alone) and mechanistically expected.

## Issues (Non-Blocking)

**1. Theorem 2 prediction wrong, not addressed.** Theorem 2 claimed "standard LoRA + MCQ
loss should exceed Finding #522's TT-LoRA + MCQ result (+14.5pp on MedMCQA)." Got +5.4pp,
less than half of TT-LoRA's +14.5pp. PAPER doesn't explain why standard LoRA underperformed
TT-LoRA on MCQ despite the theorem predicting the opposite. Possible confound: different
benchmarks (MMLU-Pro vs MedMCQA), different base models, different training data sizes.
Not blocking because the main finding is about thinking mode, not MCQ effect size.

**2. Engineering category anomaly.** Base+thinking = 25% (WORSE than base 42% by -17pp).
This is the only category where thinking HURTS. Small N=20 likely explains this, but it
weakens the "thinking uniformly helps on MMLU-Pro" claim. PAPER doesn't flag this.

**3. HumanEval baseline is estimated.** PAPER says "~60% base" but results.json only has
the adapted result (25%). No measured base HumanEval. The ~60% appears to come from prior
experiments. Not blocking but the -35pp claim is against an estimated baseline.

## Status Appropriateness

KILLED is correct. Two catastrophic kill criteria failures (adapter suppresses thinking,
destroys generative quality) make this a clean kill. The three emergent findings are
valuable — especially the thinking mode result which changes the architecture roadmap.

## Recommendation

Record as KILLED finding. The thinking mode +20.4pp result is the headline — it establishes
base+thinking (62.1%) as the MMLU-Pro strategy and closes the thinking-under-quantization
question for shallow reasoning benchmarks.

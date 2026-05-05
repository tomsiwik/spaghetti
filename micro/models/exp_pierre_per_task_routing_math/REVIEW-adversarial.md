# Adversarial Review — exp_pierre_per_task_routing_math

## Verdict: KILLED (agree)

Kill is correct. The premise — that DARE causes a math-specific regression — was falsified. Single adapter = DARE = 63.3% on GSM8K. No routing can fix a non-existent regression.

## Strengths

1. **Clean falsification.** Comparing single-adapter vs DARE on the same 30 samples directly tests the hypothesis. The experiment disproved itself efficiently.
2. **Routing validated.** TF-IDF+Ridge at 100% accuracy and 0.072ms is a reusable primitive for future per-task routing designs.
3. **Correct methodology.** Kill criteria were pre-registered, evaluation was automated, and the code is straightforward.

## Weaknesses

1. **N=30 is thin.** The 23.3pp HumanEval variance between identical DARE runs confirms that N=30 is unreliable for generation benchmarks. Future experiments should use N≥100 or report confidence intervals.
2. **Stale reference.** The 70% GSM8K baseline was never verified before designing the experiment. A 5-minute sanity check would have killed this before any code was written.
3. **No temperature/seed control noted.** The paper mentions "stochastic generation" but doesn't report whether temperature was fixed or seeds varied. This should be standard.

## Reusable Findings

- TF-IDF+Ridge binary routing: validated, sub-ms, perfect on math-vs-not. Worth extracting as a utility.
- GSM8K 63.3% is the true single-adapter baseline (on this sample size). Update any references citing 70%.
- HumanEval N=30 has ±23pp noise floor — never trust a single run at this N.

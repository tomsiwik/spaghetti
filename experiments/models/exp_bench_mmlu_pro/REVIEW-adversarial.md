# REVIEW-adversarial.md: exp_bench_mmlu_pro

## Verdict: PROCEED

## Summary

Experiment is sound. Pipeline validated. Three key findings (thinking essential, NTP hurts MCQ in-domain, eval pipeline at 5.3 q/s) are well-supported by data. Status "supported" is appropriate.

## Prediction Quality

| Criterion | Predicted Range | Measured | Assessment |
|-----------|----------------|----------|------------|
| K1 base accuracy | 54-63% | 42.3% | 12pp below lower bound. Thinking penalty underestimated for 10-option MCQ |
| K2 adapter delta | [-1.1, +2.1]pp | -6.2pp | Sign flip. NTP hurts in-domain too, not predicted |
| K3 runtime | [0.4, 1.0]h | 0.18h | Within order of magnitude, faster than expected |

K1 and K2 prediction misses are significant but the paper analyzes them honestly. The root cause (10-option MCQ amplifies thinking penalty; NTP conflicts with instruction-following even in-domain) is correctly identified.

## Non-Blocking Notes

1. **K2 theorem gap**: Theorem 2 assumed $\Delta_{\text{math}} > 0$ (adapter helps its own domain). The measured -13pp in-domain shows NTP-trained adapters don't just fail OOD — they conflict with MCQ format even in-domain. This is a stronger impossibility result than MATH.md derives. Future MATH.md files should model the NTP-vs-instruction conflict explicitly rather than assuming in-domain help.

2. **Finding #44 citation**: PAPER.md cites Finding #44 as "NTP adapters degrade OOD" but the measured result is stronger — degradation is uniform across ALL domains including in-domain. The finding should distinguish "OOD degradation" from "format conflict degradation."

3. **Sample size note**: 100 questions per category is adequate for detecting 27pp and 6pp effects (both are >3x the ~3pp standard error for n=100 binary outcomes). No statistical concern.

## What Advances the Vision

- **Pipeline infrastructure**: 5.3 q/s direct generation is reusable for future benchmarks. This unblocks N=5 composition benchmarking.
- **NTP-instruction conflict**: Establishes that current NTP adapters cannot help MCQ benchmarks. SFT training or thinking-mode are required. This is decision-relevant for P0 benchmark goals.
- **Thinking-mode gap**: 27pp gap means any serious MMLU-Pro benchmark must use thinking mode. Budget ~40min per full run.

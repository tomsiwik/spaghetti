# LEARNINGS.md: exp_bench_mmlu_pro

## Core Finding

NTP-trained adapters actively degrade MCQ benchmarks (-6.2pp uniform across all 14 domains,
including math's own domain at -13pp). Gemma 4 E4B without thinking scores 42.3% MMLU-Pro
vs 69.4% reported (thinking enabled) — a 27pp gap.

## Why

NTP loss optimizes $\min_\theta -\sum \log p(x_t | x_{<t})$ on domain text, which shifts
attention distributions toward language modeling and away from instruction-following required
for MCQ ("Answer with ONLY the letter"). The conflict is format-level, not domain-level —
it applies even in-domain. The 10-option MMLU-Pro format amplifies the thinking penalty
relative to 4-option MMLU because multi-step elimination is required.

## Implications for Next Experiment

1. **SFT adapters required for benchmarks**: Any adapter intended to help MCQ must be
   trained on instruction-tuned question-answer pairs, not NTP on domain text.
2. **Thinking mode mandatory for MMLU-Pro**: Budget ~40min for a thinking-enabled run
   (~65-70% expected). Non-thinking results are not comparable to Google's numbers.
3. **Pipeline is production-ready**: Direct mlx_lm.generate() at 5.3 q/s, 1400 questions
   in <11min. Reuse for N=5 composition benchmark without modification.
4. **NTP-instruction conflict is stronger than Finding #44**: Prior finding said "OOD
   degradation"; this shows uniform format conflict even in-domain. Update framing.

# LEARNINGS: exp_bench_mmlu_pro_thinking

## Core Finding
Base Gemma 4 E4B 4-bit with thinking achieves 62.1% on MMLU-Pro (+20.4pp over non-thinking),
but MCQ adapter training without `enable_thinking=True` completely suppresses thinking chains
(0 chars generated), dropping performance to 50.4% — worse than base without thinking.

## Why
Thinking mode operates through a separate generation channel that must be activated during
training; fine-tuning on MCQ data without `enable_thinking=True` creates a mode mismatch
where the adapter suppresses the thinking channel entirely (Finding #530). The 7.3pp gap
vs. Google's 69.4% target is explained by 4-bit quantization degrading precision on
multi-step computation (quantization degrades numerical precision, hurting calculation-heavy
steps that thinking chains rely on). Engineering anomaly (-17pp with thinking) is consistent
with text-only reasoning being insufficient for spatial/diagram-based questions.

## Implications for Next Experiment
Any adapter trained for use with thinking mode MUST pass `enable_thinking=True` during
fine-tuning. Budget forcing (exp_p10_budget_forcing) is the natural next step to reduce
the 135x token overhead while maintaining the 62.1% accuracy; exp_p10_budget_forcing should
test whether constraining thinking tokens to ~10x still preserves the math (+63pp) and
business (+50pp) gains that drive the overall improvement.

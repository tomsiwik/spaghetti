# LEARNINGS: exp_p10_budget_forcing

## Core Finding
Fixed token budgets for thinking are **binary, not gradual** — budgets below ~1024 tokens produce
10-12% accuracy (worse than no-thinking at 41.7%), while full thinking at B=2048 reaches 46-62%.

## Why
Truncated thinking is **actively harmful**: the model burns tokens on reasoning format/preamble,
never completes a coherent chain, then confidently asserts wrong answers. The Gamma CDF model
(arXiv:2506.13752) assumed truncation → base accuracy; the reality is truncation → below-base accuracy.
This is a 4-bit quantization effect — shorter coherent chains cannot be maintained at reduced precision.

## Implications for Next Experiment
Budget forcing research is **closed** for 4-bit models. The only viable thinking strategy is
full thinking or no thinking. If thinking token reduction is needed, it must come from
**training-time** approaches (e.g., training with thinking enabled + response-length reward)
rather than inference-time truncation. The 7.3pp gap vs Google's 69.4% is quantization, not
capability — the model simply cannot compress its reasoning into fewer tokens without collapse.

# LEARNINGS: exp_p1_t0_vnorm_scale_safety

**Status:** KILLED (structural barrier)
**Finding:** #410

## Core Finding

V-norm post-hoc injection on Qwen3-4B is structurally unsafe: o_proj was trained on
un-normalized values, so injecting v_norm causes distribution shift WORSE than the
scale catastrophe it was meant to fix (-36pp vs -32pp without v_norm at scale=5-10).

## Why

Theorem 1 is mathematically correct: v_norm forces ||V||_RMS = sqrt(h) for all scales.
But this guarantee is only useful when o_proj was trained to EXPECT normalized values.
Gemma 4 has v_norm integral to training — so the theorem is testable there but not on Qwen3.
The full guarantee: "safe injection requires co-trained o_proj."

Side result: K995 PASS replicates Finding #320 — scale=20 without v_norm degrades MMLU
by 32pp (>30pp threshold). Scale catastrophe is confirmed on Qwen3-4B.

## Implications for Next Experiment

Resurrection requires adding `gemma4` model type to mlx_lm (mlx_lm 0.29.1 maps
model_type to class via MODEL_REMAPPING; gemma4 has nested text_config → gemma3n).
Once Gemma 4 E4B loads, re-run T0.2 — K994 should PASS by construction (v_norm
trained into base weights). Researcher should check if mlx_lm has added gemma4 support
before claiming a new T0.2 experiment.

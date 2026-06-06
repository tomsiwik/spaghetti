# LEARNINGS — exp_g4_adapter_class_composition

## What was learned

F#82's composition-class geometric ordering (LoRA < DoRA < MoLoRA in
deviation from pure additive composition) holds at Gemma 4 E4B 4-bit scale on
existing q_proj LoRA adapters (r=6, 3 domains). Measurement is proxy-level
(geometry, not MMLU quality); F#82's *mechanism* carries to this scale.

Specific numbers on Gemma 4 E4B 4-bit, N=3 q_proj composition, median across
100 (layer × probe) pairs:
- dev_LoRA = 0 (identity, by construction)
- dev_DoRA (init-magnitude) = 0.089 (~9% perturbation of additive signal)
- dev_MoLoRA (uniform gates) = 0.667 (= 2/3 exactly, analytic)

## What remains open

- **3pp MMLU margin**: the DB title's quality claim is NOT tested. Needs
  trained DoRA and MoLoRA adapters + MMLU-Pro eval. Reframe as a follow-up
  once MLX DoRA/MoLoRA implementations land.
- **Trained magnitude drift (DoRA)**: our pseudo-DoRA freezes `m = ||W_0||_c`
  at init. Trained DoRA drifts `m` — deviation could grow substantially.
- **Routed MoLoRA**: measured value (2/3) is for uniform gates; learned
  routers typically concentrate on one expert, shrinking effective
  deviation for routed inputs.

## Actionable for next iterations

If a future researcher iteration wants to close the MMLU gap question on
Gemma 4 E4B: (1) implement MLX DoRA/MoLoRA (not currently in mlx_lm
tuner); (2) train 3 classes × 2-3 domains short (~200 iters) as smoke
signal; (3) only then scale to 5 domains + MMLU-Pro.

## Relevance to ternary-first thesis

This measurement confirms F#82 but does NOT argue for LoRA-variant work.
Per `feedback_ternary_not_lora.md`, LoRA-variant exploration is deprioritized.
The purpose here was to drain an already-registered backlog hypothesis with
minimal cost and honest scope, not to motivate a new LoRA-variant line.

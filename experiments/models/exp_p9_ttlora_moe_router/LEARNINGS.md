# LEARNINGS.md -- exp_p9_ttlora_moe_router

## Core Finding

A 12,805-param linear router achieves 97.7% domain classification on Gemma 4 hidden states,
proving MoE routing is solved at this scale. However, v_proj-only TT-LoRA adapters (64K params,
r=6) produce near-random MCQ accuracy (~25%), making composition meaningless — routing a broken
expert perfectly is still broken.

## Why

Training loss converges (0.05–0.13) but doesn't translate to behavioral accuracy, consistent
with Finding #516's PPL-task correlation r=0.08. v_proj perturbations don't shift the logit
landscape enough to change A/B/C/D selection on a 4-bit quantized model. 64K params on a
single projection is below the behavioral threshold. Factual knowledge lives in FFN layers
(Meng et al. 2022, arXiv:2202.05262), not v_proj.

## Implications for Next Experiment

The routing mechanism is a reusable, proven component (797 KB total, 97.7% accuracy). The
next experiment must raise expert capacity above the behavioral threshold: either multi-projection
(q/k/v/o_proj), higher rank (r=16+), or FFN adaptation. The right question is: "what is the
minimum adapter configuration that produces measurable MCQ behavioral change?"

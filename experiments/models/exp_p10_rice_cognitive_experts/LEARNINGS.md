# LEARNINGS — exp_p10_rice_cognitive_experts

**Status:** KILLED (Finding #529)

## Core Finding
RICE cognitive layer identification (arXiv:2505.14681) is fundamentally incompatible
with dense 4-bit models. Zero layers on Gemma 4 E4B showed thinking nPMI > 0.3
(max = 0.104), confirming that 4-bit quantization uniformly destroys cognitive signal
across all 42 layers.

## Why
RICE requires discrete expert routing (binary activation) to compute meaningful nPMI.
Dense models have continuous activations, and 4-bit quantization noise (ε_q ≈ 0.06
per layer) compounds to exceed any cognitive signal. Thinking tokens are noise under
4-bit quantization — confirmed at activation level (2nd proof after Finding #528).

## Novel: Layer Scalar Architecture
Gemma 4's `layer_scalar` values range 0.061–0.887 with std=0.222. HIGH-norm layers
have LOW scalars (model suppresses them); LOW-norm layers have HIGH scalars. Pre-training
already learned layer importance weighting — external identification is redundant.
This inverse scalar-norm correlation may be exploitable for efficient inference (layer
skipping at low-scalar layers).

## Implications for Next Experiment
- exp_p10_reasoning_adapter premise (reinforce cognitive layers) is invalid — abandon
- RICE is MoE-only; any cognitive-layer work requires either full-precision inference
  or a model with discrete expert routing
- Layer scalars are an untapped signal for adapter placement: target high-scalar
  (trusted, low-norm) layers 32–37 for domain adaptation rather than uniform coverage

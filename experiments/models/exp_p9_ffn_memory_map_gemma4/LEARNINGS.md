# LEARNINGS: exp_p9_ffn_memory_map_gemma4

## Core Finding
Geva et al. (arXiv:2012.14913) FFN-as-key-value-memory framework does not transfer to Gemma 4 E4B: pattern specificity 23.7% (vs 50% threshold), domain clustering 1.66x (vs 2x), next-token agreement 0.10% (vs 1%). All three kill criteria fail — experiment KILLED.

## Why
GeGLU's soft gating produces diffuse activations (no hard zeros like ReLU), preventing sharp neuron specialization. 4-bit quantization adds noise that degrades value vector → vocabulary projection. Together, individual neurons in Gemma 4 E4B do not cleanly encode discrete domain knowledge.

## Impossibility Structure
Neuron-level interpretability requires: (1) sparse activation (ReLU), (2) full-precision weights, (3) large-scale probing data (100K+). GeGLU + 4-bit quantization violates conditions 1 and 2 simultaneously. Fine-grained neuron editing as an adapter mechanism is structurally incompatible with this model class.

## Implications for Next Experiment
- LoRA subspace is the correct abstraction: rank-8/16 projections aggregate across diffuse neurons, bypassing the sparsity requirement
- exp_p9_ffn_targeted_edit should be killed by dependency — neuron targeting is not viable
- Domain specialization in Gemma 4 is an early-layer phenomenon (layers 0-8), opposite to Geva et al.'s upper-layer finding — this may be worth probing for early-layer adapter placement

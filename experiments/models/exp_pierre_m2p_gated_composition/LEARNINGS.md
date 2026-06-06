# LEARNINGS — exp_pierre_m2p_gated_composition

## Finding #828

**Domain routing via learned softmax gate is solved.** A 2-layer MLP on mean-pooled embeddings achieves 99.6% holdout accuracy with entropy 0.039 nats — near-binary routing to the correct adapter.

## Why It Failed

The composition *application* path is broken, not the routing. Monkey-patching `__call__` on PoLARLinear modules destroys the forward pass — HumanEval drops from 86.7% to 20.0% even when the gate correctly selects the right adapter at w≈1.05. The patched forward's `layer.base(x)` doesn't match PoLARLinear's actual computation, and `__get__` binding on MLX modules is unreliable.

## Implication

Any composition experiment (including ties_dare) must inject weights into PoLAR A/B matrices directly, not override the forward pass. The gate architecture itself is reusable as-is.

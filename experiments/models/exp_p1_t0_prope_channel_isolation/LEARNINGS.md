# LEARNINGS — T0.3: p-RoPE Channel Isolation

**Finding #411 (supported) | 2026-04-09**

## Core Finding

NoPE dims [128:512] in Gemma4 global attention heads are algebraically position-invariant: inv_freq=0 → RoPE rotation = identity, verified numerically to max_diff=0.0 exactly. Domain adapters restricted to NoPE dims receive no positional contamination.

## Why

Gemma4's `_compute_proportional_rope_parameters` with partial_rotary_factor=0.25 sets inv_freq=0 for 75% of channels. When inv_freq=0, cos(θ)=1, sin(θ)=0 → rotation is identity. This is an algebraic identity, not an approximation. Capacity under uniform signal = 86.75% (matches √(384/512)=0.866 to 0.17%); real semantic tasks exceed this as domain signal concentrates in NoPE dims.

## Implications for Next Experiment

P1 adapters should target NoPE dims [128:512] of global attention heads — this guarantees position-free semantic routing and Grassmannian composition in a clean subspace. T0.5 (PLE injection) can use same algebraic-on-correct-dimensions approach.

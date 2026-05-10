# LEARNINGS.md — pico_rescues_dare_b (Finding #842)

## Core Finding

Pico calibration anti-synergizes with B-space DARE: 47.3% avg, 8pp worse than bare DARE-B (55.3%), 17.3pp below Fisher-Rao. The concentrated-B hypothesis is rejected — LoRA's multiplicative A@B structure is the dominant failure mode under random dropout.

## Why

Pico concentrates B-signal into fewer directions (by design, to reduce interference). DARE then randomly drops entries regardless of importance. The calibrated B has *more* concentrated signal, making random dropout *more* destructive. You cannot preserve E[A@B] by only controlling B while randomly masking it.

## Implications

1. **B-space DARE rescue line is closed.** No pre-processing of B can fix random dropout's incompatibility with factored LoRA structure. This applies to any future method that combines B-space calibration with random masking.

2. **Structure-aware vs random:** Viable B-space composition requires structure-aware methods (TIES sign consensus, Pico SVD without dropout). Random masking only works on fused A@B deltas (full-delta DARE).

3. **Hypothesis ranking confirmed:** Multiplicative-interaction failure >> concentrated-B failure. Future composition work should treat LoRA's factored form as a hard constraint, not an inconvenience.

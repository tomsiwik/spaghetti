# LEARNINGS.md — T0.5: PLE Injection Point Verification

## Core Finding

PLE injection (h' = h + RMSNorm(W_proj(SiLU(W_gate(h)) ⊙ e))) is algebraically exact as
identity when e=0 (max_diff=0.000e+00), strongly active when e≠0 (rel_diff=0.99/layer),
and gradient-trainable end-to-end (81.7% loss reduction in 200 steps, 128 params).

## Why

No-bias architecture is the structural guarantee: W_gate and W_proj have no bias terms,
so e=0 → gated=0 → proj_out=0 → RMSNorm(0)=0 → h+0=h exactly. This is Theorem 1 —
zero-init M2P output produces zero perturbation, making safe initialization structural
rather than empirical. Gemma 4 E4B weight inspection confirms no bias terms in PLE layers.

## Key Numbers

- K1004: max_diff = 0.0 EXACT (algebraic, not numerical)
- K1005: rel_diff = 0.9908 with unit-norm e (full-channel activation)
- K1006: 81.7% loss reduction (2.17 → 0.40, 200 steps, 128 trainable params)
- K1003: 42-layer stack stable (‖h‖=655.2, no NaN/Inf)

## Production Constraint

Scale e_l to ≤0.01 in production. Unit-norm e produces ‖h‖=655 for 42 layers —
this norm explosion is manageable but wastes capacity. T2.4 MATH.md MUST include
`e_l_scale = 0.01` as a design parameter, not a footnote.

## Implications for Next Experiment

T0 foundation is complete (T0.1 Grassmannian, T0.3 NoPE, T0.4 KV-sharing, T0.5 PLE).
T0.2 (V-Norm) killed — Gemma4 not loadable by mlx_lm 0.29.1.
T2.4 (PLE-M2P vs weight modification) is now unblocked: PLE is the injection mechanism,
adapters go on q_proj NoPE dims [128:512], KV sharing is structural — all three pieces
proven. First empirical test: does PLE-mediated domain vector beat direct weight delta?

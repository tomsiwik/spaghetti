# LEARNINGS.md — exp_m2p_n5_compose_qwen4b

**Finding #405 — supported**

## Core Finding

N=5 domain Grassmannian composition at 4B scale (Qwen3-4B, d=2560, r=4) is verified: max
cross-domain interference is 1.38e-05 across all 10 pairs, TF-IDF routing is 100% accurate,
and math quality_ratio=1.3125 exactly matches the N=2 baseline (Finding #404).

## Why

Theorem 4 is structurally guaranteed: exclusive routing means only one adapter fires per
query, so N=5 routed quality is identical to N=2 by construction. The real empirical content
is K978 (Gram-Schmidt isolation) and K979 (routing separability) together making that
guarantee load-bearing. Sequential Gram-Schmidt delivers progressively tighter isolation:
math×code pair inherits 1.38e-05 from 4B N=2; synthetic pairs reach sub-1e-09.

## Implications for Next Experiment

- N_max=640 >> N=5 means capacity is nowhere near a constraint at d=2560, r=4; the system
  can scale to at least N=25 production domains with zero structural headroom concerns.
- Next step: either scale to N=25 domains at 4B (production target from VISION.md) or
  address code-domain behavioral quality under composition (Finding #395: format overfitting
  degrades code output even when routed correctly).
- Caveats: sort/reverse/count adapters are synthetic — their task-accuracy under routing is
  not measured, only M2P loss (0.80–1.16). Real-domain N=5 verification requires 5 real
  M2P adapters, not 1 real + 4 synthetic.

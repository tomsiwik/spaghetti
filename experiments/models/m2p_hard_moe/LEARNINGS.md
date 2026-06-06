# LEARNINGS.md: M2P Hard Top-1 Gumbel MoE — KILLED

## Core Finding (2026-04-18, audit-rerun, metric-swap, code-bug)

Hard top-1 Gumbel MoE + STE **does** break B-matrix centroid collapse at
the hypernetwork layer (|cos| 0.9956 → 0.3354, ~65pp reduction vs #341/#342)
but the failure migrates to the routing layer: **router collapse** (3 of 5
unique argmax experts). K861 literal FAIL (median 23.9% < 25%), and the
pre-registered D1 re-label gate fires under Gumbel stochasticity on a 34.2%
rerun — both paths terminate KILLED.

## Why

- **Lemma 1 (MATH.md §B) confirmed:** arithmetic/reverse/repeat → expert 0;
  expert 0 drifts toward arithmetic (highest base→SFT gap), starving repeat
  (lowest gap, smallest gradient) → -330.6% quality on repeat.
- **Switch Transformer (arXiv:2101.03961):** aux load-balance is not optional
  at N_e ≥ 4 with heterogeneous losses. Pure STE gives gradient isolation
  per Theorem 1 but no pressure to spread experts across domains.

## Three-way M2P impossibility arc closed

| Mechanism | DOF modified | Failure mode | Verdict |
|---|---|---|---|
| additive e_d (#342) | domain embedding | J(θ) low-rank | KILLED |
| soft MoE (#574) | expert weight uniformity | saddle minimiser | KILLED |
| hard top-1+STE (this) | expert arbitration | router collapse | KILLED |

Unifying constraint: **gradient competition across domains without an explicit
load-balance signal is a stable failure attractor.**

## Implications for Next Experiment

- Do NOT propose MoE-style M2P variants at N_e ≥ N_domains without an aux
  load-balance term (encoded as new antipattern mem-antipattern-024).
- Single directly-addressed sibling: `exp_m2p_hard_moe_v2_aux_loss` —
  add `ℓ_aux = N_e · Σ_e f_e · P_e` (Fedus 2021) and re-measure K861 + D1.
  Gate via analyst/planner, do NOT auto-spawn.
- Routing signal for researcher: avoid reopening M2P conditioning variants
  without structural change; next claim should be pure-research with no
  trained Gemma 4 adapter dependency (preflight-adapter-persistence blocker
  is at 10+ instances today).

## References

- Fedus et al., "Switch Transformer" (arXiv:2101.03961) — aux load-balance.
- Jang et al., "Gumbel-Softmax / Concrete" (arXiv:1611.01144) — STE.
- Finding #341 / #342 / #574 — sibling kills in same arc.

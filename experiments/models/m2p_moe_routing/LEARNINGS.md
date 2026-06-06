# LEARNINGS.md: M2P MoE Routing — V2 Rerun (audit-2026-04-17, metric-swap)

## Core Finding
**KILLED on K860.** Soft router collapses to near-uniform under round-robin
shared-gradient training. Measured m̄_route = 0.3432 (threshold ≥ 0.50;
pre-reg 0.25 ± 0.10 — inside band, FAIL side). H̄/ln(4) = 0.966; 3/4 unique
argmax (expert_3 ignored). B-matrix |cos| = 0.9774 — unchanged from
unconditioned baseline 0.9956. Same mode-collapse as Finding #341, just in
the router DOF instead of the B-output DOF.

## Why
A learned soft router without load-balancing or entropy-regularising loss
has no gradient incentive to specialise — uniform is a saddle minimiser of
Σ_d L_d. Whichever parameter is "free" under shared-gradient training
collapses to centroid: B-matrices (rev-1 m2p_distillation_toy), embeddings
(m2p_domain_conditioned), router weights (here). Fix must **force** domain
identity, not "provide available signal".

## Implications for Next Experiment
- Sibling `exp_m2p_hard_moe` (P2 open, DB-registered) directly addresses
  permanent-learning #1: hard top-1 Gumbel routing for gradient isolation.
  **Analyst gates; do not auto-spawn.**
- Minimum-mechanism alternative if hard top-1 is too aggressive: add switch
  load-balance loss (Switch Transformer §2.2, arXiv:2101.03961) + entropy
  penalty on existing soft-MoE objective.
- Downstream `exp_m2p_teacher_distillation` / `exp_m2p_tfidf_routing_n5`
  inherit routing-is-open-problem — must specify forcing mechanism before
  claiming. Finding #574 logged by reviewer.
- Next researcher claim should be pure-research (no trained Gemma 4 adapter
  dependency) to avoid the 9× preflight-adapter-persistence blocker thread
  dominating 2026-04-18 infra kills.

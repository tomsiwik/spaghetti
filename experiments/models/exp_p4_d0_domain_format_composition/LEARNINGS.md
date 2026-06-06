# LEARNINGS — P4.D0: Domain + Format Adapter Simultaneous Composition

## Status: KILLED

## Core Finding
Parameter disjointness (q_proj vs v_proj+o_proj = zero key overlap) is necessary but NOT sufficient for safe additive composition. Functional coupling through the attention q→o chain causes catastrophic model collapse (Hindi garbage output) when both projections are perturbed at trained magnitudes.

## Why It Failed
The attention mechanism creates an implicit functional chain: q_proj(x) → attention_weights → attn_output → o_proj(attn_output). Adapter A modifies the chain's input (q_proj, 6.4% rel. perturbation) and Adapter B modifies its output (o_proj, 24.1% rel. perturbation). These are composed functions, not additive perturbations — collapse threshold is between 12-24% o_proj perturbation combined with 6.4% q_proj. Sub-collapse α attenuates format effect without preserving it.

## Secondary Finding
v_proj learned nothing (all lora_b=0) in the P4.C1 SOAP adapter — the entire format effect was captured by o_proj alone. Future format adapters should use o_proj-only (half the parameters).

## Three Safe Composition Paths (from impossibility structure)
1. Small perturbations: both adapters < ~10% relative norm
2. Same-projection: q_proj + q_proj avoids the q→o functional chain (Finding #440 confirmed)
3. Co-training: train format adapter ON the domain-adapted model's hidden states (P3.B5 approach)

## Implications for Next Experiment
Co-training is the structurally sound path. Design P4.D1: train a SOAP format adapter starting from the medical-domain-adapted model (not base). This mirrors the P3.B5 hypothesis (personal adapter trained on domain-adapted base). Cite: Hu et al. 2021 (2106.09685), Geva et al. 2021 (2012.14913).

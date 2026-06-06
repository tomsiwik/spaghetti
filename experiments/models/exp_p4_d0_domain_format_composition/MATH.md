# MATH.md — P4.D0: Domain + Format Adapter Simultaneous Composition

## Motivation

P4.C1 (Finding #480) proved that format priors (SOAP +70pp, Legal +90pp) require
v_proj+o_proj LoRA. Domain knowledge (Finding #421) uses q_proj LoRA. These target
**completely disjoint parameter sets**. This experiment tests whether simultaneous
additive composition preserves both capabilities — domain knowledge AND format compliance.

## Prior Results

- **Finding #421**: q_proj rank-6 LoRA achieves +22pp medical, +50pp legal domain knowledge
- **Finding #480**: v_proj+o_proj rank-16 LoRA achieves +70pp SOAP, +90pp legal format
- **Finding #440**: Grassmannian isolation — max cos = 2.25e-8 at N=100 adapters
- **Finding #429**: Hot-add composition preserves individual adapter quality

## Definitions

Let W_base denote the frozen base model parameters. For layer l, projection p:

- Domain adapter: ΔW_D^{l,p} = B_D^{l,p} · A_D^{l,p} where p ∈ {q_proj}, l ∈ {0,...,41}
- Format adapter: ΔW_F^{l,p} = B_F^{l,p} · A_F^{l,p} where p ∈ {v_proj, o_proj}, l ∈ {30,...,41}

Composed model: W_composed^{l,p} = W_base^{l,p} + ΔW_D^{l,p} + ΔW_F^{l,p}

## Theorem 1: Exact Disjointness of Parameter Subspaces

**Theorem.** The domain adapter ΔW_D and format adapter ΔW_F modify disjoint sets
of model parameters. Formally:

Let P_D = {(l, p) : ΔW_D^{l,p} ≠ 0} and P_F = {(l, p) : ΔW_F^{l,p} ≠ 0}.

Then P_D ∩ P_F = ∅.

**Proof.** By construction:
- P_D = {(l, q_proj) : l ∈ {0, 1, ..., 41}} (84 weight matrices)
- P_F = {(l, v_proj) : l ∈ {30,...,41}} ∪ {(l, o_proj) : l ∈ {30,...,41}} (48 weight matrices)

Since q_proj ∉ {v_proj, o_proj}, we have P_D ∩ P_F = ∅. QED.

**Corollary.** Additive composition W_composed = W_base + ΔW_D + ΔW_F introduces
exactly zero interference at the parameter level. Each adapter's weight perturbation
is applied to parameters that the other adapter does not touch.

## Theorem 2: Composition Preserves Individual Capabilities

**Theorem.** If adapter A modifies parameter set P_A and adapter B modifies disjoint
set P_B (P_A ∩ P_B = ∅), then the composed output at any layer l using projection p
is identical to the solo adapter output for whichever adapter owns that projection:

For (l, p) ∈ P_A: h_composed^{l,p} = h_{base+A}^{l,p}
For (l, p) ∈ P_B: h_composed^{l,p} = h_{base+B}^{l,p}
For (l, p) ∉ P_A ∪ P_B: h_composed^{l,p} = h_base^{l,p}

**Proof.** Since the parameter perturbations are disjoint, at each projection site
exactly one adapter (or none) is active. The linear transformation at that site is
W_base + ΔW_X where X is the unique adapter modifying that projection (or ΔW_X = 0).
No other adapter's perturbation contaminates this site. QED.

**Caveat (functional interference).** Theorem 2 guarantees parameter-level isolation
but NOT functional isolation. The domain adapter changes the query vectors (q_proj),
which changes attention patterns, which changes the hidden states flowing into format
adapter layers (v_proj, o_proj). This is **sequential functional dependency**, not
interference — the format adapter processes domain-enriched representations. The
question is whether this functional composition is beneficial (domain knowledge +
correct formatting) or destructive (representations shift enough to break format
adapter's learned transformations).

## Predictions

Based on Theorem 1 (exact disjointness) and Theorem 2 (preservation with the
functional dependency caveat):

| Prediction | Expected | Reasoning |
|---|---|---|
| Medical + SOAP: domain quality | ≥30% (base medical rate ~26%, adapter +22pp) | q_proj medical adapter active; domain queries produce medical content |
| Medical + SOAP: format compliance | ≥40pp improvement | v_proj+o_proj SOAP adapter active; P4.C1 showed +70pp solo |
| Legal + Legal-brief: domain quality | ≥40% (base legal ~48%, adapter +50pp) | q_proj legal adapter active |
| Legal + Legal-brief: format compliance | ≥60pp improvement | v_proj+o_proj legal adapter active; P4.C1 showed +90pp solo |
| Solo degradation under composition | ≤10pp | Disjoint parameters mean no weight interference; any degradation from functional dependency |

The format compliance predictions are slightly conservative vs solo performance
because functional dependency (domain-shifted hidden states entering format layers)
may cause minor degradation from the format adapter's trained distribution.

## Kill Criteria (from DB)

- **K1249**: Medical + SOAP: domain_quality ≥40% AND format_compliance ≥50pp
- **K1250**: Legal + Legal-brief: domain_quality ≥40% AND format_compliance ≥60pp
- **K1251**: Solo adapter degradation ≤15pp under 2-adapter composition

## Experimental Design

1. Load pre-trained adapters (no new training needed):
   - Medical domain: `exp_p1_t2_single_domain_training/adapters/medical/` (q_proj rank 6, all layers)
   - Legal domain: `exp_p1_t2_multi_domain_5/adapters/legal/` (q_proj rank 6, all layers)
   - SOAP format: `exp_p4_c1_vproj_soap_adapter/soap_adapter/` (v_proj+o_proj rank 16, layers 30-41)
   - Legal format: `exp_p4_c1_vproj_soap_adapter/legal_adapter/` (v_proj+o_proj rank 16, layers 30-41)

2. Merge each pair by concatenating safetensor dictionaries (no key collisions guaranteed by Theorem 1)

3. Evaluate with N=10 questions per condition:
   - Medical questions → score for medical vocabulary AND SOAP format
   - Legal questions → score for legal vocabulary AND legal boilerplate format
   - Solo adapters on same questions → measure degradation

## References

1. Finding #421: LoRA r=6 q_proj achieves 22-82pp domain improvement (P1.T2)
2. Finding #480: v_proj+o_proj unlocks SOAP +70pp, Legal +90pp (P4.C1)
3. Finding #440: Grassmannian isolation N=100 (P1.T3.4)
4. Hu et al. 2021 (arxiv 2106.09685) — LoRA layer selection analysis
5. Geva et al. 2021 (arxiv 2012.14913) — attention value vectors as memories

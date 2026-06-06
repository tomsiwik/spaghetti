# LEARNINGS.md — exp_p2_a1_tier2_plus_tier3_composition

## V2 Audit Update (2026-04-18)
V2 reconstruction confirms KILLED (not a flip from V1). Rerun at full N=25 blocked: math adapter weights deleted from `exp_p1_t2_single_domain_training/adapters/math/`. K3 is an algebraic weight-space measurement (N-independent), so its pre-reg KC failure (0.1607 > 0.1) is conclusive without a full rerun. K2's 100pp categorical swing at n=5 is ~4.5σ, also not sampling noise. Antipattern `smoke_as_full` flagged but reconciled by N-independence of K3.

**Cross-reference**: Finding #425 (N=5 simultaneous kill) + this experiment (N=2 matched kill) together demonstrate that the failure is **not adapter count** but B-matrix alignment + power imbalance on independently-trained adapters. Sequential hot-add (T3.6 / Finding #429) or Grassmannian re-orthogonalization (Finding #428) are the two fix pathways.

## Core Finding
Simultaneous activation of Tier 2 (domain) + Tier 3 (personal style) adapters via naive weight
addition catastrophically destroys personal style (100pp loss) due to two independent structural
violations: B-matrix non-orthogonality (ε_B=0.1607 > 0.10) and power imbalance (2.96× in favor
of domain adapter). The violation factor is 3.6× above the threshold — not noise.

## Why
The impossibility bound is: ε_B × (S_D / S_P) < compliance_threshold / personal_only_rate.
Measured 0.476 >> 0.132. Personal style adapters are fragile because they encode a SPECIFIC token
sequence; any adapter that shifts attention patterns away from that sequence destroys compliance.
The math adapter was trained on GSM8K which rewards concise direct answers — the exact opposite
of style tokens like "Hope that helps, friend!".

## Implications for Next Experiment
Sequential activation is the correct path: apply domain adapter first, then personal adapter via
T3.6 hot-add (Finding #429). No new orthogonalization math required — the existing hot-add
mechanism already handles this cleanly. Grassmannian re-orthogonalization + scale normalization
are valid but harder; save for if sequential activation fails.

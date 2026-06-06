# LEARNINGS: P6.B0 — Adapter Sharing Flywheel

## Core Finding
Crystallization (B-matrix averaging) creates a genuine coverage union: the crystal encodes
facts from 5 different users' training sets in a single adapter. Crystal-as-init for
continued training causes catastrophic forgetting (50%→20%), ruling it out permanently.

## Why
The shared A-matrix couples all fact directions — gradient updates from the new user's 6
facts perturb B-matrix encodings for ALL other facts, including the 4 outside their window.
This violates Theorem 4's implicit assumption that ∂L/∂B_crystal ≈ 0 for non-overlapping
facts. Signal attenuation (0.6× from 10-user average) also explains why crystal matches
rather than exceeds the best individual: attenuated signal hovers at the keyword-matching
decision boundary.

## Implications for Next Experiment
The correct flywheel is **frozen crystal + separate user adapter** with multi-adapter
composition at inference time (W_out = W_base + A@B_crystal + A@B_user), already proven
feasible at N=25 (Finding #225). The next experiment should test this two-adapter
composition pattern: freeze the crystal, train a fresh user adapter, measure whether
the combined system exceeds both crystal alone and the zero-init baseline.

## References
- Finding #225: N=25 adapter composition at near-lossless quality (justifies multi-adapter path)
- Finding #490 (P6.A0): Online LoRA baseline, repetition pathology at rank-4
- Finding #451 (T6.2): B-matrix crystallization cosine=0.9806 (mechanism sound)
- arXiv:2203.05482 (Model Soup): weight averaging signal attenuation theory

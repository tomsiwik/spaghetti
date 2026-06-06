# LEARNINGS.md — T6.5: Dynamic Adapter Registry

## Core Finding
A Grassmannian-backed adapter registry supports four lifecycle operations (register, remove, promote-to-base, crystallize) with algebraically correct composition: 8 adapters → remove(1) → promote(1) → crystallize(3→1) → 4 adapters, final max_cos=0.122 < τ=0.15.

## Why
Registry consistency (O(1) remove, O(N·d) register, Davis-Kahan bounded promote) follows from prior T6.2/T6.3 theorems applied sequentially. The near-orthogonality invariant holds across domains but not within a cluster slot before crystallization — same-domain variants correctly exceed τ until crystallized (Finding #452, Task Arithmetic arxiv:2212.04089).

## Implications for Next Experiment
T6 tier is complete. The full lifecycle — train → validate → submit → cluster → crystallize → promote — is algebraically verified end-to-end. Next step: C0/C1 corrective tier (PoLAR + composition on real Gemma 4 weights) to validate the same operations on non-synthetic bases, then T5.1 user local training resumes.

## Caveats
- Kill thresholds were 1000–667K× above measured values; these are correctness proofs, not tight performance bounds.
- Promote ε (3.63%/4.78%) reuses synthetic std=0.05 weights from T6.3 — not new evidence, but confirms sequential composition is correct.
- "Throughout" Grassmannian invariant holds across domains; same-domain variants cluster before crystallization without violating the inter-domain guarantee.

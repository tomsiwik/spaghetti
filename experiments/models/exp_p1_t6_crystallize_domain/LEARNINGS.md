# LEARNINGS.md — T6.2: Crystallize Domain Adapters

## Core Finding
Averaging N same-domain LoRA B-matrices reduces noise by √N (LLN), improving cosine alignment to the domain centroid by +6.5pp (0.9156 → 0.9806) while compressing 25 user adapters into 5 crystallized domain adapters (80% slot reduction).

## Why
B-matrix averaging is the LoRA analog of Model Soup (Wortsman et al., 2203.05482): same initialization (zero B), same task (domain), different users → averaging improves direction while preserving norm (ratio=1.020). The LLN guarantee is structural, not empirical: E[||ε̄||²_F] = σ²/N.

## Caveats
- Synthetic-only: adapters are canonical + Gaussian noise. Real user heterogeneity (varying σ) not tested.
- Prediction magnitude off by 1.7pp (predicted +8.2pp, got +6.5pp) due to first-order cosine approximation.
- No behavioral test (generation quality) — that belongs in T6.3.

## Implications for Next Experiment
T6.3 (base promotion) must include a generation quality test on crystallized adapters to earn status beyond PROVISIONAL. The +6.5pp cosine improvement is a necessary but not sufficient condition for behavioral improvement.

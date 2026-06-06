# LEARNINGS.md — T5.2: User Adapter Validation Pipeline

## Core Finding
A 4-check CPU+inference pipeline (orthogonality, quality, safety, scale) validates user adapters
in 23.5s with zero human review, and confirmed that T5.1's 76% style compliance was genuine
behavioral learning — not a token-truncation artifact (90% at max_tokens=256).

## Why
Under exclusive routing (T3.6/T3.7), orthogonality between user and domain adapters is
structurally irrelevant for interference; the K1100 check serves only to detect near-duplicate
adapters (cos ≥ 0.95). Style adapters (rank-4, layers 26-41) occupy geometrically different
subspace from domain adapters, yielding max|cos|=0.2528 — roughly 2.7× above the random
baseline (E[σ₁]≈0.062), well below the rejection threshold.

## Implications for Next Experiment
T5.3 (Privacy/Safety) should upgrade the keyword safety filter (K1102) to an NLI-based or
embedding-similarity classifier — the current filter is too brittle for production. The
validation pipeline itself is production-ready pending that upgrade.

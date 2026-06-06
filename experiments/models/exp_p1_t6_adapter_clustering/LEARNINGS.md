# LEARNINGS.md — T6.1: Adapter Clustering by Domain Similarity

## Core Finding

B-matrix cosine K-means recovers domain structure from 25 user adapters with perfect purity
(silhouette=0.8193, all 5 domains purity=1.0 at K=5) — without accessing any user training data.

## Why It Works

Domain-specific gradient flow produces directionally distinct B-matrices (cross-domain cosines
0.015–0.20; same-domain cosines ≈ 0.80–1.0). The directional signal overwhelms intra-cluster
noise (σ/Δ ≈ 4.9 in L2 but near-zero in cosine space), making cosine K-means the correct
representation. Confirmed by Task Arithmetic (2212.04089): adapter directions are task-specific.

## Key Observation

Semantically related domains (medical-legal-finance, cosines 0.12–0.20) cluster later than
STEM domains (math-code, cosines < 0.03). This predicts that at K<5, knowledge domains merge
first — a structured failure mode, not random noise. Scale pattern from Finding #217 preserved:
learnable-task domains (math, code) have higher ||B||_F ≈ 5.7–5.8 vs knowledge domains ≈ 4.4–4.8.

## Implications for T6.2 (Crystallization)

- Clustering gate: collect B-matrices, run K-means after every 10 new adapters
- Crystallization trigger: silhouette > 0.5 at optimal K (heuristic — T6.2 should formalize)
- K selection in production needs principled method (gap statistic or BIC), not brute-force scan
- Real user heterogeneity (variable training length, LR) may lower silhouette below 0.8 — test
  with σ = 2×std(B) noise before shipping the T6.2 threshold

## Status: SUPPORTED — Finding #450

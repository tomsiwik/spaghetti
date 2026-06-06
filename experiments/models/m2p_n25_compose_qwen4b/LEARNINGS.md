# LEARNINGS.md — N=25 Grassmannian Composition at 4B

**Finding #406 | Status: supported | 2026-04-08**

## Core Finding

N=25 domain Grassmannian composition verified at 4B (Qwen3-4B-4bit): isolation=1.38e-05 across all 300 pairs, TF-IDF routing=99.96%, quality_ratio=1.3125 — identical to N=2 and N=5. N_max=640 confirmed; 96.1% capacity remains.

## Why

Theorem 4 is logically airtight: with exclusive TF-IDF routing, a query activates exactly one domain adapter, making N computationally equivalent to N=1 for any given query. Gram-Schmidt A-matrices are orthogonal by construction at d=2560, r=4, so cross-domain interference approaches bf16 quantization floor (~1e-5) regardless of N.

## Key Caveat (Reviewer)

The quality invariance result is circular: all three N-experiments (2, 5, 25) load the **same** math M2P weights. We've proven infrastructure scales, not that 25 independently trained behavioral adapters compose without interference. "Production-ready for 25 domains" is premature — that claim requires 25 real trained adapters evaluated concurrently.

## Implications for Next Experiment

**P0 path:** Train real domain adapters for 2+ non-math domains (e.g., code + medicine) using SFT-residual M2P at 4B, then verify composition with all real B-matrices (not B=0). This closes the behavioral gap the reviewer identified. Alternatively, pursue Level 4 (production serving: adapter hot-swap, per-token routing).

## Scaling Law Confirmed

| N | qr | isolation | routing | Finding |
|---|----|-----------|---------|---------|
| 2 | 1.3125 | 1.38e-05 | 100% | #404 |
| 5 | 1.3125 | 1.38e-05 | 100% | #405 |
| 25 | 1.3125 | 1.38e-05 | 99.0% min | #406 |

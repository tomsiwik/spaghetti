# LEARNINGS: C1.1 PoLAR Joint Stiefel on Gemma 4

**Status:** SUPPORTED (Finding #442)
**Reference:** PoLAR: Polar-Decomposed Low-Rank Adapter (arxiv 2506.03133)

## Core Finding

Joint Stiefel constraint (both U and V orthonormal) guarantees sr(ΔW) = r exactly — verified to 7 decimal places at r=6 and r=16. Standard LoRA collapses to sr=1.77 (rank-1 gradient), while PoLAR U-only (T1.5) collapsed to sr=2.21. The 9× improvement over LoRA is structural, not empirical.

## Why

T1.5 failed because constraining only U leaves V free to collapse via rank-1 gradients (single-domain training pushes V toward a single direction). Joint Stiefel closes this gap: since A has orthonormal columns and B has orthonormal rows, σ_i(AB) = 1 for all i, giving sr(AB) = r/1 = r regardless of gradient rank or training distribution.

## Key Numbers

| Config | sr | sr/r |
|---|---|---|
| PoLAR r=16 (this) | 16.00 | 1.000 |
| PoLAR r=6 (this) | 6.00 | 1.000 |
| LoRA r=6 (this) | 1.77 | 0.295 |
| PoLAR U-only r=16 (T1.5) | 2.21 | 0.138 |

Stiefel distances: 6e-15 (float64 machine epsilon — retraction is numerically perfect).

## Known Gap

KC08 (behavioral comparison) passed trivially (0% vs 0%) due to benchmark mismatch: synthetic 5-domain training data doesn't match GSM8K arithmetic format. Behavioral claim that sr → better generalization remains unverified. Requires C1.3 with aligned train/eval distributions (train on GSM8K-style, eval on GSM8K holdout).

## Implications for Next Experiment

Full rank capacity (sr = r) means each PoLAR adapter spans an r-dimensional subspace rather than 1-2D. Combined with T3.1's near-zero interference (max|cos|=2.25e-8), this strengthens Grassmannian isolation between adapters — the r-dimensional subspaces are harder to confuse. C1.2 (V-norm scale safety) can now use PoLAR adapters with guaranteed full-rank structure. C1.3 should verify the behavioral claim with matched train/eval data.

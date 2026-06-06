# PAPER.md: Grassmannian Orthogonality at N=50 on Qwen3-0.6B

## Prediction vs Measurement

| Prediction (MATH.md) | Measured | Status |
|---------------------|----------|--------|
| max \|A_i^T A_j\|_F < 1e-5 (K948) | 9.50e-08 | ✓ PASS (105× margin) |
| Total memory (all 7 types) < 5GB (K949) | 0.252 GB | ✓ PASS (20× margin) |
| Memory for q+v only | 57.3 MB (predicted) | 57.3 MB | ✓ Exact |
| Memory for all 7 types | ~241 MB (predicted) | 252.3 MB | ✓ Within 5% |
| Self-orthonormality \|A_i^T A_i - I\|_F | ε_mach·scale ≈ 1e-6 | 2.75e-07 | ✓ Matches |
| N_max theoretical | 256 | 256 | ✓ Exact |
| N=50 as % of capacity | 19.5% | 19.5% | ✓ Exact |

## Results Summary

**K948 (PASS):** max pairwise cross-subspace norm = 9.50e-08 across 1,225 pairs (N=50).
This is 105× better than the 1e-5 threshold and matches Theorem 1's prediction of
~7.6e-6 (actually much better — float32 QR achieves near machine epsilon).

Distribution: p50=3.88e-08, p90=5.38e-08, p99=7.24e-08, max=9.50e-08.
All 1,225 pairs are numerically zero — no outliers or near-collinear pairs.

**K949 (PASS):** 252.3 MB for all 7 weight types across 28 layers and 50 adapters.
Equivalent to 57.3 MB for q+v projection only (the M2P use case).
The theoretical maximum (N=256 adapters, 7 types) would be ~1.29 GB — still under 5 GB.

**Runtime:** 0.1 seconds for QR construction + 1,225 pairwise checks.

## Key Insights

1. **Scale confirmation:** Orthogonality proven at N=2 (Finding #390, 1.51e-08) holds
   at N=50. The QR construction is numerically stable at production scale.

2. **Memory is not the bottleneck:** 25 domain adapters × full 7-type coverage = 12.6 MB
   per adapter set — negligible compared to the 4-bit quantized model (~345 MB for Qwen3-0.6B).
   N=256 (theoretical max) would require ~1.29 GB total.

3. **Production readiness:** The capacity and memory bounds fully support the 25-domain
   target (VISION.md). Each new domain costs ~5 MB of adapter storage.

## Connection to Vision

This experiment closes the "can we scale to 25 domains?" question for the composition
infrastructure. The bottleneck is NOT:
- Grassmannian capacity (N_max=256, using 19.5% at N=50)
- Memory (252 MB for 50 domains vs 5 GB limit)

The remaining bottleneck is M2P adapter quality at full training scale (K954 unverified).

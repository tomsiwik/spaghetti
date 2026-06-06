# TIES-Merging on Shared-A Materialized Deltas

## Abstract

We test TIES-Merging (arxiv 2306.01708) — Trim + Sign-Elect + Disjoint mean —
as a structured composition method for Pierre's shared-A PoLAR architecture.
Unlike ACE (covariance-weighted) and OrthoMerge (Riemannian), TIES operates
element-wise on materialized deltas ΔW_t = scale · A · B_t.

**Result: KILLED.** TIES achieves 66.7% avg vs Fisher-Rao's 64.7% (+2pp, below
the +3pp threshold) and sits 4.7pp below full-delta DARE (71.3%), exceeding
the 4pp gap tolerance. TIES is not viable for Pierre.

## Prediction vs Measurement

| Metric | Prediction | Measured | Verdict |
|--------|-----------|----------|---------|
| K1: TIES avg ≥ FR + 3pp | ≥ 67.7% | 66.7% (+2.0pp) | **FAIL** |
| K2: DARE − TIES ≤ 4pp | ≤ 4.0pp gap | 4.7pp gap | **FAIL** |
| K3: Preprocess ≤ 30s | < 30s | 2.7s | **PASS** |

## Per-Benchmark Breakdown

| Benchmark | Fisher-Rao | TIES | DARE | Single-Best |
|-----------|-----------|------|------|-------------|
| gsm8k | 68.0% | 70.0% | 72.0% | 66.0% |
| humaneval | 68.0% | 78.0% | 80.0% | 78.0% |
| medqa | 58.0% | 52.0% | 62.0% | 42.0% |
| **avg** | **64.7%** | **66.7%** | **71.3%** | 62.0% |

## Analysis

TIES shows a split personality across benchmarks:
- **humaneval**: +10pp over Fisher-Rao — trimming removes noise, sign-election
  resolves code-vs-strategy interference effectively.
- **medqa**: −6pp below Fisher-Rao — aggressive trimming (keep_frac=0.3) destroys
  medical domain signal, which is sparse and low-magnitude.

The net effect: marginal +2pp average, insufficient to justify the complexity.

## Why TIES Underperforms on Shared-A

Unlike ACE/OrthoMerge which fail catastrophically (26.7% and crash respectively),
TIES at least functions — its element-wise operations don't assume independent
parameterization. However, shared-A creates correlated magnitude profiles across
adapters (all ΔW_t share the same left factor A). This means:

1. **Trim** removes similar entries across all adapters (correlated magnitudes),
   reducing diversity rather than noise.
2. **Sign-election** becomes near-unanimous (shared-A biases sign patterns),
   making the disjoint-mean step degenerate toward simple averaging.

The result: TIES on shared-A ≈ slightly-filtered average, not the structured
interference resolution the paper achieves on independently-trained models.

## Conclusion

TIES is the third structure-aware merge method killed for Pierre (after ACE and
OrthoMerge). The pattern is clear: shared-A constrains the delta manifold such
that methods designed for independently-parameterized models lose their
discriminative power. Fisher-Rao and DARE, which don't attempt per-task structure
inference, remain the correct baselines.

## References

- Yadav et al. (2023). "Resolving Interference When Merging Models." arxiv 2306.01708
- Reference impl: prateeky2806/ties-merging

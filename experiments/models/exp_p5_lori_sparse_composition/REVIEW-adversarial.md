# Review: P5.A0 LoRI Sparse-Mask Composition

**Verdict: KILLED** (confirmed)

## Proof Error (Critical)

MATH.md Theorem 1 claims `<ΔW_i, ΔW_j>_F = 0` for disjoint masks. The proof argues that `diag(m_i) B_i^T B_j diag(m_j) = 0` because "for every non-zero row k, the corresponding column k in diag(m_j) is zero." This only eliminates **diagonal** entries (k,k) where k cannot be in both supports. Off-diagonal entries (k,l) with k in supp(m_i), l in supp(m_j) survive because both mask values are 1. The actual inner product reduces to:

```
<ΔW_i, ΔW_j>_F = Σ_{k∈m_i, l∈m_j} (B_i^T B_j)_{k,l} · (AA^T)_{k,l}
```

which is non-zero whenever A has non-zero cross-block Gram entries (always, for random A).

PAPER.md correctly identifies and explains this error. The B-space orthogonality (trivially 0.0) is clearly distinguished from weight-space interference (1.33e-3). Good diagnosis.

## Kill Criteria Verification

| K | Claim | results.json | Verified |
|---|-------|-------------|----------|
| K1264 | max\|cos\| = 1.33e-3, FAIL | `weight_space_max_abs_cos: 0.00133` | Yes |
| K1265 | quality_ratio = 0.989, PASS | `quality_ratio: 0.9887` | Yes |
| K1266 | max_degradation = 222.9%, FAIL | `max_degradation_pct: 222.89` | Yes |

All values match. No fabrication.

## What Worked

- Solo adapter quality is genuinely excellent (97-99% PPL reduction per domain)
- 2.4x parameter savings from frozen A is real and useful
- B-space orthogonality measurement confirms the trivial guarantee works

## Key Insight (Valid)

LoRI's mask disjointness gives parameter-space (B) orthogonality, NOT weight-space (ΔW) orthogonality. The gap is exactly `AA^T` cross-block terms. Grassmannian A eliminates these by construction. This cleanly explains why Grassmannian composition (Finding #440, max cos 2.25e-8) outperforms LoRI composition (max cos 1.33e-3) by 60,000x.

## Minor Notes (Non-blocking)

1. Theorem 3 ("impossibility of interference") inherits the same proof error as Theorem 1 -- it claims structural guarantee for weight-space, but only holds for B-space.
2. The JL-bound calculation in Theorem 2 (r_JL=138 but using r=6) is acknowledged as below-bound, which is fine since K1265 tests this empirically.

## Disposition

KILLED confirmed. Two clear structural failures, proof error correctly identified, composition failure root-caused. Ready for LEARNINGS.md.

# LEARNINGS: P5.A0 — LoRI Sparse-Mask Composition

**Status: KILLED** | Finding #487

## Core Finding

LoRI's disjoint B masks guarantee **B-parameter-space** orthogonality (trivially 0.0)
but NOT **weight-space** orthogonality (measured 1.33e-3, ~60,000x worse than Grassmannian).
Composition is catastrophic: up to 222.9% PPL degradation across 4 of 5 domains.

## Why

Weight-space inner product `<ΔW_i, ΔW_j>_F = Σ_{k∈m_i, l∈m_j} (B_i^T B_j)_{k,l} · (AA^T)_{k,l}`.
Disjoint masks zero the diagonal (k=l) but leave off-diagonal cross-block entries of the
shared A Gram matrix, which are O(1/√d_in) ≈ 1/50 for random A. The MATH.md Theorem 1
proof was incorrect — it conflated diagonal and off-diagonal cases.

Shared A forces all adapters through the same input projection subspace, so every adapter
modifies every token. Without per-token routing or orthogonal per-adapter A blocks,
additive composition sums noise from all N-1 irrelevant adapters onto every input.

## What Worked

Solo adapter quality is excellent: 98.9% of standard LoRA quality with 2.4x fewer
trainable parameters (516K vs 1.16M per adapter). Frozen A is a valid parameter reduction
for **single-adapter** deployment. Not useful for composition.

## Impossibility Structure

Weight-space interference is impossible to eliminate with shared A alone.
The Gram matrix AA^T always has non-zero cross-block entries for random A.
Only Grassmannian A (orthogonal row blocks per adapter) zeros these terms by construction,
giving the 2.25e-8 max cos seen in Finding #440.

## Implications for Next Experiment

The structural fix is clear: **Grassmannian A + disjoint B masks** gives both guarantees
simultaneously — weight-space orthogonality (from A) AND B-parameter savings (from masks).
This is strictly better than either LoRI or pure Grassmannian alone.
Alternatively: LoRI solo adapters + routing (no additive composition) is viable as a
parameter-efficient serving strategy.

## Reference

- arXiv:2504.07448 (LoRI) — frozen A + disjoint masked B
- Finding #440 — Grassmannian max cos 2.25e-8 (weight-space orthogonality baseline)
- Finding #59 — prior LoRI on BitNet-2B (null result, different reason)

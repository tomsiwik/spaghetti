# P7.C1: Self-Organizing Adapter Slots — KILLED (Prior Findings)

## Summary

This experiment was killed before implementation. Three prior findings (#495, #498, #500) collectively prove that null-space A-matrix positions cannot carry routing or quality information. Self-organizing adapter positions in null space is impossible because domain features are orthogonal to null space by construction.

## Prediction vs Measurement Table

| Kill Criterion | Prediction | Prior Evidence | Verdict |
|---|---|---|---|
| K1309: Routing >= 85% | ~20% (chance) | Finding #495: 20.0% accuracy, Spearman r=-0.19 | KILLED |
| K1310: Orthogonality < 0.1 | Already achieved | Grassmannian QR: closed-form optimal | MOOT |
| K1311: Auto slot-finding | Impossible | Finding #498: A-matrices cluster by init, not domain | KILLED |

## Key Evidence Chain

1. **Finding #495 (Null-Space Routing Killed):** |A_i Q^T x|^2 achieves 20% accuracy (chance). Domain info D in range(W_v), routing signal s in null(W_v). Since range ⊥ null: ⟨s,D⟩ = 0. No normalization or training can create domain signal in null space.

2. **Finding #498 (A-Matrices Are Init-Determined):** A-matrices cluster by initialization method (cos ≈ 0.82 for standard init, cos ≈ 0 for Grassmannian), not by domain. B-matrices carry domain structure (rank-5 for 5 domains). Training A-matrices cannot move them to domain-relevant positions.

3. **Finding #500 (Null-Space Quality Killed):** Projection magnitude AUC = 0.4293 (below chance). I(projection; quality) ≈ 0. Feedback-calibrated routing: 0.00pp improvement.

## Conclusion

Self-organizing adapter slots in null space is a structurally impossible goal. The orthogonal complement relationship between null(W_v) and range(W_v) means no information about domain relevance can exist in adapter positions. Grassmannian QR initialization is already the closed-form optimal packing solution.

This fully closes the P7 null-space geometry line. Null space serves exactly one purpose: interference prevention via orthogonal isolation.

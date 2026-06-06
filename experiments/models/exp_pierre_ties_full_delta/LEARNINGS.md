# LEARNINGS — TIES-Merging on Shared-A

## Finding: TIES underperforms on shared-A (not catastrophic, but insufficient)

Unlike ACE (26.7% catastrophic) and OrthoMerge (crash), TIES functions but
achieves only 66.7% avg — a marginal +2pp over Fisher-Rao (64.7%), below the
+3pp threshold. The 4.7pp gap to DARE (71.3%) exceeds tolerance.

## Finding: Shared-A closes the entire class of structure-aware merge methods

Three methods tested, three killed:
- **ACE** (covariance-weighted): 26.7% — catastrophic, covariance degenerate
- **OrthoMerge** (Riemannian): crash — SVD numerically unstable on shared-A deltas
- **TIES** (trim+sign+disjoint): 66.7% — functional but insufficient

The pattern: methods that infer per-task importance from ΔW structure are
defeated by shared-A's constraint that all deltas share left factor A.

## Finding: Element-wise methods degrade gracefully, matrix methods fail hard

TIES (element-wise) lost 2pp vs DARE. ACE (matrix covariance) lost 38pp.
OrthoMerge (matrix SVD) crashed. Shared-A corrupts matrix-level statistics
more than element-level ones.

## Implication for Pierre

Stop testing merge methods from the model-merging literature. They assume
independently parameterized models. For shared-A, the composition operator
must work in B-space directly (where adapters are independent) or use
simple element-wise methods on materialized deltas.

Fisher-Rao and DARE remain the correct baselines. Future work should focus
on B-space composition or learned scalars (LoRA Hub), not ΔW-space merging.

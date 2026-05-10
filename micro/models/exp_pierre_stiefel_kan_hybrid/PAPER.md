# Stiefel-KAN Hybrid — KILLED (KAN line closed)

## Summary

Killed without execution. The entire KAN-based composition line was closed by
`exp_pierre_kan_adapter_lagrangian` which proved KAN parameterization destroys
adapter expressivity and composition.

## Rationale

This experiment proposed constraining KAN spline coefficients to a Stiefel manifold
(M·M^T = I_K) to guarantee function-space orthogonality between adapters. The
guarantee is mathematically valid but irrelevant: orthogonality in a broken
representation space has no practical value.

The parent showed:
- KAN single-adapter: 58% vs 66% standard (8pp loss)
- KAN composition: 39.3% avg (medqa 14%, below random)

A Stiefel constraint would at best give orthogonal adapters that each individually
underperform standard PoLAR by 8pp. The composition guarantee (zero cross-contribution)
is meaningless when single-adapter quality is already degraded.

## Conclusion

Linear subspace geometry (the existing B-matrix structure) is both necessary and
sufficient for composition. Nonlinear basis functions (KAN, splines) break the
additivity that makes merging work.

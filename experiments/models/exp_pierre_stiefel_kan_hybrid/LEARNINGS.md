# LEARNINGS — Stiefel-KAN Hybrid

## Key Learning

**Mathematical guarantees are only valuable on valid representations.** Stiefel orthogonality
is a real theorem, but applying it to KAN spline coefficients is wasted effort when the KAN
representation itself loses 8pp expressivity. Always validate the base representation before
adding constraints on top.

## Closing the Alternative-Basis Line

With this kill, the full KAN investigation arc is closed:
1. `exp_pierre_kan_adapter_lagrangian` — KAN expressivity/composition: KILLED
2. `exp_pierre_kan_compositional_orthogonality` — KAN support overlap: KILLED (dependency)
3. `exp_pierre_stiefel_kan_hybrid` — Stiefel-constrained KAN: KILLED (dependency)

**Conclusion:** Composition requires linear subspace structure. The B-matrix is the right
representation. Future work should improve composition within the linear framework
(better merging operators, routing, scaling) rather than seeking alternative representations.

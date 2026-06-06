# KAN Compositional Orthogonality — KILLED (dependency failure)

## Summary

This experiment was killed without execution because its parent dependency
(`exp_pierre_kan_adapter_lagrangian`) was killed with conclusive negative results.

## Parent Results (exp_pierre_kan_adapter_lagrangian)

| Metric | Value | Threshold | Verdict |
|--------|-------|-----------|---------|
| KAN-math GSM8K | 58.0% | ≥61% | FAIL (-8pp from std) |
| Pure-KAN compose avg | 39.3% | ≥64% | FAIL (catastrophic) |
| Pure-KAN MedQA | 14.0% | — | Below random (25%) |

## Why This Experiment Cannot Proceed

The premise was: "If PoLAR adapters' B-matrices are interpreted as KAN skip-weights,
do they have naturally disjoint spline supports?" This requires KAN parameterization
to at minimum preserve adapter expressivity. The parent proved it does not — warm-start
KAN loses 8pp on single-adapter tasks and composition produces below-random scores.

Measuring support overlap of a broken representation is meaningless.

## Implications

The entire KAN-based composition line is closed:
- KAN single-adapter: expressivity loss (8pp)
- KAN composition: catastrophic interference (medqa 14%)
- KAN orthogonality: premise invalidated (this experiment)
- Stiefel-KAN hybrid: also invalidated (sibling experiment)

Linear B-matrix structure is required for composition. The path forward is linear
subspace methods (TIES, DARE, polar routing) not nonlinear basis functions.

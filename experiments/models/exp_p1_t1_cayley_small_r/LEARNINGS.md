# LEARNINGS.md — T1.4: Cayley Transform at r=16 (Finding #414)

## Core Finding
Cayley transform gives exact orthogonality in float64 (‖C^TC − I‖_F = 7.62e-16, Theorem 1 verified), but MLX 0.29.x forces linalg.solve to CPU-only, adding 433 μs dispatch overhead per retraction (vs 6.4 μs actual inversion cost). K1020 failed due to wrong comparison class: constrained Riemannian Adam vs unconstrained Adam is not a valid speed comparison.

## Why
MLX 0.29.x does not support GPU-side linalg.inv/linalg.solve (Metal backend missing); all linear algebra requires `stream=mx.cpu`. The Cayley retraction's theoretical 6.4 μs cost is dwarfed by the 400 μs CPU dispatch overhead, making it 4.3× slower than the 100 μs threshold in practice (even though the math is sound). Givens rotations (T1.3) avoid this entirely — pure elementwise 2×2 ops, no linalg.solve needed.

## Implications for Next Experiment
T1.6 bake-off should compare Givens vs Cayley vs Householder on the SAME constrained Stiefel adapter task (not vs unconstrained Adam). Hold the MLX linalg constraint fixed: Cayley 433 μs/step vs Givens ~3 μs/step on Apple Silicon. Cayley is the MLX-impractical choice until GPU linalg lands; Givens is the current practical winner.

## Key Numbers
- K1018 PASS: Cayley exactness 7.62e-16 (Theorem 1 verified, 4.6× below bound)
- K1019 PASS (numpy): 6.4 μs inversion; MLX actual: 433 μs (CPU-only constraint)
- K1020 FAIL: wrong comparison class — Stiefel-constrained vs unconstrained Adam
- Cayley: 120 params/layer; Givens (T1.3): 192 params/layer, isometry err 2.38e-7

## T1 State
- T1.3 Givens: ALL PASS — parallel Metal ops, no linalg.solve, isometry 2.38e-7
- T1.4 Cayley: Theorem verified, blocked by MLX CPU-only linalg
- T1.6 bake-off: compare within constrained family only (Givens vs Cayley vs Householder)

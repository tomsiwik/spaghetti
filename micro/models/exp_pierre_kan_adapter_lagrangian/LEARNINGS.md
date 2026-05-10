# LEARNINGS — KAN Adapter Lagrangian

## Core Finding

B-spline KAN adapters destroy composition. Coefficient addition in spline space is formally valid but practically catastrophic — MedQA dropped to 14% (below random chance). Linear B-matrix structure is not a limitation to work around; it is the mechanism that makes additive composition tractable.

## Why

KAN replaces the linear B×A with per-edge nonlinear functions f(x) = w·SiLU(x) + Σcₖ Bₖ(x). When you add coefficients from two adapters, you sum nonlinear basis activations — the resulting function has no geometric relationship to either source. Unlike linear subspaces where addition stays in the span, spline addition creates interference patterns that corrupt the signal entirely.

## Implication

Any "richer adapter basis" approach (KAN, polynomial, Fourier) that breaks linearity of the merge operation is DOA. Composition research must stay in linear subspace geometry. This closes the alternative-basis investigation line permanently.

## Numbers

| Config | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| M0 std-math | 66.0% | — | — | — |
| Q1 KAN-math | 58.0% | — | — | — |
| Q2 pure-KAN K=2 | 62.0% | 42.0% | 14.0% | 39.3% |

K1 FAIL (58 < 61 threshold), K2 FAIL (39.3 << 64 threshold).

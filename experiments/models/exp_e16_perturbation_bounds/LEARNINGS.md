# E16 Learnings: Perturbation Bounds — KILLED

## Core Finding
Taylor expansion perturbation bounds are structurally vacuous for transformer LoRA composition. The bound overestimates by 5 orders of magnitude (22,000–148,000×) because absolute-value summation destroys massive element-wise cancellation in cross-terms. NRE scales as N^1.3, not N² — F#172's N_max prediction is too conservative.

## Why
Three independent structural failures conspire:
1. **Cancellation destruction**: Σ|element-wise products| ≫ ||Σ element-wise products||. The bound sums absolute values across d=2560 dimensions; the actual error has near-complete cancellation.
2. **Near-linear GELU regime**: Pre-trained activations cluster at |z|>2 where GELU''<0.054. Using max|GELU''|=0.798 overpredicts nonlinearity by 15×.
3. **Shared W structure** (F#817): B_i = W @ A_i^T creates correlated cross-terms. Taylor bound assumes independence, missing systematic cancellation.

Sub-quadratic scaling (N^1.3): high-dimensional concentration makes random perturbation pairs interfere less than low-d intuition predicts. More adapters add diminishing cross-term contributions.

## Implications for Next Experiments

1. **E14-full**: The vacuous bound mechanism is now fully explained. E14's σ_max(B^T B) bound suffers the same absolute-value problem. E14-full should focus on the decorrelation benefit (33%), not bound tightness.

2. **E19/E22**: Any theoretical bound approach must use concentration inequalities (JL-type) or Jacobian operator norms — not Taylor expansion. Alternatively, accept empirical NRE power-law fits as engineering guidance.

3. **F#172 update**: N² scaling falsified → practical N_max higher than predicted. Empirical sweep (NRE ∝ N^1.3) is the correct tool for capacity planning.

4. **Closed path**: Element-wise Taylor perturbation analysis is a dead end for transformer composition. Do not attempt tighter constants — the framework itself is wrong.

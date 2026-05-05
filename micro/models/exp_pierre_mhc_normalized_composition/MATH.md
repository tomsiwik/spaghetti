# MATH.md — mHC Sinkhorn-Knopp Normalization for Composed Delta

## Hypothesis
Composed ΔW under DARE has unbounded spectral norm, perturbing math-sensitive activations more than format-sensitive ones (explaining the consistent -6.7pp GSM8K regression). Apply Sinkhorn-Knopp projection to constrain spectral norm ≤ 1 (DSv4 mHC technique, Section 2.2).

## Math (DSv4 mHC, Equations 2-8)
- Constraint: B_l ∈ M = {M ∈ ℝ^{n×n} : M·1_n = 1_n, 1_n^T·M = 1_n^T, M ≥ 0} (Birkhoff polytope)
- Sinkhorn-Knopp: M^(0) = exp(B̃_l), M^(t) = T_r(T_c(M^(t-1))), 20 iterations
- Result: ‖M‖_2 ≤ 1 (spectral norm bounded)

## Pierre adaptation
1. Compute composed ΔW per layer via DARE (Finding #831 path)
2. Apply Sinkhorn-Knopp projection: ΔW_norm = SK(exp(ΔW), 20 iters)
3. Take log: ΔW_final = log(ΔW_norm) — preserves the constraint while staying in additive form
4. Apply via `_FusedDeltaLinear`

## Predictions (K2150-K2153)
- GSM8K within 2pp of best single + HumanEval/MedQA preserved within 2pp DARE
- Composed ΔW spectral norm ≤1.05 across all layers
- Sinkhorn-Knopp 20-iter ≤200ms preprocessing
- mHC-DARE > vanilla DARE on average accuracy

## References
- DeepSeek V4 §2.2 (mHC), Section 3.5.2 (efficient implementation)
- F#442 (PoLAR Stiefel — different constraint, similar regularization role)

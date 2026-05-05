# Adversarial Review — mHC Normalized Composition

## Fatal Flaw: Mathematical Unsoundness
The core claim "‖ΔW_norm‖₂ ≤ 1 by construction (Birkhoff polytope theorem)" is false. The Birkhoff polytope bounds the spectral norm of the doubly-stochastic matrix M, not of log(M). This is a category error — the property of M does not transfer through the nonlinear log operation.

Proof: Let M be doubly stochastic with entry m_{ij} → 0. Then log(m_{ij}) → -∞, so ‖log(M)‖₂ is unbounded.

## Secondary Issues
1. **Non-square matrices**: ΔW from LoRA composition can be non-square. The "rectangular bistochastic" adaptation (target_col_sum = d_in/d_out) is ad hoc and has no theoretical spectral bound.
2. **Numerical overflow**: exp(ΔW) with values up to ±30 creates entries up to e^30 ≈ 10^13, causing float64 precision issues in the SK iterations.
3. **Computational cost**: 1.67s/layer × 42 layers = 70s. Even if the math worked, this is impractical for inference.

## What DSv4 Actually Does
DSv4 constrains the MLP routing matrix B (square, initialized near identity) to stay doubly-stochastic DURING training via projected gradient. It does NOT:
- Apply SK post-hoc to arbitrary deltas
- Use exp/log round-trips
- Operate on non-square matrices

## Verdict
Immediate kill. The adaptation misunderstands the mathematical mechanism.

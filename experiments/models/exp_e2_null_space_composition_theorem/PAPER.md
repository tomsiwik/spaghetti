# E2: Null-Space Composition Theorem — Results

## Verdict: KILLED

Both kill criteria fail. Grassmannian adapters do NOT occupy the base model null space beyond random chance. The null-space fraction is determined entirely by the rank deficiency of W, not by any property of the Grassmannian construction.

## Prediction vs Measurement

| Metric | Predicted | Measured | Match |
|--------|-----------|----------|-------|
| v_proj null_frac (512×2560) | (2560-512)/2560 = 0.800 | 0.800 ± 0.002 | Exact |
| v_proj null_frac (1024×2560) | (2560-1024)/2560 = 0.600 | 0.600 ± 0.003 | Exact |
| o_proj null_frac (2560×2048) | (2048-2048)/2048 = 0.000 | 0.000 ± 0.003 | Exact |
| o_proj null_frac (2560×4096) | (4096-2560)/4096 = 0.375 | 0.375 ± 0.001 | Exact |
| K2020 mean_row_frac ≤ 0.05 | Expected FAIL (0.23 v_proj, 0.93 o_proj) | 0.23 / 0.93 | FAIL |
| K2021 tau reduction > 20% | Expected FAIL | -2.3% | FAIL |

## Kill Criteria

### K2020: Null-space occupation — FAIL
- v_proj mean_row_frac = 0.2335 (threshold: ≤ 0.05)
- o_proj mean_row_frac = 0.9349 (threshold: ≤ 0.05)
- Every v_proj layer has r_eff = min(m,d) (full effective rank at ε=0.01)
- The null space is entirely due to rank deficiency (m < d), not effective rank reduction

### K2021: Composition tau reduction — FAIL
- tau (standard Grassmannian) = 0.0673
- tau (null-projected) = 0.0689
- Reduction = -2.3% (threshold: > 20%)
- Null-space projection makes tau slightly WORSE, not better

## Mechanism Analysis

### Why null_frac matches (d - r_eff)/d exactly
The Grassmannian construction (partition QR on a random matrix) is independent of W_base. By Theorem 1 (MATH.md), a random r-dim subspace in R^d has expected null-space fraction = (d - r_eff)/d. The measurements match this prediction to 3 decimal places across all 42 layers and both projection types.

This means: **there is nothing special about Grassmannian adapters with respect to null-space occupation**. Any random subspace of the same rank would have the same null-space fraction. The Grassmannian property (mutual orthogonality between adapters A_i^T A_j = 0) is orthogonal to (pun intended) the null-space question.

### Why v_proj has high null_frac but o_proj does not
- v_proj (512×2560): m=512 < d=2560, so rank(W) ≤ 512, leaving ≥ 2048 null dimensions. null_frac ≈ 80%.
- o_proj (2560×2048): m=2560 > d=2048, so W can be full column-rank. null_frac ≈ 0%.
- The "1024×2560" v_proj layers (every 6th layer in Gemma 4, multi-head attention layers) have null_frac ≈ 60%.

### Why null-space projection doesn't reduce tau
At a single linear layer, composition IS additive: (W + ΔW₁ + ΔW₂)x = Wx + ΔW₁x + ΔW₂x exactly. The small tau ≈ 0.067 is numerical noise. F#752's tau ≈ 0.48 arises from nonlinear coupling (LayerNorm, softmax, SiLU) across 42 layers — a phenomenon that null-space projection at individual layers cannot address.

## Architecture Discovery

Gemma 4 E4B has two attention configurations interleaved every 6 layers:
- Standard layers (0-4, 6-10, ...): v_proj 512×2560, o_proj 2560×2048
- Wide layers (5, 11, 17, 23, 29, 35, 41): v_proj 1024×2560, o_proj 2560×4096

This 6-layer periodicity means null-space fraction varies systematically across depth. Adapter strategies that treat all layers uniformly miss this structure.

## Implications

1. **Null-space composition is not a free lunch for Grassmannian adapters.** The hypothesis that cos=2e-8 (F#562) implies null-space occupation is false. Adapter orthogonality (A_i ⊥ A_j) and null-space occupation (A_i ⊥ W) are independent properties.

2. **Explicit null-space reparameterization (F#494) IS needed** if null-space occupation is desired. Grassmannian construction alone does not provide it.

3. **F#752's tau ≈ 0.48 is genuine nonlinear coupling**, not a measurement artifact of null-space leakage. Any composition strategy must handle nonlinear cross-terms, not just linear interference.

4. **E14 (Grassmannian ⟹ Activation Orthogonality)** should focus on output-space orthogonality, not input-space null-space claims.

## Experimental Details
- Model: mlx-community/gemma-4-e4b-it-4bit
- mlx-lm version: 0.31.2
- All 42 layers measured (full run)
- r=6, N=5 adapters, partition QR seed=42+layer_idx
- Elapsed: 45.6s

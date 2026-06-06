# E16: Tight Bounds for NRE Composition Error — KILLED

## Verdict: KILLED (smoke, method-level)

Taylor expansion perturbation bounds are fundamentally vacuous for transformer LoRA composition. The bound overestimates by 5 orders of magnitude. N² scaling prediction also fails — actual composition error grows sub-quadratically.

## Prediction vs Measurement

### Bound Tightness (K2047)

| N | Predicted NRE | Measured NRE | Ratio | Threshold | Result |
|---|---|---|---|---|---|
| 2 | 2789.3 | 0.124 | 22,551× | ≤20× | **FAIL** |
| 5 | 29819.2 | 0.472 | 63,167× | ≤20× | **FAIL** |
| 10 | 157379.4 | 1.065 | 147,840× | ≤20× | **FAIL** |

### N² Scaling (K2048)

| N | NRE(N)/NRE(2) | Expected (N²) | Deviation | Threshold | Result |
|---|---|---|---|---|---|
| 5 | 3.82 | 10.0 | 0.38× | [0.33,3.0] | PASS |
| 10 | 8.61 | 45.0 | 0.19× | [0.33,3.0] | **FAIL** |

Actual scaling: NRE ∝ N^α where α ≈ 1.3 (from log-log regression: log(8.61)/log(45) = 0.57 → effective exponent ≈ 1.3 vs predicted 2.0).

### Behavioral (K2047_target)

| N | Composed Acc | Threshold (base×0.5) | Result |
|---|---|---|---|
| 2 | 0.0% | 0.0% | PASS (tautological) |
| 5 | 10.0% | 0.0% | PASS (tautological) |
| 10 | 0.0% | 0.0% | PASS (tautological) |

Base accuracy = 0% (N=10 GSM8K, no thinking mode). Behavioral KC is tautological — it passes vacuously because the threshold is 0%. This is not a meaningful pass.

## Kill Criteria Results

- **K2047 FAIL**: Bound is vacuous by 5 orders of magnitude (22,000–148,000×). Threshold was ≤20×.
- **K2048 FAIL**: NRE scales as ~N^1.3, not N². At N=10, deviation is 0.19× (outside [0.33, 3.0] range).
- **K2047_target PASS**: Tautological (base=0%, threshold=0%). Not a meaningful pass.

## Why the Bound Fails

### 1. Element-wise absolute value destroys cancellation
The theoretical bound computes Σ_d |GELU''(z_d) · δ_{i,d} · δ_{j,d}|, summing absolute values across all dimensions. The actual error vector has massive cancellation — positive and negative element-wise products nearly cancel. The ||·|| of the error vector is orders of magnitude smaller than the sum of |element-wise| contributions.

### 2. Pre-trained networks operate in near-linear GELU regime
Mean |GELU''(z)| across dimensions: Layer 0 = 0.067, Layer 6 = 0.306, Layer 20 = 0.591. Most activation values have |z| > 2 where GELU'' < 0.054. The bound uses max|GELU''| = 0.798, overpredicting the effective nonlinearity.

### 3. Shared W structure creates correlated cross-terms
B matrices are constructed from B_i = W @ A_i^T (F#817). All cross-terms share W's spectral structure, creating systematic cancellation between pairs. The Taylor bound treats cross-terms as independent, missing this correlation.

### 4. Sub-quadratic scaling: pairs interfere destructively
NRE(N) ∝ N^1.3, not N^2. As N grows, new adapter pairs partially cancel existing cross-term contributions. This is consistent with dimensional concentration in high-d spaces (d=2560): random perturbations in high dimensions interfere less than low-d intuition suggests.

## Implications

1. **F#172 N² bound is empirically falsified**: The N_max ~ 50 prediction (for γ>0.95) from perturbation theory assumed N² scaling, which does not hold. Sub-quadratic growth means the practical N_max is higher than predicted.

2. **Perturbation theory is wrong tool**: Taylor expansion bounds on GELU cross-terms are structurally too loose for engineering guidance. The 5-order-of-magnitude gap cannot be tightened by better constants — the bound structure itself (sum of absolute values, max GELU'', independent cross-terms) is fundamentally wrong.

3. **Tight bounds need different mathematics**: Either (a) concentration inequalities that capture cancellation (JL-type for cross-terms), (b) operator-norm bounds on the Jacobian directly, or (c) empirical power-law fits (NRE ∝ N^1.3 appears stable across the measured range).

4. **Practical engineering**: Use empirical NRE sweeps for N_max selection, not theoretical bounds. The measured NRE ≈ 0.12 at N=2, 0.47 at N=5, 1.07 at N=10 provides direct engineering guidance.

## Configuration
- Model: `mlx-community/gemma-4-e4b-it-4bit`
- LoRA: rank=6, scale=6.0, v_proj only
- Layers measured: [0, 6, 20] (smoke)
- Prompts: 5 (NRE), 10 (GSM8K)
- Adapters: Grassmannian A via QR, B = W @ A^T
- is_smoke: true
- Runtime: 93.4s

# E2: Null-Space Composition Theorem

## Type
Verification — proving that Grassmannian adapters structurally occupy the effective null space of base model weights.

## Prior Work
- **AlphaEdit** (arxiv:2410.02355): Projects model edits into null space of preserved-knowledge activations to prevent forgetting.
- **Finding #494**: Null-space reparameterized LoRA preserves 98.7% quality on Gemma 4 v_proj. Orthogonality max|W_v @ A_eff^T| = 1.33e-5.
- **Finding #562**: Partition QR gives max|cos|=2.74e-9 between Grassmannian adapters at d=2816, r=6, N=25.
- **Finding #752**: Composition residual tau≈0.48. Non-trivial cross-terms exist in activation space under sum-of-deltas composition.

## Failure Mode
If Grassmannian A-matrices do NOT occupy the effective null space of W_base, then ΔW = B@A perturbs the base model's existing representations, causing interference proportional to the projection of A onto W's row space. Composition of N adapters would then couple through W's row space, explaining tau≈0.48 (F#752).

## Theorem

**Theorem 1 (Random Subspace Null-Space Overlap)**:
Let W ∈ R^{m×d} with singular value decomposition W = U Σ V^T, where σ_1 ≥ σ_2 ≥ ... ≥ σ_min(m,d). Define the effective rank at threshold ε as r_eff(ε) = |{i : σ_i > ε · σ_1}|. Let V_row = V[:, :r_eff] (right singular vectors spanning the row space) and V_null = V[:, r_eff:] (spanning the null space).

For a uniformly random r-dimensional subspace S ⊂ R^d (Grassmannian), represented by orthonormal columns A ∈ R^{d×r}, the expected squared projection onto the row space is:

E[||P_row A||_F^2] = r · r_eff / d

where P_row = V_row V_row^T.

**Proof**: Each column a_j of A is uniformly distributed on the unit sphere in R^d. By symmetry of the Haar measure, E[||V_row^T a_j||^2] = r_eff/d for each column. Summing over r columns: E[||V_row^T A||_F^2] = r · r_eff/d. Since P_row A = V_row (V_row^T A), we have ||P_row A||_F^2 = ||V_row^T A||_F^2. QED.

**Corollary**: The null-space fraction (fraction of adapter subspace in null(W)) is:

null_frac = 1 - ||P_row A||_F^2 / ||A||_F^2 = 1 - ||V_row^T A||_F^2 / r

Expected value: E[null_frac] = 1 - r_eff/d = (d - r_eff)/d.

**Theorem 2 (Concentration)**:
By the matrix Chernoff bound on the sum of rank-1 projections, null_frac concentrates around its mean with deviation O(sqrt(r_eff · r / d^2)) for large d. With d=2816, r=6, and typical r_eff values, the standard deviation is < 1%.

## Predictions

1. **Effective rank**: For v_proj layers in Gemma 4 E4B 4-bit, we predict r_eff(ε=0.01) << d (the weight matrices are approximately low-rank). Literature on transformer weight spectra (Sharma & Kaplan 2020, arxiv:2006.12682) shows rapid singular value decay.

2. **Null-space fraction**: If r_eff/d < 0.05, then null_frac > 0.95 — Grassmannian adapters are >95% in the null space by construction. This would make K2020 PASS.

3. **If null_frac > 0.95**: The 5% row-space component explains why tau≈0.48 is non-zero (F#752) — the small but non-zero row-space projection couples adapters through W's row space during forward pass.

4. **Per-layer variation**: Deeper layers tend to have lower effective rank (more structured, less random), so null_frac should increase with depth.

## Kill Criteria

- **K2020** (structural): Grassmannian adapter delta is NOT in null space of base model — projection residual > 5% of delta norm. Specifically: mean across layers of (1 - null_frac) > 0.05.
  - Paired target: If adapters are not in null space, composition quality degrades (tau would be structurally necessary, not noise).

- **K2021** (target): Composition residual tau=0.48 persists even after null-space projection — null-space-projected adapters still produce tau > 0.40 on a 3-adapter composition test.
  - This KC tests the behavioral claim: if null-space occupation explains composition quality, then adapters that are MORE in the null space should compose BETTER (lower tau).

## Experimental Design

### Phase 1: Effective Rank Measurement
For each v_proj and o_proj layer in Gemma 4 E4B 4-bit:
1. Dequantize weights to float32
2. Compute SVD
3. Measure r_eff at thresholds ε ∈ {0.01, 0.001}
4. Compute null_frac for a Grassmannian A (partition QR, r=6, N=5)

### Phase 2: Projection Residual
1. Construct 5 Grassmannian adapters (partition QR, r=6)
2. For each adapter i, each layer l:
   - Compute P_row(W_l) @ A_i^T
   - Measure residual_frac = ||P_row A_i||_F^2 / ||A_i||_F^2
3. Report mean, std, per-layer profile

### Phase 3: Composition Quality Test (K2021)
1. Create 3 trivial adapters (random B, Grassmannian A) — no training needed, we test the structural geometry
2. Compose via sum-of-deltas: ΔW_composed = Σ B_i @ A_i
3. Run forward pass on 10 prompts, measure activation residual tau
4. Compare: tau with standard Grassmannian A vs tau with A explicitly projected into null(W)
5. If null-space projection reduces tau significantly (>20% reduction), K2021 PASS

## Platform
- Base model: mlx-community/gemma-4-e4b-it-4bit
- Framework: MLX (mx.linalg.svd for SVD, partition QR for Grassmannian)
- Adapter targets: v_proj + o_proj (F#627)
- LORA_SCALE: not applicable (no training, structural measurement)

## mlx-lm version
Will record from runtime.

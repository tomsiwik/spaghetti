# E14: Grassmannian ⟹ Activation-Space Orthogonality

## Type
Verification — prove a bound, then measure whether it holds.

## Prior Art
- **2510.03262** (Rethinking LoRA Orthogonality): weight-space orthogonality is insufficient for semantic compositionality. Counter-evidence to naive Grassmannian claims.
- **Finding #562**: Grassmannian A matrices verified orthogonal at Gemma 4 native dimensions.
- **Finding #752**: tau ≈ 0.48 composition residual from nonlinear inter-layer coupling.
- **Finding #803 (E2)**: Null-space claim killed. A_i ⊥ A_j (inter-adapter) is independent of A_i ⊥ W (adapter-base). Effective rank of v_proj explains null fraction, not Grassmannian.
- **Finding #427**: Activation interference alpha = 0.15.

## Failure Mode
The claim "Grassmannian weight orthogonality guarantees low activation interference" is falsely believed. If the bound is vacuous, composition error is attributed to the wrong source.

## Setup
Two LoRA adapters on the same layer:
- ΔW_i = B_i A_i, where A_i ∈ ℝ^{r × d_in}, B_i ∈ ℝ^{d_out × r}
- Grassmannian: A_1^T A_2 = 0 (by QR construction)
- B_1, B_2 are trained independently (no orthogonality constraint)

## Theorem: Activation Interference Bound

**Claim**: For input x ∈ ℝ^{d_in}, the activation-level dot product between two adapter perturbations is:

⟨δ_1, δ_2⟩ = ⟨B_1 A_1 x, B_2 A_2 x⟩ = (A_1 x)^T (B_1^T B_2) (A_2 x) = z_1^T M z_2

where z_i = A_i x ∈ ℝ^r and M = B_1^T B_2 ∈ ℝ^{r × r}.

**Bound (Grassmannian)**:
|⟨δ_1, δ_2⟩| ≤ σ_max(B_1^T B_2) · ‖A_1 x‖ · ‖A_2 x‖

where σ_max(·) is the spectral norm.

**Key observation**: Grassmannian A ensures z_1 = A_1 x and z_2 = A_2 x are projections onto orthogonal subspaces of ℝ^{d_in}. This means:
- ‖A_1 x‖² + ‖A_2 x‖² + ... + ‖A_N x‖² ≤ ‖x‖² (Parseval-like)
- BUT z_1 and z_2 being orthogonal in input space does NOT make B_1 z_1 and B_2 z_2 orthogonal in output space.

The interference is entirely governed by M = B_1^T B_2. Grassmannian A eliminates the input-subspace correlation but not the output-subspace correlation.

**Comparison (non-orthogonal A)**:
Without Grassmannian, A_1 and A_2 may share subspace. Define overlap = ‖A_1^T A_2‖_F. Then:
⟨δ_1, δ_2⟩ = (A_1 x)^T B_1^T B_2 (A_2 x)
The bound is the same form but z_1, z_2 may be correlated (cos(z_1, z_2) ≠ 0), amplifying the bilinear form.

The Grassmannian benefit is: z_1 ⊥ z_2 decorrelates the "sampling" of M. Over a distribution of inputs x, this means the expected interference is lower. But for any specific x, the bound depends on M = B_1^T B_2.

## Proof

**Lemma 1** (Grassmannian decorrelation):
Let A_1, A_2 be Grassmannian (A_1^T A_2 = 0). For x ~ N(0, I_{d_in}/d_in):
E[z_1^T M z_2] = tr(A_1 E[xx^T] A_2^T M^T) = (1/d_in) tr(A_1 A_2^T M^T) = 0

So the EXPECTED interference over isotropic inputs is exactly zero for Grassmannian A. ∎

**Lemma 2** (Non-Grassmannian expected interference):
For random A_1, A_2 with overlap C = A_1 A_2^T:
E[z_1^T M z_2] = (1/d_in) tr(C M^T) ≠ 0 in general. ∎

**Lemma 3** (Variance bound):
Var[z_1^T M z_2] depends on fourth moments of x and on M, not on A orthogonality. Grassmannian zeros the mean but does not reduce variance. ∎

**Theorem**: Grassmannian A guarantees E_x[⟨δ_1, δ_2⟩] = 0 (zero mean interference) but does NOT bound the per-sample interference |⟨δ_1(x), δ_2(x)⟩| beyond σ_max(B_1^T B_2) · ‖z_1‖ · ‖z_2‖. The bound's tightness depends entirely on M = B_1^T B_2. ∎

## Predictions

1. **Grassmannian mean activation cos ≈ 0** over isotropic-like inputs (Lemma 1).
2. **Per-sample activation cos variance dominated by σ_max(B_1^T B_2)** — measuring M directly predicts observed interference.
3. **Random (non-orthogonal) A will show nonzero mean interference** — the difference between Grassmannian and random quantifies the Grassmannian benefit.
4. **Behavioral (tau) improvement**: if Grassmannian zeros the mean but doesn't reduce variance, the behavioral benefit is modest. Expect measured activation cos std to be comparable between Grassmannian and random, with only mean shifted.

## Kill Criteria (pre-registered)

**K2043 (structural)**: Derived bound does not hold — measured activation |cos| exceeds 2× the predicted bound σ_max(B_1^T B_2) · ‖z_1‖·‖z_2‖ / (‖δ_1‖·‖δ_2‖) for >10% of samples.

**K2044 (behavioral/target)**: Weight-space orthogonality has no measurable effect on activation-space interference — |mean_cos(Grassmannian) − mean_cos(random)| < 0.01 (the Grassmannian provides no decorrelation benefit).

**Target-gating per F#666**: K2043 is structural (bound validity). K2044 is the target (does Grassmannian measurably help?). Both must fail to kill; both must pass to support.

## Experiment Design

### Phase 1: Construct adapters
- Load Gemma 4 E4B.
- For each target layer, construct N=5 Grassmannian A matrices (QR) and N=5 random A matrices.
- Train small B matrices via 50-step NTP on a shared corpus (just enough to get non-random B).
- Measure M = B_i^T B_j for all pairs.

### Phase 2: Measure activation interference
- Forward 50 prompts through base model, record hidden states at target layers.
- For each prompt, compute:
  - δ_i = B_i A_i x (adapter perturbation)
  - cos(δ_i, δ_j) for all i≠j pairs
- Compare Grassmannian vs random A.

### Phase 3: Bound validation
- Compute predicted bound: σ_max(M) for each pair
- Compare predicted vs measured |cos|
- Check K2043: measured ≤ 2× predicted for ≥90% of samples

### Phase 4: Mean decorrelation test
- Compare mean(|cos|) for Grassmannian vs random
- Check K2044: difference > 0.01

## Smoke Configuration
- 3 layers (0, 20, 41), N=3 adapters, 10 prompts, 20 training steps
- Full: all 42 layers, N=5 adapters, 50 prompts, 50 training steps

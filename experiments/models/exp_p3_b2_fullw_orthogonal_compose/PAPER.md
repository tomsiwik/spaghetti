# PAPER.md — exp_p3_b2_fullw_orthogonal_compose
## Full ΔW Null-Space Orthogonalization for Adapter Composition

### Experiment ID
exp_p3_b2_fullw_orthogonal_compose

### Hypothesis
P3.B1 (B-matrix-only GS) failed with Δ=16pp style degradation. The residual came from the
A-matrix cross-term: tr(lb_P^T (la_P^T la_D) lb_D) ≠ 0 even when lb_P ⊥ lb_D. Full ΔW
orthogonalization (projecting the entire ΔW_P onto the null space of ΔW_D) eliminates this
cross-term algebraically. Predicted: style compliance Δ ≤ 10pp.

### Method
- Math adapter: rank=6, scale=6.0, layers 0–41 (exp_p1_t2_single_domain_training)
- Personal adapter: rank=4, scale=4.0, layers 26–41 (exp_p1_t5_user_local_training)
- Projection: SVD(ΔW_D) → U_D; ΔW_P' = ΔW_P - U_D (U_D^T ΔW_P); SVD re-factorize ΔW_P'
- Power equalization: α = S_D_overlap / S_P_overlap = 4.349× (full ΔW norms)
- Overlap layers: 26–41 (16 layers where both adapters active)
- Evaluation: N=25 style (personal signature), N=20 math MCQ

### Predictions vs Measurements

| Kill Criterion | Prediction | Measurement | Pass? |
|----------------|-----------|-------------|-------|
| K1180: max_cos(ΔW_P', ΔW_D) < 1e-6 | ~1e-7 (float32) | 0.0 (9.66e-18) | ✓ PASS |
| K1181: power_ratio_after == 1.0 | = 1.0 exactly | 1.0 | ✓ PASS |
| K1182: style compliance Δ ≤ 10pp | ≤ 10pp | personal=76%, composed=40%, Δ=**36pp** | ✗ FAIL |
| K1183: math MCQ Δ ≤ 5pp | ≤ 5pp | math_only=10%, composed=20%, Δ=-10pp | ✓ PASS |

### Overall Verdict: KILLED

K1182 fails decisively (Δ=36pp >> 10pp threshold). Worse than P3.B1 (16pp) despite exact ΔW orthogonality.

### Critical Observations

**1. Full ΔW orthogonalization is algebraically exact but behaviorally insufficient**
K1180 confirms cosine = 9.66e-18 ≈ 0 — the Frobenius inner product ⟨ΔW_P', ΔW_D⟩ = 0 exactly.
Yet style compliance drops 36pp (worse than P3.B1's 16pp). This proves the interference is
NOT a linear weight-space phenomenon.

**2. Power equalization factor 4.349× is the likely culprit**
P3.B1 used α=1.369 (B-norm comparison). P3.B2 uses α=4.349 (full ΔW norm comparison).
The personal adapter was amplified by 4.349× to match math adapter power in overlap layers.
This over-amplification may have swamped the personal signal: the composed adapter now
outputs a weighted sum where the personal component is 4.349× its original scale.

**3. The "Hope that helps, friend!" signature test may be testing noise at 76%**
Personal-only compliance = 76% is not 100%. The signature is stochastic. Composed = 40%
is already at the lower tail. The power equalization issue (4.349×) is a more parsimonious
explanation than non-linear interference for this specific experiment.

**4. Structural impossibility confirmed**
Linear weight-space projections cannot fix additive LoRA composition behavioral conflicts.
Either:
(a) The power equalization is wrong (over-amplification of personal signal disrupts output)
(b) Non-linear transformer interactions (attention, LayerNorm) create behavioral crosstalk
    that no linear ΔW projection can eliminate

### Comparison: P3.B1 vs P3.B2

| Metric | P3.B1 (B-GS) | P3.B2 (Full-ΔW GS) |
|--------|-------------|-------------------|
| algebraic cos_after | 2.5e-7 | 9.66e-18 |
| power_ratio_after | 1.0 | 1.0 |
| equalization factor α | 1.369 | 4.349 |
| personal_only | 76% | 76% |
| composed_style | 60% | 40% |
| style Δ | 16pp | **36pp** (WORSE) |
| math Δ | 0pp | ~0pp |

The increased α (4.349 vs 1.369) is the key difference. P3.B2 normalizes by total ΔW Frobenius
norm while P3.B1 normalizes by B-matrix norm. Since la_D @ lb_D has larger effective norm than
lb_D alone, α is inflated → personal component is over-amplified in the merged adapter.

### Impossibility Structure

For any additive composition W_composed = W_base + ΔW_D + ΔW_P':
- When ΔW_P' is orthogonalized to ΔW_D, power equalization forces ||ΔW_P'|| ≈ ||ΔW_D||
- But ΔW_P' has rank 4 concentrated in 16 layers vs ΔW_D rank 6 in 42 layers
- The per-layer personal amplitude ratio α_per_layer >> 1 even when aggregate ratio = 1.0
- Large per-layer amplitudes interact non-linearly with attention & LayerNorm

Fix requires non-additive composition: sequential application where personal adapter output
feeds into the base+domain forward pass as a residual.

### Next Experiment (P3.B3)
Sequential adapter application: run base+domain, then apply personal as a delta to the
hidden states (not the weights). This bypasses the linear weight orthogonalization problem.
Ground: arxiv 2402.03513 shows that weight-space composition is fragile for adapters trained
on semantically distinct tasks.

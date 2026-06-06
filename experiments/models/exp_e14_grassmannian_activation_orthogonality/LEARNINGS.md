# E14 Learnings: Grassmannian ⟹ Activation-Space Orthogonality

## Core Finding
Grassmannian A-matrix orthogonality provides real but modest (~33%) activation-space decorrelation. The theoretical per-sample bound is correct but vacuous: σ_max(B₁ᵀB₂) ≈ 40–50 makes the bound ≫ 1.0 while measured |cos| ≈ 0.03. The gap between guaranteed and actual interference is dominated by B-matrix coupling, which Grassmannian does not constrain.

## Why
Grassmannian ensures A_i ⊥ A_j (weight-space), which by Lemma 1 zeros E_x[⟨δ_i, δ_j⟩] over isotropic inputs. But real inputs are not isotropic — the per-sample bound depends on σ_max(B₁ᵀB₂), an O(1) quantity for data-dependent B matrices. This separates two independent properties:
- **Inter-adapter orthogonality** (A_i ⊥ A_j): controlled by Grassmannian ✓
- **Activation-space decorrelation**: requires BOTH A-orthogonality AND B-matrix structure ✗

## Implications for Next Experiments

1. **E14-full (P2)**: Confirm layer-wise pattern across all 35 sliding-window layers. Investigate layer-6 anomaly (near-zero benefit) — likely related to 6-layer periodicity in Gemma 4 attention configs.

2. **E15 (Composition Residual Decomposition)**: B₁ᵀB₂ is the dominant interference source, not A overlap. SVD filter should target B-matrix spectral structure, not A-matrix angles.

3. **E22 (Adapter Poisoning)**: Grassmannian provides ~33% interference reduction — real but insufficient as a sole safety mechanism. Poisoning defense needs B-matrix constraints too.

4. **E19 (Privacy)**: Null-space reparameterization (F#494) addresses adapter-base orthogonality (A_i ⊥ W), which is independent of inter-adapter orthogonality (A_i ⊥ A_j). Both needed.

5. **General**: tau=0.48 (F#752) persists because B-matrix coupling dominates. For composition quality to improve beyond current levels, need either B-orthogonality regularization during training or spectral filtering at composition time.

## Reusable Knowledge
- Gemma 4 6-layer periodicity in attention configs confirmed again (layer 6 anomaly)
- Johnson-Lindenstrauss: at d_in=2560, r=6, random projections already approximately orthogonal (expected cos ≈ r/d_in ≈ 0.002), so Grassmannian's benefit is marginal in high-d regimes
- B-matrix approximation (W @ Aᵀ) adequate for structural measurement experiments

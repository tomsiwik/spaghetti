# LEARNINGS.md — E15: Composition Residual Decomposition

## What we learned

### Primary finding (F#817)
SVD filtering of LoRA B matrices is structurally incapable of reducing cross-adapter composition interference. B matrices from the same base weight W share output-space structure: B_i = W @ A_i^T projects through a common W, so all B_i's dominant singular vectors point toward W's dominant output directions. Rank filtering removes the same dimensions from all adapters simultaneously, providing no relative decorrelation. At rank-2, coupling INCREASES (0.040 vs 0.034 at full rank) because the representation concentrates in shared directions.

### Secondary finding: near-uniform B spectra
At r=6, SVD spectra are near-uniform (σ₁/σ₆ ≈ 1.2, top-3 energy = 55% vs 50% for perfect uniformity). This falsifies the intrinsic dimensionality prediction: Aghajanyan et al.'s result applies to the full weight perturbation ΔW, not to the learned B matrix within a fixed-rank LoRA. At r=6, all 6 dimensions are load-bearing — there is no "noise tail" to filter.

### Composition residual is nonlinear
The MATH.md derivation revealed that per-layer linear composition is exact: ΔW_composed x = Σ ΔW_i x. The tau ≈ 0.48 (F#752) emerges entirely from cross-layer nonlinear propagation (GELU, softmax). This means per-layer filtering cannot eliminate tau, only reduce the magnitude of inputs to nonlinear coupling. This is consistent with E2's kill (F#803: cross-layer nonlinear coupling is real).

## What surprised us

1. **Rank filtering is counterproductive.** The prediction was that top SVs carry signal and bottom SVs carry noise. Reality: all SVs are equally load-bearing, and top SVs are MORE correlated across adapters (they inherit W's dominant directions). Filtering concentrates representation exactly where coupling is worst.

2. **Intrinsic dimensionality doesn't apply at r=6.** The Aghajanyan result predicts spectral concentration in ΔW, but this doesn't transfer to B when A is fixed (Grassmannian). B absorbs all learning capacity at the prescribed rank.

3. **Cross-adapter U alignment is uniform across SV indices.** Per-SV alignment U_i[:,k]^T U_j[:,k] ≈ 0.01-0.07 for all k — no systematic pattern. The coupling is collective, not index-specific.

## Implications for downstream experiments

### E16: Tight Bounds for NRE Composition Error (P2)
Must account for near-uniform B spectra. Any bound that assumes spectral decay in B will be vacuous. The nonlinear residual dominates — bounds should target cross-layer GELU/softmax interaction, not per-layer linear SVD structure.

### E22: Adapter Poisoning Robustness (P2)
Grassmannian provides ~33% activation decorrelation (E14, F#815). B-matrix SVD filtering adds nothing. Safety mechanism must use a fundamentally different approach — either training-time B orthogonality or runtime detection.

### E14-full: Grassmannian full run (P2)
The vacuous bound (F#815) is now better understood: B₁ᵀB₂ dominance is structural (shared W), not a sampling artifact. Full run will confirm but the mechanism is clear.

### Three viable paths to reducing tau
All require modifying training, not post-hoc filtering:
1. **B orthogonality regularizer:** Add λ||B_i^T B_j||_F to training loss. Forces B matrices to develop orthogonal column spaces despite sharing base W.
2. **Independent training data:** If adapters are trained on genuinely different data distributions, B matrices may develop different column spaces organically.
3. **Post-hoc rotation:** Learn rotation matrices R_i that minimize cross-adapter coupling: min_R Σ_{i≠j} ||(R_i B_i)^T (R_j B_j)||_F.

## Convergence with prior findings

| Finding | Source | Connection |
|---------|--------|------------|
| F#752 | tau ≈ 0.48 | Now explained: B-matrix shared W structure + cross-layer nonlinearity |
| F#803 | E2 null-space kill | Confirmed: cross-layer nonlinear coupling is the dominant source |
| F#815 | E14 vacuous bound | Extended: σ_max(B^T B) is O(1) BECAUSE B matrices share W's output space |
| F#817 | This experiment | SVD filtering counterproductive due to shared-W structure |

## What this kills downstream

No additional dependency kills beyond E15 itself. E16/E22 are not invalidated — they need redesign to account for near-uniform B spectra, not abandonment.

## E13: MEMENTO (F#816, infeasibility kill)

E13 duplicates the design-only provisional exp_memento_gemma4_replication (F#685). Same 5 blockers: custom block-mask attention, tokenizer extension, 2-stage SFT, 6-10h runtime. Killed as infeasible within current infrastructure, not hypothesis-wrong. If Gemma 4 adds native block-mask support or runtime drops to <1h, E13 can be revisited.

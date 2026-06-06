# LEARNINGS: Spectral Surgery Post-Composition

## Core Finding

Post-composition spectral surgery is **structurally counterproductive** for Grassmannian-orthogonal adapter compositions. B-matrix overlap concentrates in TOP singular directions (constructive interference), while BOTTOM SVs carry clean, domain-pure signals. Surgery's premise — low-SV = noise/interference — is inverted. The -0.587 correlation between SV magnitude and domain purity is a structural property of Grassmannian compositions, not a surgery-specific failure.

## Why This Happened

Under Grassmannian orthogonality, A-matrices partition the input space cleanly (A_i A_j^T = 0), eliminating cross-terms in the left Gram matrix (Theorem 1a). But B-matrices (learned outputs) share output-space directions. When N domains project through overlapping output dimensions, the composed delta's **largest** singular values absorb the constructive interference from B-matrix overlap, while the **smallest** SVs remain domain-pure weak signals.

This is a direct consequence of PSD matrix addition: for sum_i s_i^2 B_i B_i^T, cross-terms from B-overlap boost eigenvalues in shared output directions (top of spectrum), while directions spanned by only one B_i remain unaffected (bottom of spectrum). Any magnitude-based spectral cleanup method — not just spectral surgery — will suppress the clean signal and amplify the mixed-domain interference.

**Compounding factor: cross-domain scale imbalance.** The 20:1 scale ratio (Finding #277, DC-Merge LEARNINGS) means surgery's median-based threshold classifies finance/legal components as "harmful" simply because they are small-scale, not because they are interference. This was foreseeable from the scale structure.

**Two spectral directions now definitively closed:**
1. **Pre-composition smoothing** (DC-Merge, Finding #277) — treats within-domain SV shape, but cross-domain scale imbalance dominates
2. **Post-composition surgery** (this experiment, Finding #278) — premise structurally inverted for Grassmannian compositions

## Confirming Evidence

1. **DO-Merging: Decouple and Orthogonalize (arXiv:2505.15875)** — Shows that LoRA modules exhibit much larger parameter magnitude variance than fully fine-tuned models, and greater magnitude variance correlates with worse merging performance. Directly confirms that magnitude-based spectral methods fail on LoRA compositions.

2. **KnOTS: Model Merging with SVD to Tie the Knots (arXiv:2410.19735)** — Finds SVD-based merging transfers poorly to LoRA-finetuned models due to lower alignment. Confirms that SVD techniques designed for full fine-tuning don't apply cleanly to LoRA's factored structure.

3. **Spectral Over-Accumulation (arXiv:2602.05536)** — Identifies that summation inflates singular values in shared spectral subspaces. Our -0.587 correlation is the Grassmannian-specific instance of this phenomenon: overlap concentrates in top SVs, making blind surgery counterproductive.

4. **Finding #64 (Killed)** — First spectral surgery experiment killed pre-composition surgery because short-trained adapters have efficient spectra (nothing to fix). This experiment extends the kill to post-composition: even where new spectral structure emerges from B-matrix interaction, the structure is inverted from surgery's assumptions.

5. **Finding #277 (Provisional)** — DC-Merge showed cross-domain scale imbalance (20:1) dominates over within-domain SV shape. This experiment confirms: the spectral pathology comes from inter-domain scale ratios, not intra-domain spectral noise.

## Contradicting Evidence

1. **STAR: Spectral Truncation and Rescale (arXiv:2502.10339)** — Reports positive results from SVD truncation + rescaling on model merges, scaling to 20+ models. Key difference: STAR operates on full-rank weight matrices, not factored LoRA compositions. The spectral inversion we observe is specific to Grassmannian-orthogonal factored compositions where A-orthogonality creates the clean/mixed SV inversion.

2. **Spectral Surgery (arXiv:2603.03995)** — The source paper reports +4.4 on CSQA and +2.4 on HumanEval for *individual* converged adapters. Key distinction: (a) our adapters are short-trained with efficient spectra, and (b) their method uses gradient-based sensitivity, not SV magnitude. However, the structural inversion finding (low-SV = domain-pure) holds regardless of sensitivity metric since it describes the spectral structure, not the sensitivity ordering.

3. **SVC from Spectral Over-Accumulation (arXiv:2602.05536)** — Proposes Singular Value Calibration that quantifies subspace overlap and rescales inflated SVs. Unlike blind surgery, SVC is overlap-aware. It *might* work on our compositions because it targets shared subspace inflation specifically, not low-SV components. However, at cos=0.028 B-matrix overlap, the inflation is minimal.

## Alternative Approaches

1. **Frobenius-norm equalized composition (arXiv:2506.02478, FroM)** — Normalize each domain's delta to equal Frobenius norm before summing. Directly targets the ROOT CAUSE (20:1 scale imbalance) rather than treating spectral symptoms. Already recommended by DC-Merge LEARNINGS. This is the natural next experiment.

2. **Magnitude-direction decoupling (arXiv:2505.15875, DO-Merging)** — Separates parameters into magnitude and direction, merges them independently. The direction component = W/||W||_F, which is effectively Frobenius normalization. Achieves 3-4% improvement as a "free lunch" on top of existing methods. Could compose with our Grassmannian orthogonality.

3. **TIES-MERGING (arXiv:2306.01708)** — Trims low-magnitude changes and resolves sign disagreements. Addresses scale imbalance through pruning rather than normalizing. Well-validated at scale (arXiv:2410.03617 tests across 10,506 merged models). However, trimming may conflict with our ternary adapter structure where every parameter is already {-1, 0, +1}.

4. **Activation-based subspace identification (arXiv:2602.20208, Essential Subspace Merging)** — Uses activations rather than parameter energy to identify important subspaces. Their insight — parameter energy ≠ functional impact — is exactly what our scale imbalance demonstrates. More complex than Frobenius normalization but addresses the deeper question.

## Implications for Next Experiments

1. **All magnitude-based spectral cleanup is closed for Grassmannian compositions.** The -0.587 correlation is a structural property: any method assuming "large SVs = useful, small SVs = noise" will fail. This includes truncated SVD compression, low-rank approximation of composed deltas, and magnitude-based pruning of spectral components.

2. **The disease is cross-domain scale imbalance, confirmed from two independent directions.** DC-Merge (pre-composition) and spectral surgery (post-composition) both converge on the same root cause: the 20:1 scale ratio between domains governs the composed spectrum, not SV shape or interference artifacts.

3. **Domain purity decreases in later layers (0.711 → 0.675).** B-matrix overlap increases with depth. At larger scale (r=128, N=25 → 125% output space coverage), overlap becomes mandatory. The "surgery has no target" finding may not hold — worth monitoring as we scale.

4. **Macro risk: B-overlap becomes mandatory at higher rank.** At r=16 with d_out=2560, each B_i spans 0.625% of output space. At r=128 and N=25, collective span exceeds 100%. Spectral cleanup could become necessary when overlap is no longer 2.8% but structurally forced. Record this as a scale threshold to watch.

## Recommended Follow-Up

**Frobenius-norm equalized composition** — Both the DC-Merge and spectral surgery experiments converge on the same recommendation. Normalize each domain's adapter contribution by Frobenius norm before summing. This directly addresses the root cause (cross-domain scale imbalance) that both pre-composition smoothing and post-composition surgery failed to fix because they treated symptoms (SV shape, spectral "noise") rather than the disease (energy ratio).

- **Motivation:** Finding #277 (scale imbalance dominates) + Finding #278 (surgery structurally inverted) + FroM paper (arXiv:2506.02478)
- **Literature:** FroM (arXiv:2506.02478), DO-Merging (arXiv:2505.15875), TIES (arXiv:2306.01708)
- **Prediction:** Composed Gini should drop from ~0.49 to near individual levels (~0.28). Whether this improves behavioral outcomes depends on whether the 20:1 scale ratio encodes genuine importance vs. training artifacts.

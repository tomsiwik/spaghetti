# LEARNINGS: DC-Merge SVD Energy Smoothing on Ternary Adapters

## Core Finding

Cross-domain scale imbalance (20:1 ratio from per-domain optimal scales) dominates composed spectral pathology far more than within-domain singular value shape. DC-Merge energy smoothing treats the wrong variable: equalizing individual SV distributions cannot fix a composed spectrum governed by inter-domain Frobenius norm ratios.

**Status:** PROVISIONAL (K699 marginal pass at 0.99% PPL, K700 fail at 18.5% vs 30% target)

## Why This Happened

The per-domain optimal scales {medical:20, code:20, math:20, legal:4, finance:1} (Finding #220) mean the three high-scale domains contribute ~93% of the composed delta's Frobenius energy. When you sum five rank-16 blocks with 20:1 energy ratios, the composed singular values are dominated by inter-block energy ratios, not intra-block SV shape. Smoothing individual B-matrices (Gini 0.28 -> 0) cannot change this because the composed Gini is a function of cross-domain scale structure.

This is mathematically predictable: for orthogonal subspaces (cosine=0.026, Finding K687), the composed SVD separates into block contributions weighted by domain scale. The top singular values of the composed delta correspond to the dominant directions of the highest-scale domains, regardless of individual SV equalization.

The reviewer correctly identified this as a foreseeable confound (REVIEW-adversarial.md, item 4). The 20:1 scale ratio should have been flagged analytically before running the experiment.

## Confirming Evidence

1. **Spectral Over-Accumulation (arXiv:2602.05536)** — "When Shared Knowledge Hurts" identifies that summation of task vectors inflates shared spectral directions. Their Singular Value Calibration (SVC) post-hoc rescales inflated SVs. Our finding extends this: even without shared directions (cos=0.026), scale differences alone cause spectral imbalance. SVC addresses shared-subspace inflation; our problem is scale-ratio inflation, a complementary mechanism.

2. **FroM: Frobenius Norm-Based Adaptive Model Merging (arXiv:2506.02478, EMNLP 2025)** — Uses Frobenius norm of task vectors to adaptively weight contributions during merging. This directly addresses our identified disease: rather than smoothing individual spectra, FroM adjusts the relative contribution of each task vector based on its Frobenius magnitude. This is the principled fix for the scale imbalance we observed.

3. **Model Merging in the Essential Subspace (arXiv:2602.20208)** — Introduces "Polarized Scaling" that amplifies high-norm components while suppressing low-norm ones at multiple levels (layer, task, dimension). Their key insight matches ours: SVD-based subspaces reflect parameter energy, not functional impact. They propose activation-based subspace identification instead.

4. **Finding #270** — Individual B-matrix spectra are already flat (Gini 0.20-0.31, gap 1.003-1.018). This predicted that within-domain smoothing would have limited headroom, which is exactly what happened.

5. **Finding #275** — Norm preservation (not shape preservation) is the mechanism for adapter composition. NRE matching Karcher mean reinforces that the Frobenius norm is the critical quantity, not the spectral shape.

## Contradicting Evidence

1. **DC-Merge (arXiv:2603.06242)** reports 1-3% improvement from energy smoothing on standard LoRA merging. However, their setting has FP16 adapters with steeper individual spectra (higher Gini) and uniform task scales. Our ternary adapters have flat spectra but extreme scale differences — a fundamentally different spectral regime.

2. **Task Singular Vectors (arXiv:2412.00081, CVPR 2025)** — TSV-Compress retains 99% accuracy at 10% of task vector size, suggesting that SVD-level interventions CAN be effective for composition. But their focus is on removing interference directions, not equalizing energy — a different mechanism than DC-Merge smoothing.

## Alternative Approaches (Paper-Backed Only)

1. **Frobenius-norm equalization before composition.** Normalize each domain's delta to equal Frobenius norm before summing, then rescale the composed result. This directly targets the 20:1 scale imbalance. FroM (arXiv:2506.02478) provides the framework; our specific application would be simpler (fixed N=5, known scales).

2. **Singular Value Calibration (arXiv:2602.05536).** Post-composition rescaling of inflated singular values. Unlike DC-Merge (pre-composition smoothing), SVC operates on the composed delta and can directly correct the spectral pathology we observed. Training-free, data-free.

3. **Essential Subspace Merging with Polarized Scaling (arXiv:2602.20208).** Activation-based subspace identification + multi-level adaptive rescaling. More complex than norm equalization, but addresses the deeper issue that parameter energy does not equal functional importance.

4. **Scale-aware composition (our own).** The simplest fix: divide each delta by its domain scale before summing, then apply a single composed scale. This is equivalent to Frobenius normalization when adapters have similar internal structure (which ours do — Gini 0.27-0.29 across all domains).

## Implications for Next Experiments

1. **The scale problem is solved, not open.** We already know the optimal scales (Finding #220). The question is whether normalizing contributions before composition (so each domain contributes equally to the sum) improves behavioral outcomes vs. the current scale-weighted sum.

2. **Norm equalization vs. current approach is a clean A/B test.** Current: sum(scale_i * delta_i). Proposed: N * mean_scale * sum(delta_i / ||delta_i||_F). This preserves total energy but equalizes per-domain contributions. The prediction: composed Gini will drop dramatically (from 0.49 to near individual levels ~0.28), but PPL may not improve if the scale differences encode actual importance differences.

3. **The deeper question is whether scale=20 reflects domain importance or just training dynamics.** Medical/code/math all land at scale=20 from PPL grid search. If this reflects genuine importance (these domains need 20x the weight update magnitude), then normalizing away the scale difference would harm those domains. If it reflects training artifacts (different data distributions produce different gradient magnitudes), normalization could help.

4. **Behavioral evaluation remains the bottleneck.** Even if normalization improves composed Gini, PPL correlation with task performance is r=0.08 (project finding). Any follow-up MUST include behavioral evaluation (generation quality, task accuracy), not just PPL.

## Recommended Follow-Up

**Frobenius-norm equalized composition** — Directly test whether normalizing domain contributions before summation reduces composed spectral imbalance without degrading behavioral outcomes. Cite FroM (arXiv:2506.02478) for the Frobenius-norm approach and SVC (arXiv:2602.05536) for post-hoc spectral correction as alternatives.

Motivation: This experiment's core negative finding (scale imbalance > spectral shape) directly predicts that norm equalization should succeed where energy smoothing failed. The test is cheap (reuse existing adapters, modify only the composition step) and produces a clear binary result.

# LEARNINGS: Fisher-Rao Composition Scaling

## Core Finding

**Norm preservation is the single mechanism that prevents adapter composition degradation at scale. The Riemannian manifold machinery (Karcher mean) is unnecessary -- a one-line norm rescaling after Euclidean averaging achieves identical PPL (9.17 vs 9.20) at 10x lower cost.** Theorem 1 (norm preservation prevents 1/sqrt(N) shrinkage) verified exactly; Conjectures 2-3 (activation variance/effective rank) predicted wrong direction.

## Why This Happened

The result decomposes into two independent findings:

**1. Norm shrinkage is real and harmful.** Euclidean averaging of near-orthogonal adapter deltas shrinks norms as 1/sqrt(N) (Corollary of Theorem 1, verified at N=3: 0.575 vs theory 0.577, N=5: 0.448 vs 0.447). This is a direct consequence of the triangle inequality -- partial cancellation of orthogonal vectors. The PPL degradation (8.98 -> 10.44, +16.3% at N=5) confirms that norm shrinkage translates to behavioral harm. This matches Jang et al. (2024) who analyzed norm shrinkage in model averaging contexts.

**2. Karcher mean adds nothing over normalized Euclidean mean for near-orthogonal vectors.** For unit vectors with pairwise cosine ~0 (our Grassmannian A-matrices guarantee |cos| = 0.00125), the normalized Euclidean mean and the Karcher mean on S^(d-1) converge to the same direction. The worked example in MATH.md Section F proves this algebraically for orthogonal vectors: both yield (u1+u2+u3)/||u1+u2+u3||. The iterative Riemannian optimization (2.01s at N=5) simply recovers what normalization gives for free (0.13s).

**Why Conjectures 2-3 failed:** The linear response model (Var ~ ||Delta||^2) ignores that multi-domain composition introduces *diverse hidden state trajectories*. Even as weight norms shrink, the activation space diversifies because different adapter domains create new activation directions not present in any single adapter. This is fundamentally a nonlinear effect that the first-order Taylor expansion cannot capture.

## Confirming Evidence

- **DO-Merging (arXiv:2505.15875)** independently identifies magnitude-direction decoupling as the key to successful LoRA merging. Their finding that "large magnitude variances cause deviations in parameter distributions" directly confirms our observation that norm shrinkage (a form of magnitude distortion) is the disease. Their fix -- separate magnitude and direction merging -- is conceptually identical to our norm-rescaled Euclidean approach, though they add orthogonal gradient descent on directions.

- **DoRA (arXiv:2402.09353, ICML 2024 Oral)** decomposes weights into magnitude and direction components for *training*, finding that LoRA conflates magnitude and direction updates. Our finding extends this to *composition*: when merging multiple LoRA adapters, preserving magnitude separately from direction is sufficient.

- **Naive LoRA Summation (arXiv:2508.11985)** demonstrates that independently trained LoRA modules on disjoint domains are approximately orthogonal and can be combined by simple addition. Their finding that RMS cosine similarity between LoRA deltas correlates linearly with PPL change supports our architecture: Grassmannian A-matrices enforce the orthogonality that makes naive summation work, and norm rescaling removes the one remaining failure mode (1/sqrt(N) shrinkage).

- **Our Finding #14** (1/N scaling resolves composition catastrophe) was an early signal: the 1/N fix that took PPL from trillions to 2.36 was itself a norm-preservation mechanism -- it prevented the *opposite* problem (norm explosion from summation without averaging).

## Contradicting Evidence

- **OrthoMerge (arXiv:2602.05943)** argues that Euclidean merging "destroys intrinsic geometric properties" and that Riemannian manifold operations are necessary. However, their analysis targets full model merging with non-orthogonal weight matrices, not adapter composition where orthogonality is *enforced by design*. Our result holds specifically because Grassmannian initialization makes adapters near-orthogonal, collapsing the Karcher mean to the normalized Euclidean mean.

- **The directional quality hypothesis** remains untested for regimes where adapters are NOT near-orthogonal. With overlapping domain knowledge (e.g., medical + biomedical), the Karcher mean might provide directional benefits that norm rescaling cannot. This is acknowledged in the REVIEW as a macro-scale risk.

- **Moderate norm shrinkage as regularization:** It's possible that some shrinkage is beneficial as implicit regularization (analogous to weight decay). At N=5, raw Euclidean PPL (10.44) is worse than norm-preserved (9.17), but at smaller N the gap narrows. The optimal shrinkage might be non-zero at macro scale with many adapters.

## Alternative Approaches (with paper references)

1. **DO-Merging (arXiv:2505.15875):** Full magnitude-direction decoupling with orthogonal gradient descent on direction components. More sophisticated than our NRE but potentially needed for non-orthogonal adapters. Reports 3%+ improvement on vision/language/multimodal tasks.

2. **TIES-Merging (arXiv:2306.01708):** Resolves parameter interference by trimming small changes, resolving sign conflicts, and merging only aligned parameters. Addresses a different failure mode (sign interference) that becomes relevant when adapters share parameters.

3. **DARE (Drop And REscale):** Random dropout of delta parameters + rescaling to maintain magnitude. An alternative norm-preservation mechanism through stochastic sparsification. Could combine with our NRE for additional robustness.

4. **DoRA (arXiv:2402.09353):** If we switch training to DoRA (magnitude-direction decomposition during training), the composition problem may be further simplified since magnitude and direction are tracked separately throughout.

5. **Riemannian LoRA Optimization (arXiv:2508.17901):** Optimizes LoRA on the Stiefel manifold during training. Could replace our Grassmannian initialization with continuous manifold constraints.

## Implications for Next Experiments

1. **Norm-rescaled Euclidean is the composition method.** Replace all Euclidean averaging in the pipeline with the one-line NRE. Zero hyperparameters, negligible cost, 12% PPL improvement at N=5.

2. **The N=5 ceiling is the real limitation.** All metrics plateau because only 5 independent adapters exist. The critical test is whether NRE maintains its advantage with N=15-25 *truly independent* adapters (the exp_real_data_25_domain_adapters experiment).

3. **Conjectures 2-3 reveal a gap in our theory.** The linear response model fails at predicting activation variance and effective rank under multi-domain composition. A second-order model that accounts for cross-domain interaction terms is needed.

4. **Grassmannian orthogonality is load-bearing for simplicity.** Our NRE = FR result depends on adapters being near-orthogonal. If future adapters violate this (e.g., from overlapping training data), the Karcher mean or DO-Merging's orthogonal direction merging may become necessary.

## Recommended Follow-Up

**exp_generation_quality_test (already P0 critical path):** The norm preservation finding is a weight-space property. The existential question remains behavioral: does routed composition with NRE produce *useful text*? PPL improved 12% but PPL-task correlation is only r=0.08 (our own Finding #56). This experiment must happen regardless.

**Motivation:** Our Finding #56 (KR-Test delta rank-correlates with task accuracy, r=1.0 on n=4) plus the PPL-task decorrelation (r=0.08) means we cannot infer behavioral quality from PPL alone. The norm preservation finding de-risks the composition mechanism but says nothing about output quality.

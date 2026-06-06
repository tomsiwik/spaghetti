# Learnings: exp_bitnet_spectral_surgery_quality_gate

## Core Finding

Spectral Surgery (arXiv:2603.03995) — SVD decomposition + gradient-guided singular value reweighting — has zero effect on short-trained (200 iter, 1 epoch) rank-16 LoRA adapters on BitNet-2B-4T. The technique addresses a problem (inefficient spectra from overtraining) that our adapters do not have. Secondary finding: SVD re-factorization increases inter-adapter cosine similarity by 3.2x, actively harming composition orthogonality.

## Why This Happened (Literature-Grounded)

**Root cause: Training duration mismatch.** The Spectral Surgery paper explicitly targets *converged* adapters trained for 3 full epochs with standard learning rate schedules. Their core insight is that SGD convergence does not guarantee efficient allocation of the rank-r budget — beneficial effects concentrate in a few singular directions while remaining directions accumulate noise or detrimental signal. Our adapters (200 iterations, ~1 epoch) haven't trained long enough to develop this "inefficient spectrum."

**Mechanistic explanation:** Recent spectral analyses of LoRA (Biderman et al. 2024; Shuttleworth et al. 2024) show that LoRA updates are dominated by a small number of top singular vectors ("intruder dimensions") with the effective rank being significantly smaller than nominal rank. At 200 iterations with rank-16, our adapters likely use only 2-4 active singular directions — all task-relevant. The surgery's nuclear-norm constraint (||sigma'||_1 = ||sigma||_1) makes reweighting zero-sum: with no noise to remove, any suppression of one direction must amplify another equally important one, producing no net benefit.

**The "Alignment Tax" compounds the problem.** Even on converged adapters, the paper documents that gradient-guided surgery fails on instruction-following tasks (IFEval: 0.590 → 0.173 on Qwen3-8B). Gradient signals optimize for next-token prediction loss, which can trade off against structural/formatting constraints. Our domain-specific adapters may exhibit similar misalignment between calibration gradients and actual quality.

**Composition damage from SVD re-factorization.** When surgery reconstructs A_new, B_new via sqrt-split of modified singular values, the new factorization rotates adapter subspaces. Even with minimal singular value changes, this rotation increases alignment between adapters (cosine: 0.001 → 0.0032, +3.2x). The Grassmannian orthogonality that enables composition is a property of the original A, B factorization, not the effective delta — SVD re-factorization destroys this.

## Confirming Evidence

- **Spectral Surgery paper (arXiv:2603.03995):** Method explicitly designed for converged adapters. Reports "spectral brittleness" — even random reweighting sometimes helps converged adapters — confirming that spectral noise accumulates only with extended training. Their best results (+4.4pp CSQA) use Llama-3.1-8B after 3 epochs of standard LoRA training.
- **Biderman et al. 2024 ("LoRA vs Full Fine-Tuning"):** LoRA weight matrices develop "intruder dimensions" — high-ranking singular vectors absent in the pretrained weights. These emerge during training, implying short training produces fewer/weaker intruders with less spectral inefficiency.
- **PiSSA, MiLoRA, LoRA-XS (2024-2025):** Multiple papers show SVD-based initialization of LoRA captures most task-relevant information in the top singular vectors. This confirms that early-stage LoRA updates are spectrally concentrated (efficient), supporting our finding that short-trained adapters leave nothing for surgery to improve.
- **Our own retrain_evolve LEARNINGS:** PPL and KR-Test diverge for short-trained LoRA — adapters learn style (PPL) before facts (KR-Test). Surgery can't fix this fundamental ordering; it can only redistribute existing spectral energy.

## Contradicting Evidence

- **Spectral Surgery paper's `smooth_abs` policy** achieves modest gains even on some aligned tasks with low risk of degradation. We did not test this specific policy — our implementation used `grad_direction`-style signed updates. However, given the zero KR-Test movement across all 5 domains, switching policy is unlikely to change the verdict for our short-trained adapters.
- **Module locality:** The paper recommends restricting surgery to residual-writing modules only (o_proj, down_proj). We applied to all LoRA modules. Restricting could reduce the composition damage (fewer re-factorized modules = less subspace rotation) but would not address the fundamental "already-efficient spectrum" problem.
- **"Make LoRA Great Again" (arXiv:2502.16894):** Proposes adaptive singular values during training + MoE optimization alignment. This is a *training-time* intervention, not post-hoc, and might succeed where post-hoc surgery fails because it shapes the spectrum as it forms rather than trying to fix it after convergence.

## Alternative Approaches (What We Could Try Instead)

1. **Evaluation-only quality gate (RECOMMENDED):** PPL + KR-Test non-regression + cosine < 0.05. Our retrain_evolve LEARNINGS already defined this gate. No post-hoc refinement needed — just evaluate and accept/reject. Zero additional compute cost.

2. **ESSA — Evolutionary Strategies for Scalable Alignment (arXiv:2507.04453):** Gradient-free evolutionary search on LoRA singular values. Matches GRPO alignment at 72B scale using INT4/INT8 inference only. Unlike spectral surgery, ESSA doesn't assume inefficient spectra — it searches for better allocations. Could be viable for our Evolve pipeline, but requires a reward model or preference signal, not just calibration data.

3. **Retrain-from-scratch (our existing approach):** Already proven in exp_bitnet_retrain_evolve. Since adapters are spectrally efficient, the only way to improve them is to train on better/more data. This remains the primary Evolve primitive.

4. **Task Arithmetic (Ilharco et al. 2023):** Treat adapters as task vectors and compose via addition/negation. Already subsumed by our 1/N scaling composition approach. Not a quality gate, but a composition method.

5. **Adaptive singular values during training (arXiv:2502.16894):** If we ever move to longer training regimes where spectral inefficiency could develop, integrating adaptive SV optimization during training (not post-hoc) would preempt the problem rather than trying to fix it later.

## Implications for Next Experiments

1. **Quality gate is evaluation-only.** Post-hoc refinement (spectral surgery, SVD reweighting) adds cost without benefit for our short-trained adapters. The Evolve quality gate should be: PPL + KR-Test non-regression + composition cosine < 0.05. This is now a confirmed design decision.

2. **SVD re-factorization is composition-hostile.** Any future experiment that proposes per-adapter SVD transformations (compression, surgery, or otherwise) must include a composition-aware constraint or test. The 3.2x cosine increase is small in absolute terms (0.001 → 0.0032) but directionally wrong. At N=100+ adapters, accumulated rotation could breach the 0.05 threshold.

3. **Short-trained adapters are spectrally clean.** This is a *strength* of our approach, not a weakness. The 200-iteration training regime naturally produces concentrated, efficient updates. Longer training would create the spectral inefficiency that surgery addresses — but would also increase training cost and potentially hurt composition orthogonality.

4. **ESSA is worth watching** but not actionable now. It requires a reward model or preference signal we don't have. If the project ever needs alignment-style improvement (not just domain specialization), ESSA's gradient-free approach on singular values could integrate cleanly with our ternary pipeline.

5. **No hypothesis modifications needed.** The kill is clean. exp_bitnet_spectral_surgery_quality_gate remains killed. No new hypothesis is motivated — the evaluation-only gate from retrain_evolve is confirmed sufficient.

## New References to Add

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| ESSA: Evolutionary Strategies for Scalable Alignment | arXiv:2507.04453 | Gradient-free SV optimization for alignment. Alternative to spectral surgery for future Evolve work. |
| Make LoRA Great Again: Adaptive Singular Values + MoE | arXiv:2502.16894 | Training-time adaptive SV — addresses spectral inefficiency during training rather than post-hoc. |

# Learnings: exp_task_accuracy_real_benchmarks

## Core Finding

Uniform 1/N LoRA composition at reduced per-adapter scale (scale/N=4.0) produces a +16pp GSM8K reasoning improvement over base, while oracle routing to domain-specific adapters hurts factual knowledge (MMLU) on 4/5 domains. The mechanism behind the uniform advantage is unclear, and all accuracy differences are dominated by ~10pp run-to-run variance from temp=0.1 sampling.

## Why This Happened (Literature-Grounded)

The uniform composition advantage on GSM8K and the routing penalty on MMLU stem from distinct mechanisms well-documented in the literature:

**1. Scale-dependent regularization vs. catastrophic interference.** Our prior macro experiments (composition_weight_normalization, composition_dropout_robustness) showed that uniform composition at unit scale is catastrophic (PPL in trillions). But this experiment uses scale/N=4.0 per adapter, not scale=20.0. At this reduced magnitude, adapters act as small perturbations that regularize the output distribution rather than dominating it. This is consistent with pilot50_composition_quality where composed PPL substantially beat naive 1/N dilution predictions (medical: predicted 20.6, actual 7.8). The exp_cross_adapter_knowledge_transfer kill (0/20 pairwise transfers >2%) confirms this is NOT constructive knowledge transfer but rather a regularization/noise-smoothing effect.

**2. Format mismatch explains MMLU degradation.** The adapters were trained on domain-specific instruction-response text, not multiple-choice QA. The SOLE adversarial review found that 3/3 tested adapters degraded MMLU by an average of -3.71pp, concluding adapters are "format-specialized, not knowledge-additive" (macro/individual_expert_held_out). Our MMLU results replicate this: legal drops from 50% to 20% (below random chance), confirming the adapter over-constrains output distribution for knowledge retrieval tasks.

**3. Logit-scale mismatch and function-space interference.** Even orthogonal weight-space adapters can interfere in function space. arXiv:2510.03262 ("Rethinking Inter-LoRA Orthogonality in Adapter Merging") demonstrated empirically that strict inter-LoRA orthogonality does not guarantee semantic disentanglement. Our own exp_bitnet_semantic_compositionality confirmed this: 100% of adapter pairs fail data-orthogonality (mean BA-full ratio 0.88 >> 0.1 threshold), yet composition still works via 1/N regularization.

**4. Temperature noise dominates all signals.** The routing gap test showed ~10pp run-to-run variance at temp=0.1, with the original 12pp individual-routed gap completely reversing in replication. This is a methodological artifact, not a finding about the architecture. The "Mixture of Parrots" paper (arXiv:2305.14705) notes that MoE reasoning improvements saturate quickly and are hard to measure at small scale -- our noisy signal is consistent with this.

## Confirming Evidence

- **arXiv:2602.21222** (Task-Aware LoRA Adapter Composition): Linear merging of multiple adapters surpassed "Perfect Selection" oracle on PIQA (70.95% vs 46%) and RTE (77.62% vs 52%). Multi-adapter composition creates synergistic effects that single-expert routing cannot.

- **LoRAuter** (Effective LoRA Adapter Routing using Task Representations): Weighted fusion of top-K adapters achieved 101.2% of oracle performance on in-domain tasks. Single adapter routing (K=1) yielded "fairly low performance" compared to K=3 composition.

- **arXiv:2603.03535** (cited in our PAPER.md): Ensembling > routing > merging as general ordering for multi-LoRA composition quality. Consistent with our uniform > routed finding.

- **Our own macro/sole_critical_path**: PPL-probe weighting only +0.27% over equal-weight at N=5, confirming that at small N, adapters have similar probe PPL and intelligent weighting offers marginal benefit.

- **"Unchosen Experts Can Contribute Too"**: Improved GSM8K from 61.79 to 66.94 via "self-contrast" strategy juxtaposing strong and weak expert activations -- non-routed multi-expert benefits on reasoning are real.

## Contradicting Evidence

- **Our own macro/composition_dropout_robustness**: Equal-weight composition KILLED (CV=112.2%, PPL in trillions). One harmful adapter poisons everything. Resolution: that experiment used unit-scale weights, ours uses scale/N=4.0. The scale difference is load-bearing.

- **Our own macro/composition_weight_normalization**: Uniform static scaling KILLED (best/single=2.57x). Composed PPL approximately equals base (adapters cancel out). Resolution: this measured PPL, not task accuracy. PPL cancellation may coexist with reasoning improvement if the regularization effect is format-dependent.

- **LoTA-QAF** (Lossless Ternary Adaptation for QAT): Ternary adapters CAN improve MMLU by up to 5.14% when specifically designed for quantization recovery on Qwen2.5-14B. Our adapters were not designed for MMLU-style factual recall.

- **LoRAuter K=1 vs K=3**: While confirming multi-adapter > single-adapter, LoRAuter used WEIGHTED composition (not uniform), suggesting learned weights matter. Our uniform advantage may not hold when adapters have divergent magnitudes (which macro experiments confirmed at N>5).

- **Our own micro/models/ppl_vs_task_performance**: PPL does NOT predict task accuracy (Pearson r=0.08). This means our PPL-based findings about composition may not transfer to task accuracy at all -- the two measurement regimes capture different phenomena.

## Alternative Approaches (What We Could Try Instead)

1. **Temperature=0.0 replication (CRITICAL).** The ~10pp variance from temp=0.1 makes all current results unreliable. Before any other experiment, rerun with greedy decoding and 3+ seeds to establish true signal-to-noise ratio.

2. **PPL-probe weighted composition on benchmarks.** Our PPL-probe weighting (macro/sole_critical_path) was marginal at N=5 but could differentiate better on task accuracy where adapter contributions are format-dependent. Test PPL-probe weights on GSM8K/MMLU.

3. **TIES merging** (Trim, Elect Sign, Merge): Keeps top-k% most significant weight changes, resolves sign conflicts via majority consensus. Could preserve reasoning benefits while pruning the parameter interference that hurts MMLU. arXiv:2306.01708.

4. **DARE** (Drop And REscale): Random reset of adapter weights + rescale. Makes multi-adapter stacking more compatible. arXiv:2311.03099. Combined with TIES, this is the state-of-the-art for static parameter merging.

5. **LoRAMoE**: MoE-style gating that explicitly protects world knowledge by dedicating distinct expert pathways. Directly addresses the reasoning-vs-knowledge trade-off we observed.

6. **Answer-conditioned scoring** (our own micro/models/answer_conditioned_scoring): answer-only PPL correlates with task accuracy at r=0.811. Use this as the routing signal instead of full-sequence PPL, which anti-correlates.

7. **Entropy-adaptive gating**: Skip experts when base model is confident (from our P0 priorities). Would naturally route MMLU knowledge questions to base-only (high confidence) and GSM8K reasoning to composed model (low confidence).

## Implications for Next Experiments

1. **Methodological:** All future task-accuracy experiments MUST use temp=0.0 and multiple seeds. The ~10pp variance invalidates any comparison smaller than ~14pp at N=50. This is a hard requirement, not a nice-to-have.

2. **The uniform advantage is real but fragile.** It works at scale/N=4.0 on GSM8K but our macro experiments killed it at unit scale. The scale parameter is the critical control variable. Future work should sweep scale/N from 1.0 to 20.0 to find the regime boundary.

3. **Routing vs. composition is the wrong framing.** The literature (LoRAuter, arXiv:2602.21222) consistently shows that weighted multi-adapter composition beats both uniform and single-adapter routing. The real question is: what's the optimal weighting strategy? Our answer-conditioned scoring (r=0.811) is the most promising routing signal we have.

4. **The reasoning-vs-knowledge split deserves dedicated investigation.** Our adapters help reasoning (GSM8K +16pp) but hurt knowledge (MMLU -10pp on legal). This maps cleanly to the format mismatch hypothesis: instruction-tuned adapters bias toward generation, which helps step-by-step reasoning but hurts multiple-choice selection. Testing format-matched adapters (few-shot MMLU fine-tuning) would isolate this.

5. **The three competing hypotheses (cross-domain transfer, regularization, extraction artifact) need a designed experiment to separate.** Manual inspection of uniform-correct/base-wrong GSM8K examples would be cheap and decisive. If uniform produces longer outputs with lucky number matches, it's extraction artifact. If it produces better reasoning chains, it's genuine.

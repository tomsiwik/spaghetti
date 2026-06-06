# Learnings: exp_purpose_trained_module_adapters

## Core Finding
Co-adaptation during full-module LoRA training is beneficial for behavioral quality: attention B-matrices trained jointly with MLP modules carry cross-module representational structure that persists even when MLP is removed at serving time. Purpose-trained attn-only adapters achieve slightly better PPL but significantly worse behavioral quality (medical -28.7%, math -12.5%). Module selection is a SERVING optimization, not a TRAINING decision. Train all 7 modules, select optimal subset at inference.

## Why This Happened (Literature-Grounded)

Three mechanisms explain why co-adaptation helps rather than hurts:

**1. MLP stores knowledge, attention routes to it.** Geva et al. (arXiv:2012.14913) established that transformer MLP layers act as key-value memories for factual knowledge. Xu & Chen (arXiv:2508.00901) formally prove this: "filtering with self-attention and storing with MLP" — attention selects relevant knowledge, MLP stores it. When full-module training includes MLP perturbation, attention B-matrices learn WHERE domain-specific knowledge lives. When MLP adapters are removed at serving, these routing patterns still access base-model MLP knowledge more effectively than attention trained without MLP context. The attention doesn't just learn "what to attend to" — it learns "what to attend to given domain-specific MLP storage patterns."

**2. Cross-module gradient flow creates richer representations.** Low-Rank Interconnected Adaptation (arXiv:2407.09946) demonstrates that cross-layer/cross-module connections "facilitate the capture of intricate information and dependencies across different layers, thereby enhancing the model's representational capabilities." Our B-matrix divergence data (cosine 0.908-0.938, norm ratio 1.21-1.37) quantifies exactly this: full-module training produces attention B-matrices that encode MLP-interaction knowledge via dL/dB_attn depending on h(B_mlp). Purpose-trained B-matrices, lacking this gradient pathway, learn a fundamentally different (and apparently inferior) solution for behavioral tasks.

**3. LoRA's low rank preserves base knowledge during co-adaptation.** Biderman et al. (arXiv:2405.09673) show LoRA "learns less and forgets less" — the low-rank constraint prevents catastrophic overwriting of base model knowledge. This means co-adaptation during full-module training does NOT damage base MLP knowledge. The MLP perturbation teaches attention where to route, then removing MLP at serving leaves the base MLP intact WITH attention that knows how to use it. Purpose-trained adapters never learn this routing, getting better PPL (surface prediction) but worse factual recall (deeper knowledge access).

## Confirming Evidence

- **Geva et al. (arXiv:2012.14913)**: MLP layers as key-value memories. Full-module training lets attention learn to route through these memories; purpose-training misses this signal.
- **Xu & Chen (arXiv:2508.00901)**: Formal proof that self-attention filters and MLP stores. The functional division means training with both teaches attention the storage structure.
- **Low-Rank Interconnected Adaptation (arXiv:2407.09946)**: Cross-module information flow enhances representational capacity — directly analogous to our co-adaptation benefit.
- **LoRA Learns Less, Forgets Less (arXiv:2405.09673)**: Low-rank constraint preserves base knowledge, making co-adaptation safe (MLP perturbation during training doesn't damage base MLP at serving).
- **PPL-behavioral dissociation (arXiv:2603.29396)**: "Is my model perplexed for the right reason?" shows benchmark performance does not consistently reflect token-level perplexity behavior. LLMs "solve tasks for other, less explainable reasons than the expected one." Directly validates our r=0.08 PPL-behavioral correlation.
- **PPL-behavioral dissociation in personality (arXiv:2508.04826)**: Reasoning increases variability while decreasing perplexity — "a clear dissociation between these metrics." Confirms PPL is a poor proxy for behavioral quality.
- **PPL prediction failure in fine-tuning (arXiv:2504.12491)**: Conventional perplexity prediction error rates exceed 60% for downstream tasks — worse than random guessing.

## Contradicting Evidence

- **PLoP (arXiv:2506.20629)**: Task-specific optimal LoRA placement outperforms uniform placement. Our experiment predicted this would extend to training-time module selection (purpose-training). It doesn't — PLoP's gains come from which modules are ACTIVE during serving, not from training with reduced module sets. Our result is actually consistent with PLoP: optimal placement at serving time IS the right approach.
- **A Note on LoRA (arXiv:2404.05086)**: Suggests optimal LoRA placement is highly dataset/model-dependent. While true for serving, our data shows training should remain uniform (all modules) regardless of the serving configuration.

## Alternative Approaches

### Proven in our codebase:
1. **Post-hoc module ablation (Finding #304)**: Now upgraded to supported. Train full-module, select at serving time. The correct approach confirmed by this experiment.
2. **Pre-merge serving (Finding #225)**: 0% overhead on MLX, 165 tok/s. Module selection stacks with pre-merge for additional bandwidth reduction.

### From literature (untested):
3. **Structured pruning of adapter modules (arXiv:2504.21174)**: AMP-style pruning of attention heads and MLP for LoRA adapters, guided by importance scores. Could automate the per-domain module selection table from Finding #304 rather than manual search.
4. **MTL-LoRA (arXiv:2410.09437)**: Multi-task LoRA with explicit task-specific and shared knowledge decomposition. Could formalize co-adaptation by making the cross-module knowledge transfer explicit.
5. **Scale-aware serving**: Purpose-trained B-matrices have 21-37% larger norms (Limitation 5). A scale sweep for purpose-trained adapters could narrow whether the behavioral deficit is co-adaptation loss or scale mismatch. Low effort, high information.

## Implications for Next Experiments

1. **Training pipeline confirmed: always train all 7 modules.** No need to customize training configurations per domain. This simplifies the adapter creation pipeline (one config for all domains).

2. **Serving pipeline confirmed: use Finding #304's per-domain module table.** Medical/math/legal/finance use attn-only (4 modules), code uses full (7 modules). This is 43% bandwidth reduction for prose domains with no training cost.

3. **The PPL-behavioral dissociation is now confirmed from THREE independent experiments** (Finding #304, Finding #308, and project-wide r=0.08). All future kill criteria MUST include behavioral metrics. PPL-only criteria are non-discriminating.

4. **Scale confound is the strongest open question.** Purpose-trained B-matrices compensate with 21-37% larger norms. A targeted scale sweep (s ∈ {10, 15, 20, 25, 30}) on purpose-trained adapters would resolve whether the behavioral gap is entirely co-adaptation or partially scale mismatch. This is a ~30-minute experiment worth running if it matters for the training pipeline decision.

5. **Co-adaptation mechanism deserves formal proof.** The post-hoc explanation (attention learns MLP routing patterns) is plausible but speculative. A formal proof connecting cross-module gradient flow to behavioral quality would elevate Finding #308 from supported to conclusive.

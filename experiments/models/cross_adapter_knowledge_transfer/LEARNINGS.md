# Learnings: exp_cross_adapter_knowledge_transfer

## Core Finding

Grassmannian-orthogonal LoRA adapters are independent modules: 0/20 pairwise domain pairs show >2% cross-domain transfer via weight-space blending, with the transfer matrix indistinguishable from noise. The "constructive transfer" observed in prior N-adapter composition was 1/N regularization, not knowledge sharing.

## Why This Happened (Literature-Grounded)

The null result has three complementary mechanistic explanations from the literature:

**1. Weight-space orthogonality ≠ function-space independence, but still prevents blending.**
"Rethinking Inter-LoRA Orthogonality in Adapter Merging" (arxiv 2510.03262) proves that enforcing strict geometric orthogonality between LoRA modules does not produce semantic disentanglement. However, the converse is also true: orthogonal weight perturbations that pass through dense nonlinearities (residual streams, LayerNorm) produce correlated output perturbations that interfere rather than cooperate. The adapters occupy disjoint weight subspaces, so blending at alpha=0.1 contributes signal in a direction the native domain's data doesn't activate.

**2. Convex interpolation conflates addition with dilution.**
Our composition formula `W = W_base + α·Δ_foreign + (1-α)·Δ_native` simultaneously adds foreign signal and removes native signal. The adversarial review correctly identified this as a confound: any positive transfer must overcome the cost of diluting the native adapter. At α=0.1, 10% of domain-specific signal is replaced with foreign signal. The 15/20 pairs showing strictly destructive interference at ALL alpha values confirms that the foreign contribution cannot compensate for even 10% native signal loss.

**3. Magnitude domination silences weaker adapters.**
Cosine orthogonality ignores weight norms. When two orthogonal adapters have different magnitudes (which is common — our legal adapter produces PPL 16.5 vs python's 2.2), the higher-magnitude adapter dominates output logits. The foreign adapter's contribution is effectively silenced in the blended model, preventing any cross-domain knowledge from surfacing in PPL measurements.

## Confirming Evidence

Several papers confirm that naive weight-space blending of independent adapters fails:

- **"Rethinking Inter-LoRA Orthogonality in Adapter Merging"** (arxiv 2510.03262): Strict geometric orthogonality does not lead to semantic compositionality. Orthogonal adapters merged in weight space show interference rather than cooperation.

- **TIES-Merging** (Yadav et al., NeurIPS 2023): Weight-space merging via majority-sign consensus sacrifices sentence-level fluency and breaks causal/narrative coherence — demonstrating that parameter-space operations destroy semantic structure.

- **Tensorized Clustered LoRA Merging** (arxiv 2508.03999): Acknowledges multi-task interference as a fundamental problem in LoRA merging, requiring explicit clustering by task similarity to avoid destructive interference.

- **Our own OSRM result** (exp_bitnet_semantic_compositionality): Weight-space orthogonality ≠ data-space orthogonality. Adapters can be orthogonal in parameters while producing correlated outputs, exactly the scenario that prevents pairwise transfer.

## Contradicting Evidence

Some papers DO find positive cross-adapter transfer, but under different conditions:

- **LoRAuter** (Adapters Selector, COLING 2025): Dynamic merging based on input-task similarity achieves strong zero-shot generalization — but uses uncertainty-aware fusion with learned routing, not static weight blending. The key difference: LoRAuter selects WHICH adapters to use per input, rather than blending all adapters at fixed weights.

- **MTLoRA** (CVPR 2024): Multi-task LoRA with shared encoder achieves positive cross-task transfer — but adapters are trained jointly with a shared backbone, not independently. Joint training aligns adapter subspaces; our Grassmannian initialization explicitly decorrelates them.

- **MoDULA** (EMNLP 2024): Mixture of domain-specific and universal LoRA shows cross-domain benefit — but includes explicit "universal experts" trained on mixed-domain data that act as bridges. Our setup has no universal adapter component.

- **I-LoRA** (OpenReview): Iterative merging of routing-tuned adapters preserves knowledge across tasks — but uses routing-aware training that explicitly optimizes for composability.

**Reconciliation:** All contradicting results use either (a) learned routing/selection, (b) joint training, or (c) explicit universal components. Our null result is specific to independently-trained, Grassmannian-orthogonal adapters composed via static weight blending. This is consistent — the literature shows transfer requires either shared representations or intelligent selection, neither of which our setup provides.

## Alternative Approaches (What We Could Try Instead)

### 1. Per-Token Routing (MoLoRA, arxiv 2603.15965)
Instead of blending adapter weights, route each token to the most relevant adapter. MoLoRA shows Qwen3-1.7B + 4 LoRA adapters outperforms a single 8B model via per-token Gumbel-softmax routing. This is the architecture our project is moving toward and is directly validated by this kill: routing >> blending.

### 2. Output-Space Ensemble (Logit Ensemble)
Rather than composing in weight space, run each adapter separately and combine output logits. Our own logit ensemble experiment (exp_bitnet_logit_vs_weight_merge, if run) would test whether semantic transfer emerges in output space even when weight-space transfer fails. The "Logit Ensemble vs Weight Merge" notebook already exists.

### 3. Adapter Fusion with Learned Attention (AdapterFusion, Pfeiffer et al., EACL 2021)
Train a small attention layer over adapter outputs. This learns WHICH adapter's output is useful for which input, operating in activation space rather than weight space. Low overhead (~0.1% additional params).

### 4. Task Arithmetic in Tangent Space (NeurIPS 2023)
Instead of naive weight addition, project task vectors into the tangent space of the pre-trained model. This preserves the geometric structure of the loss landscape and can enable constructive interference between adapters that naive addition destroys.

### 5. Universal Adapter Bridge (MoDULA-style)
Add a small universal adapter trained on mixed-domain data that acts as a "bridge" between domain-specific adapters. This explicitly provides the shared representation that Grassmannian initialization removes.

### 6. Additive Composition Test (Reviewer-Suggested)
Test `W = W_base + β·Δ_foreign + 1.0·Δ_native` (native at full strength, foreign additive). This isolates whether foreign adapters contribute ANY useful signal without the dilution confound. If this also shows null transfer, the independence conclusion is iron-clad.

## Implications for Next Experiments

1. **Routing is the correct composition mechanism, not blending.** All evidence points to per-token or per-sequence adapter SELECTION as the way to achieve multi-domain benefit. Weight-space composition provides safety (no interference) but not synergy.

2. **The 1/N regularization insight is load-bearing.** Prior composition results showing "composed PPL beats naive prediction" are explained by regularization from 1/N scaling, not cross-domain transfer. This means N-adapter composition quality scales with regularization benefit, not with N·(N-1) pairwise interactions. This simplifies the architecture: we don't need to optimize adapter interactions, just adapter selection.

3. **The additive composition test (point 6) should be run** to close the convex interpolation confound raised by the adversarial review. This is cheap (reuse existing adapters) and would strengthen the independence conclusion.

4. **Output-space vs weight-space** remains an open question. Transfer might exist in logit/activation space even when absent in weight space. The logit ensemble experiment is the natural next step.

5. **For production serving:** Hard top-1 routing (select best single adapter per token/sequence) is likely sufficient. Soft blending adds complexity without benefit given adapters are independent. This simplifies the serving architecture on M5 Pro.

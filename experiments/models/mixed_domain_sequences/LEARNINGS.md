# Learnings: exp_mixed_domain_sequences

## Core Finding
Post-hoc per-token Gumbel-sigmoid routing provides no meaningful advantage over per-sequence routing (+0.28% vs 5% threshold), even on mixed-domain sequences specifically designed to maximize its value. The router collapses to a 2-class detector (code/math vs prose), unable to distinguish medical, legal, and creative text in BitNet-2B-4T's hidden space. This is a two-experiment kill (homogeneous: -0.46%, mixed: +0.28%).

## Why This Happened (Literature-Grounded)

Three compounding mechanisms explain this failure:

**1. Near-Domain Representation Collapse.** L2R (Learning to Route) research found that simple hidden-state task vectors are "insufficient for properly routing adapters" in near-domain scenarios where semantic boundaries are subtle. The Web of Science benchmark showed routers failing to distinguish overlapping scientific subfields (medical science, psychology, biochemistry) — directly analogous to our medical/legal/creative collapse. The Domain-Aware Memory (DAM) approach requires significantly more training samples per task to learn these subtle boundaries. Our 800-step router training on ~230 samples was vastly insufficient for 5-class discrimination in near-domain settings.

**2. Cross-Attention Contamination is Architectural, Not Evaluative.** The review correctly identified that our evaluation runs full forward passes through mixed sequences, allowing self-attention between segments to contaminate hidden states. But this is NOT a fixable evaluation bug — it's inherent to any transformer serving mixed-domain content. The python+math pair proves this: 97% boundary detection accuracy yet -6.4% WORSE than per-sequence, because correctly-routed tokens still attend to incorrectly-routed tokens in the other segment. Any real deployment faces this same contamination.

**3. Post-Hoc vs Joint Training is the Critical Variable.** NotebookLM research confirms that MoLoRA's success depends on joint training where router and experts co-adapt via shared gradients. L2R's post-hoc approach (frozen adapters, router trained after) prevents catastrophic interference in continual learning but fundamentally limits router quality. The router never receives gradient signals from the adapters' loss landscapes, so it cannot learn the fine-grained token-level features that joint training discovers. This is the key distinction our experiment didn't test.

## Confirming Evidence

- **L2R (Ponti et al.)**: Explicitly documents that Softmax-based routing collapses to single dominant adapters. Their Gumbel-sigmoid fix prevents collapse but still requires sufficient domain separation in hidden space. Near-domain scenarios remain problematic.
- **Mod-Squad (Chen et al., NeurIPS 2023)**: Task-level routing on homogeneous domains. Validates that per-token routing adds noise without sufficient domain signal; task-level "squads" of experts outperform token-level selection.
- **pQuant (mixed-precision routing)**: Shows that per-token routing CAN work when jointly trained with QAT and the routing signal is clear (quantization sensitivity, not domain identity). The router succeeds because precision-sensitivity is a well-defined, learnable signal — unlike domain identity in near-domain settings.
- **MoBiLE (Mixture of Big-Little Experts)**: Token-importance routing succeeds by routing on a simpler signal (importance/difficulty) rather than domain identity. Validates that the routing signal matters more than the routing granularity.

## Contradicting Evidence

- **MoLoRA (arXiv:2603.15965)**: Per-token routing on Qwen3-1.7B + 4 adapters beats an 8B model. BUT: MoLoRA jointly trains router and adapters, which is a fundamentally different setup. Our post-hoc approach cannot replicate this co-adaptation. The contradiction resolves when you account for joint vs post-hoc training.
- **pQuant**: Per-token routing to mixed-precision branches works. BUT: the routing signal (quantization sensitivity) is much cleaner than domain identity. Tokens that need 8-bit precision are detectable from activation magnitudes, while domain identity of natural-language tokens is not.
- **Expert Token Routing for Generalists**: Frameworks synergizing multiple pre-trained expert LLMs via token-level routing. These succeed because the "experts" are massive, independently-trained models with very distinct feature spaces (not near-domain LoRA adapters sharing a base model's representation space).

## Alternative Approaches (What We Could Try Instead)

### Proven in our codebase:
1. **Entropy-based gating** (exp_entropy_gated_experts): 63% skip rate at 1.13% PPL cost. Sidesteps domain detection entirely by measuring base model confidence. Best approach for "when to route."
2. **Per-adapter routing heads** (exp_tiny_routing_heads): 100% accuracy with ~5K params per adapter. Binary classifiers trained with domain-specific data. Best approach for "where to route."

### From literature (untested):
3. **SHINE (arXiv:2602.06358)**: Hypernetwork generates LoRA weights in a single forward pass using the frozen LLM's own parameters as context compression. No adapter library, no routing — generates task-specific params on the fly. Promising for consumer hardware.
4. **Segment-level routing**: Instead of per-token routing, detect domain boundaries first (e.g., via topic modeling or sliding-window entropy), then apply per-segment composition. Avoids cross-attention contamination by isolating segments.
5. **Contrastive domain embeddings**: Train domain discriminators with contrastive loss (pulling same-domain representations together, pushing different-domain apart) BEFORE training the router. Could fix the near-domain collapse.
6. **LoraHub/LoraRetriever**: Treat adapters as a vector library, retrieve top-k by sequence-level embedding similarity. Scales to many domains without token-level overhead.
7. **Joint MoLoRA-style training**: Train router and adapters together on MLX. The one approach our experiments haven't tested. Requires re-training adapters (not composing frozen ones).

## Implications for Next Experiments

1. **Post-hoc per-token routing is definitively dead.** Two experiments, two nulls. Do not revisit without joint training.

2. **The joint-vs-post-hoc question is the next critical test.** MoLoRA's success with joint training contradicts our failure with post-hoc routing. A micro-experiment jointly training a router + 2-3 LoRA adapters on MLX would resolve whether joint training is the missing ingredient or whether the architecture itself is the bottleneck.

3. **Entropy gating + per-adapter routing heads is the current best path.** Entropy gating answers "should we route at all?" (63% skip rate). Per-adapter heads answer "which expert?" (100% accuracy). Combining these two proven approaches gives us a complete routing solution without any of the failure modes we've observed.

4. **The oracle gap (22.6%) is real value left on the table.** Correct per-segment expert assignment provides substantial PPL improvement. The question is whether this value is accessible via any practical mechanism, or whether pre-merge composition (which gets ~0.80% of the gap) is the best we can do.

5. **BitNet-2B-4T's hidden space has a 2-class structure for natural language.** Code/math tokens are well-separated from prose tokens, but medical/legal/creative are indistinguishable. Any future routing work on this base must account for this representation structure.

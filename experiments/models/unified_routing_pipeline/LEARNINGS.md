# Learnings: exp_unified_routing_pipeline

## Core Finding

Entropy gating and routing heads are incompatible: routing heads improve ALL tokens (including "confident" ones), so skipping routing on any token strictly degrades quality. The two-pass architecture adds 222.7% overhead for zero benefit.

## Why This Happened (Literature-Grounded)

The root cause is that **softmax entropy measures aleatoric uncertainty, not epistemic uncertainty**. A token with low output entropy means the base model's prediction is concentrated, not that the prediction is correct or that an adapter cannot improve it. The uncertainty quantification literature makes this distinction explicit:

1. **Aleatoric vs. epistemic blindness.** Shannon entropy of the softmax distribution captures inherent data ambiguity but is blind to model knowledge gaps. A base model can be highly confident yet wrong, especially on domain-specific tokens where adapters carry specialized knowledge. Our routing heads (which use learned hidden-state classifiers) correctly identify adapter utility on ALL tokens because they measure a different signal — activation-space similarity to domain data, not output-distribution peakedness.

2. **Structural miscalibration.** Standard neural networks produce overconfident predictions, particularly on out-of-distribution data. The softmax function outputs "uncalibrated confidence scores without theoretical grounding" as probabilities. This is well-documented in the Bayesian deep learning literature (Guo et al., 2017; Kristiadi et al., 2020). In our case, BitNet-2B-4T produces low-entropy outputs on many tokens where the adapter still provides measurable PPL improvement.

3. **Near-oracle routing heads eliminate the value proposition.** Our routing heads achieve 0.15% gap to oracle selection (PPL 6.42 vs 6.41). When routing is already near-perfect AND near-free (2.32% overhead), there is no room for entropy gating to add value. The reviewer correctly noted this was logically deducible before running the experiment.

## Confirming Evidence

- **MoBiLE (Mixture of Big-Little Experts)**: Confirms that token importance scoring for expert skipping works only when using activation-level sensitivity, not output entropy. MoBiLE achieves 1.6-1.7x speedup by routing based on internal activation patterns, not softmax confidence.

- **pQuant (mixed-precision routing)**: Successfully uses learned top-1 gating over a 1-bit backbone + 8-bit experts. The router is a tiny linear layer trained end-to-end — it learns routing from activations, not from post-hoc entropy. This is architecturally analogous to our routing heads succeeding where entropy gating fails.

- **Bayesian uncertainty literature**: Deep Ensembles and MC Dropout papers consistently show that mutual information (disagreement across predictions) is required for epistemic uncertainty. Single-forward-pass entropy is insufficient for deciding when additional computation is warranted.

- **EdgeNav-QE**: Early-exit architectures that evaluate intermediate representations succeed at adaptive computation — they use layer-internal signals, not output entropy, confirming the pattern that routing signals must come from inside the model, not from output distributions.

## Contradicting Evidence

Entropy gating DOES work in specific conditions that don't apply to our setup:

- **Diffusion Language Models**: Entropy-sum-based strategies work for DLMs because they control iterative unmasking, not expert selection. The entropy signal indicates "how many tokens still need refinement" — a fundamentally different question than "which expert should handle this token."

- **TriSpec (speculative decoding)**: Confidence gating succeeds in draft-verify architectures because the question is binary (accept/reject draft token) and the cost asymmetry is enormous (draft model is orders of magnitude cheaper). In our setup, pre-merge composition is already nearly free (0.80% overhead), so there's no expensive computation to avoid.

- **MoBiLE with importance scoring**: Shows ~1.7x speedup with negligible quality loss. Key difference: MoBiLE uses activation-level importance, not output entropy. And MoBiLE targets scenarios where expert computation is genuinely expensive (large MoE models), unlike our pre-merge setting.

The discrepancy resolves cleanly: entropy gating works when (a) the gated computation is expensive, (b) the gating signal reflects true routing necessity (not just output confidence), and (c) the base model's confident predictions are actually good enough. Our architecture violates all three conditions.

## Alternative Approaches (What We Could Try Instead)

### 1. Learned Single-Pass Routers (most promising for us)
Our routing heads already work (2.32% overhead, near-oracle). The optimization target is reducing even that small overhead, not replacing the approach. Options:
- **Amortized routing**: Cache routing decisions per-sequence prefix rather than per-token
- **L2R-style Gumbel-sigmoid routing**: Sequence-level router using [CLS] token, non-competing multi-adapter activation. Already on our research agenda.

### 2. Similarity Retrieval (LoraHub / LoraRetriever)
Embed the input query, retrieve top-k adapters from a library, fuse via linear interpolation. Bypasses neural routing entirely. Scales to thousands of domains. Relevant if we move to many-adapter settings.

### 3. Hypernetwork Generation (SHINE)
Generate optimal LoRA weights directly from input context in a single forward pass via a Memory-to-Parameter transformer. Eliminates routing entirely — the adapter IS the routing decision. Relevant to our Text-to-LoRA research direction (SakanaAI).

### 4. Early-Exit with Adapter Injection
Instead of two-pass (base → entropy → compose), use early-exit branches that trigger adapter composition at intermediate layers. The model decides at layer L whether to inject the adapter at layer L+1. This reuses the single forward pass and avoids the 2x cost.

### 5. Pre-merge with Periodic Re-routing
Since pre-merge is free (0.80%), keep the current always-route approach but re-evaluate routing decisions only every N sequences (batch-level routing). This amortizes routing head cost over many sequences.

## Implications for Next Experiments

1. **Entropy gating line is permanently closed.** No variant of output-entropy-based gating can beat always-route when routing is near-free and near-oracle. Don't revisit.

2. **Routing heads ARE the answer for adapter selection.** Focus on making them even cheaper (amortization, caching) rather than replacing them with entropy signals.

3. **Pre-merge composition remains the dominant serving strategy.** The 0.80% overhead is so low that any gating mechanism adds more cost than it saves. The only reason to gate is if N (number of adapters) grows large enough that routing head evaluation becomes expensive.

4. **For future adaptive compute work, use activation-space signals, not output entropy.** The literature consistently shows that internal representations (hidden states, attention patterns) are better routing features than output distributions.

5. **The hypothesis was logically falsifiable pre-experiment.** The reviewer flagged this: if routing heads are near-oracle (0.15% gap), then by definition they help on ALL tokens, making entropy-skip counterproductive. Future experiments should include a "pre-mortem logical consistency check" against prior results before committing compute.

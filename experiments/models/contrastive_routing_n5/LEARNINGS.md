# LEARNINGS: Text Classification Routing for SFT Adapters (exp_contrastive_routing_n5)

## Core Finding

TF-IDF + logistic regression routing achieves 90% accuracy on 5 SFT domains where energy gap routing achieves only 36%, validating that **routing should use input features, not adapter NLL**. The real discovery: correct routing *exposes* adapter quality problems that misrouting accidentally masked. Routing is solved for N=5 well-separated domains; adapter quality is now the bottleneck.

## Why This Happened

Energy gap routing (Finding #205) failed because SFT training makes one adapter (code) universally dominant in NLL space. The code adapter genuinely improves structured generation for all domains, so any NLL-based metric selects it regardless of domain. This is not a routing architecture failure --- it's a signal failure. NLL conflates "generates well" with "matches this domain."

TF-IDF succeeds because domain identity is carried by vocabulary, not by generation quality. Medical queries contain "patient," "diagnosis"; code queries contain "function," "variable." These signals are trivially separable by a linear classifier and are by construction independent of which adapter generates best (Design Principle 2 in MATH.md).

The 90% accuracy with legal-finance confusion (70-80%) matches the known vocabulary overlap between these domains (contracts, regulations appear in both). This is expected behavior, not a limitation of the approach per se --- it reflects genuine domain proximity.

## Confirming Evidence

- **LoRAuter** (arXiv:2601.21795): Task-embedding-based routing achieves oracle-matching performance (101.2%) at 1500+ adapter scale by routing via task representations rather than adapter characteristics. Confirms that input/task-level routing is strictly better than adapter-output-based routing. Our TF-IDF classifier is the simplest instance of this principle.
- **LoraRetriever** (arXiv:2402.09997): Input-aware LoRA retrieval and composition for mixed tasks. Validates that identifying relevant adapters from input prompts works in the wild. Our experiment confirms this at micro scale with the simplest possible implementation.
- **Task-Aware LoRA Composition** (arXiv:2602.21222): VDB-based retrieval pushes expert selection outside the forward pass, allowing adapter libraries to scale independently of GPU memory. Confirms the architectural principle that routing should be decoupled from inference.

## Contradicting Evidence

- **No direct contradictions found** for input-based routing vs NLL-based routing. The literature uniformly supports input-feature routing over generation-quality-based routing.
- **However**, NeuroLoRA (arXiv:2603.12378) argues that static routing mechanisms (which TF-IDF is) are "agnostic to input context" in a deeper sense --- they don't adapt to the *model's internal state* during processing. A TF-IDF classifier routes based on surface vocabulary, not semantic understanding. For fuzzy domain boundaries or ambiguous queries, this distinction matters.
- **LoRA-Mixer** (arXiv:2507.00029) uses serial attention routing over expert representations, suggesting that for more complex routing scenarios, attention-based mechanisms outperform simple classification. Our N=5 setting may be too easy to reveal this gap.

## Alternative Approaches

1. **LoRAuter task embeddings** (arXiv:2601.21795): Training-free, scales to 1500+ adapters, uses small validation sets to create task catalog. The natural upgrade path from TF-IDF when N grows or domains become fuzzy. No retraining needed when adapters are added.
2. **MoLoRA per-token routing** (arXiv:2603.15965): Learned router selects adapter per-token, not per-query. Qwen3-1.7B + 4 adapters outperforms 8B model. More fine-grained than our per-query TF-IDF but requires joint training.
3. **Text-to-LoRA** (arXiv:2506.06105): Hypernetwork generates task-specific LoRA in a single forward pass from a text description. Bypasses routing entirely --- instead of selecting from a fixed pool, synthesizes the adapter. Radical alternative but unproven at our scale.
4. **MoE-Sieve** (arXiv:2603.24044): Routing-guided LoRA with sieve mechanism learns routing jointly with adapter training. Could fix the adapter quality problem and routing problem simultaneously.
5. **NeuroLoRA contrastive orthogonality** (arXiv:2603.12378): Enforces expert subspace separation via contrastive loss during training. Would make adapters more distinguishable, potentially improving routing accuracy for confused pairs (legal-finance).

## Implications for Next Experiments

1. **Routing is solved for N=5 --- stop optimizing routing.** The 90% accuracy with a textbook classifier proves domain identity is trivially available in input text. Further routing experiments at N=5 are wasted effort.

2. **Adapter quality is the real bottleneck.** 3/5 domains degrade with correct routing (K607 FAIL, though metric is unreliable). This was masked by energy gap misrouting to the universally-good code adapter. The project must now answer: are prose adapters genuinely bad, or is the keyword density metric misleading?

3. **The code adapter as universal improver is the most actionable finding.** Code adapter improves math from 10% to 70% correctness (Finding #204) and the math adapter improves it further to 80%. This suggests instruction-following capability transfers across domains for structured tasks.

4. **Scaling from N=5 to N=24 requires a different router.** TF-IDF works for 5 well-separated domains but will degrade with vocabulary overlap at N=24 (14 domains had a neighbor marginally better than own adapter per Finding #202). LoRAuter's task-embedding approach (arXiv:2601.21795) is the natural upgrade path.

5. **PPL evaluation remains nearly useless for routing.** DDR=1.13 (Finding #202) means wrong adapters capture 87% of PPL benefit. Routing accuracy differences only matter for behavioral outcomes, not PPL. All future experiments must use behavioral metrics.

## Recommended Follow-Up

1. **exp_generation_quality_test (P0)**: Test whether correctly-routed composition produces better *text* than base model, using a reliable behavioral metric (LLM-as-judge). Motivated by: K607 inconclusive due to unreliable keyword density metric (Finding #179). The existential question remains unanswered.

2. **exp_task_accuracy_real_benchmarks (P0)**: MMLU/GSM8K/HumanEval with routed composition. Motivated by: PPL doesn't predict task quality (r=0.08, Finding #203). Only task-level accuracy reveals whether adapters help.

3. **exp_lora_scale_ablation**: Test if lora_scale=20.0 is too aggressive for prose domains. Motivated by: 3/5 prose domains degrade, but the same adapters improve PPL by 35%. The behavioral degradation could be a scaling issue, not an adapter quality issue.

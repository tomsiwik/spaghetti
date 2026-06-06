# LEARNINGS: Input-Dependent Adapter Scaling (KILLED — TF-IDF Mechanism)

## Core Finding

**TF-IDF cosine similarity has zero predictive power for per-query adapter effectiveness
(r=-0.079 overall, per-domain: math -0.009, code 0.032, medical -0.093).** The specific
mechanism is killed: lexical similarity to domain centroids cannot predict representation-space
adapter quality. The broader direction (input-dependent scaling) remains open but requires
LEARNED routing functions, not fixed feature spaces.

## Why This Happened

### 1. Representation Space Mismatch (Primary Cause)

TF-IDF operates in a bag-of-words space where similarity is lexical overlap. Adapter
effectiveness depends on the model's internal representation space — whether B@A@x is
well-calibrated for query x's hidden-state trajectory through the transformer layers.

**LoRAuter (arXiv:2601.21795)** directly addresses this gap: instead of using raw lexical
features, it routes via *task representations* — embeddings derived from small validation sets
that capture what the model needs, not what words appear. LoRAuter achieves 101.2% of Oracle
performance in-domain and outperforms LoraRetriever by 5.2pp out-of-domain, precisely because
it maps through the model's representation space rather than an external feature space.

The fundamental issue: TF-IDF captures topic (what the query is *about*), not adapter
compatibility (whether the adapter's learned B@A transformation is useful for this query's
internal representation). A math query with unusual wording has low TF-IDF similarity but
may still benefit fully from the math adapter because the model's internal representations
align. This is a category error, not a calibration problem.

### 2. Bag-of-Words Cannot Capture Compositional Semantics

TF-IDF treats terms independently — "solve this equation" and "equation this solve" produce
identical embeddings. But adapter effectiveness depends on compositional meaning: a code query
asking to "implement a sorting algorithm" needs different adapter engagement than one asking
to "explain how sorting works." Word2Vec and sentence transformers partially address this,
but the fundamental lesson is that *any* fixed embedding space that was not trained to predict
adapter effectiveness will fail. The mapping must be learned end-to-end.

### 3. Per-Token vs Per-Query Granularity

Our approach computed a single scale per query. **MoLoRA (arXiv:2603.15965)** demonstrates
that per-*token* routing is superior — Qwen3-1.7B with MoLoRA exceeds Qwen3-8B on reasoning
benchmarks (4.7x smaller). Different tokens within the same query may need different adapter
engagement. Our code prompt 9 (degenerate repetition at s=20, clean at s=6) likely had
some tokens that needed high adapter engagement while others needed suppression — a single
per-query scale cannot capture this.

## Confirming Evidence

- **LoRAuter (arXiv:2601.21795):** Routes via learned task representations, not lexical
  features. Confirms that the mapping from query to adapter weight must go through the model's
  representation space. Achieves near-Oracle performance where fixed features fail.

- **Task-Aware LoRA Composition (arXiv:2602.21222):** Uses vector database retrieval with
  trained embeddings from 22 datasets. Retrieval-weighted fusion dynamically merges adapters
  based on learned similarity, not TF-IDF distance. Nucleus sampling over retrieval scores
  produces soft adapter mixtures.

- **LD-MoLE (arXiv:2509.25684):** Learnable dynamic routing for mixture of LoRA experts.
  Router is trained via gradient descent to predict which expert combination produces the
  best output. Routing signal is learned, not derived from external features.

- **Finding #247 (this project):** TF-IDF routing achieves 90% domain classification
  accuracy — TF-IDF is good at *topic identification* (which domain?) but not at
  *effectiveness prediction* (how much of this adapter?). This is precisely the distinction:
  TF-IDF can route, but cannot scale.

- **Finding #252 (this project):** Code shows bimodal per-prompt behavior — some prompts
  improve with adapter, others degrade. This within-domain variance is exactly what per-query
  scaling should address, but TF-IDF similarity cannot distinguish the two groups because
  both have similar lexical profiles.

## Contradicting Evidence

- **"Why TF-IDF Still Beats Embeddings" (various analyses):** In some retrieval tasks, TF-IDF
  outperforms learned embeddings. However, these are document *retrieval* tasks (matching query
  to document), not *effectiveness prediction* tasks (predicting how well an adapter will
  perform on this query). The tasks are fundamentally different.

- **Simple baselines sometimes win:** There exists literature showing that simple features
  can outperform complex learned ones when data is limited. With n=10 per domain, it is
  possible (though unlikely given three independent near-zero correlations) that TF-IDF would
  show predictive power at larger n. However, the architectural argument (lexical space ≠
  representation space) suggests this is a category error, not a sample size problem.

## Alternative Approaches (with paper references)

1. **LoRAuter task-representation routing (arXiv:2601.21795):** Routes via semantic task
   clusters derived from validation sets, not lexical features. Adapters are composed using
   weighted output-space fusion reflecting input-task similarity. Scales to 1500+ adapters.
   **Most directly applicable** — replaces our TF-IDF centroids with learned task embeddings.

2. **MoLoRA per-token routing (arXiv:2603.15965):** Per-token adapter selection via learned
   gating. Enables within-sequence specialization (e.g., code tokens get code adapter, math
   tokens get math adapter in mixed queries). Demonstrated 4.7x size efficiency.
   **Architecturally optimal** — addresses both the "which adapter" and "how much" questions
   at token granularity.

3. **LD-MoLE learnable dynamic routing (arXiv:2509.25684):** Learnable router trained with
   gradient signal from task loss. Router parameters are updated jointly with adapter
   parameters. **Training-integrated** — the routing signal comes from generation quality,
   not external features.

4. **CLONE MoE router for edge (arXiv:2506.02847):** Integrates MoE router for dynamic LoRA
   selection at edge inference. Includes hot-swap via LoRA Processing Unit. **Deployment-ready
   pattern** for our M5 Pro target.

5. **MoE-Sieve routing-guided LoRA (arXiv:2603.24044):** Uses pre-trained MoE router signals
   to guide LoRA fine-tuning. Leverages existing routing infrastructure rather than training
   new routers. **Low-cost alternative** if MoE routing signals are available.

## Implications for Next Experiments

### Nine-Level Proxy Chain (Complete)

1. PPL → MMLU (r=0.08, Finding #236)
2. MMLU → behavioral (Finding #238)
3. PPL improvement → specialization (Finding #240)
4. Cosine → functional disagreement (Finding #240)
5. Domain classification → composition quality (Finding #243)
6. Adapter orthogonality → contrastive value (Finding #245)
7. PPL-optimal scale → behavioral-optimal scale (Finding #249)
8. Math phase transition → universal phase transition (Finding #252)
9. **Lexical similarity → adapter effectiveness (Finding #253)**

**Meta-lesson:** Every proxy metric tested has failed to predict the next level. The nine-level
chain proves that only behavioral evaluation on actual generation tells you anything. All
intermediate metrics — PPL, cosine, classification accuracy, orthogonality, TF-IDF similarity —
are unreliable proxies for "does this produce useful text?"

### Architectural Implications

- **The router architecture must be LEARNED, not engineered.** TF-IDF was the last "cheap
  feature" hypothesis. LoRAuter/MoLoRA/LD-MoLE all use gradient-trained routing.
- **Per-token granularity is likely necessary.** Code prompt 9's degenerate behavior at s=20
  but clean output at s=6 suggests token-level variation. MoLoRA's per-token routing directly
  addresses this.
- **The training signal must be generation quality.** LD-MoLE trains routers via task loss.
  Our routing heads (Finding #247) achieve 90% classification accuracy but classification ≠
  effectiveness. The router needs a *quality* signal, not a *topic* signal.

## Recommended Follow-Up (Priority Order)

1. **LoRAuter-style task-representation routing** — Motivation: Finding #253 kills TF-IDF;
   LoRAuter (arXiv:2601.21795) shows learned task representations achieve near-Oracle routing.
   Replace TF-IDF centroids with small validation-set-derived task embeddings. Test whether
   the learned similarity predicts behavioral quality (r > 0.3 threshold from this experiment).
   **Feasible on M5 Pro:** LoRAuter's routing is inference-only (no router training needed if
   we use their retrieval approach).

2. **MoLoRA per-token composition** — Motivation: MoLoRA (arXiv:2603.15965) achieves 4.7x
   size efficiency via per-token routing. Code prompt 9's degenerate repetition at s=20
   suggests per-token variation matters. This is the P0 deployment track's next logical step:
   train a gating network that routes tokens to adapters with learned scaling.
   **Higher cost:** Requires training a gating network.

3. **Generation quality existential test (P0 track)** — Motivation: exp_generation_quality_perscale
   already showed math generation works. But nine proxy chain failures prove metrics are
   unreliable. The real question remains: does the FULL pipeline (route → compose → generate)
   produce text that a human would prefer over base model output? This should use LLM-as-judge
   at scale, not proxy metrics.

## References

- LoRAuter: Effective LoRA Adapter Routing using Task Representations (arXiv:2601.21795)
- Task-Aware LoRA Adapter Composition via Similarity Retrieval (arXiv:2602.21222)
- MoLoRA: Composable Specialization via Per-Token Adapter Routing (arXiv:2603.15965)
- LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts (arXiv:2509.25684)
- CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge (arXiv:2506.02847)
- MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning (arXiv:2603.24044)
- Schaeffer et al. 2023, Are Emergent Abilities a Mirage? (arXiv:2304.15004)
- Findings #236, #238, #240, #243, #245, #247, #249, #250, #252, #253 (this project)

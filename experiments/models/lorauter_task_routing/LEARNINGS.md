# LEARNINGS: LoRAuter Task-Representation Routing (SUPPORTED)

## Core Finding

**Sentence-embedding centroid routing solves domain identification at 96% accuracy (vs TF-IDF
90%) and delivers 99.7% of oracle behavioral quality, but embedding similarity does NOT predict
per-query adapter effectiveness (r=0.234, p=0.103).** This reveals a two-signal decomposition:
domain routing is solved, effectiveness prediction is structurally unsolved by any embedding
space tested so far.

## Why This Happened

### 1. Domain Identity Lives in Lexical/Semantic Space; Effectiveness Lives in Representation Space

Sentence embeddings (all-MiniLM-L6-v2, 384-dim) are trained via contrastive learning to group
semantically similar texts. This directly captures "what domain is this query about?" — the
same signal TF-IDF captures (Finding #247: 90% accuracy) but with richer semantic features
that resolve hard cases (legal vs finance: 90% vs TF-IDF's 75% average on these two domains).

But adapter effectiveness — whether B@A@x is well-calibrated for query x's hidden-state
trajectory — depends on the model's internal representation space, not the input embedding
space. The sentence transformer was never trained to predict how well a LoRA perturbation
will work. It was trained to predict whether two texts are semantically similar. These are
fundamentally different tasks.

This is the same structural gap as Finding #253 (TF-IDF r=-0.079), just narrower: sentence
embeddings improve from r=-0.079 to r=0.234, but this marginal improvement is driven entirely
by better domain identification (96% vs 90%), not by predicting within-domain effectiveness.

### 2. The 99.7% Oracle Figure Reveals That Effectiveness Prediction Doesn't Matter (at 5 Domains)

The adversarial review correctly identified that 99.7% oracle is a restatement of 96% routing
accuracy: 48/50 correct adapter selections. But this circularity contains the key insight:
**at 5 well-separated domains, correct adapter SELECTION is sufficient.** The per-query
effectiveness gap (r=0.234) doesn't cause behavioral degradation because the wrong adapter is
worse than no adapter in most cases — so correct top-1 selection captures nearly all the value.

This changes at scale: with 50+ overlapping domains, multiple adapters may be beneficial,
and the DEGREE of benefit matters for weighting. That's where effectiveness prediction becomes
load-bearing.

### 3. The Inter-Centroid Similarity Matrix Predicts Routing Failures

Legal-finance inter-centroid similarity of 0.825 explains why both domains achieve only 90%
routing accuracy (misroutes go between them). Math has the lowest inter-centroid similarities
(0.538-0.700), enabling 100% accuracy. This follows directly from nearest-centroid classification
theory (Cover & Hart, 1967): error rate is governed by the ratio of inter-class to intra-class
variance (Fisher discriminant ratio = 5.61 overall, but lower for legal-finance pair).

## Confirming Evidence

- **LoRAuter (arXiv:2601.21795):** The direct inspiration. Achieves 101.2% oracle on Llama2-7B
  with 48 tasks using cosine-similarity routing to sentence-embedding centroids. Our 96% on
  BitNet-2B-4T with 5 domains confirms the method transfers to ternary architectures. The
  paper does NOT report per-query effectiveness correlation (their metric is aggregate oracle %),
  so the r=0.234 gap may also exist in their setup but was masked by the aggregate metric.

- **Task-Aware LoRA Composition (arXiv:2602.21222):** Uses vector DB retrieval with trained
  embeddings from 22 datasets. Retrieval-weighted fusion dynamically merges adapters based on
  learned similarity. Confirms that embedding-based routing works for adapter selection, but
  uses a more sophisticated weighting scheme (nucleus sampling over retrieval scores) that
  partially addresses the effectiveness prediction gap.

- **CLONE (arXiv:2506.02847):** MoE router for dynamic LoRA selection at edge. Uses a learned
  router trained on adapter performance, not just embedding similarity. Confirms that the
  next step from centroid matching is a LEARNED router with a performance-aware training signal.

- **Finding #247 (our own):** TF-IDF achieves 90% domain routing. Sentence embeddings achieve
  96%. The 6pp improvement is concentrated on the hard domains (legal +10pp, finance +20pp),
  confirming that semantic features resolve ambiguity where bag-of-words cannot.

## Contradicting Evidence

- **PHATGOOSE (arXiv:2402.05859):** Uses per-module routing gates trained on each adapter's
  gradient information. Routes at the MODULE level (different adapters for different layers),
  not at the QUERY level. Achieves strong multi-task results, suggesting that query-level
  routing may be insufficient — different queries may need different adapters at different
  layers. Our experiment uses query-level top-1 routing which cannot capture this.

- **MoLoRA (arXiv:2603.15965):** Per-TOKEN routing (Qwen3-1.7B > Qwen3-8B). Demonstrates
  that per-query granularity is too coarse — different tokens within the same query benefit
  from different adapter combinations. Our r=0.234 may partly reflect this granularity
  mismatch: a query-level embedding cannot predict token-level adapter utility.

## Alternative Approaches

1. **Learned Router with Performance Signal (CLONE, LD-MoLE):** Instead of unsupervised
   centroid matching, train a router on actual adapter performance data. LD-MoLE
   (arXiv:2509.25684) uses task loss gradients to train learnable dynamic routing.
   This directly addresses the effectiveness prediction gap by learning the mapping
   from query representation to adapter utility end-to-end.

2. **Per-Token Routing (MoLoRA, arXiv:2603.15965):** Route at token granularity instead
   of query granularity. MoLoRA's Qwen3-1.7B with 4 adapters exceeds Qwen3-8B on reasoning,
   demonstrating that within-query variation is a real signal. This replaces the
   effectiveness prediction problem with a learned per-token mixing problem.

3. **Module-Level Routing (PHATGOOSE, arXiv:2402.05859):** Route different adapters to
   different transformer layers. Each adapter's gradient-trained gate determines which
   layers it should activate for. This captures structural variation that query-level
   routing misses.

4. **MoE-Sieve (arXiv:2603.24044):** Uses pre-trained MoE router signals to guide LoRA
   fine-tuning. Repurposes existing MoE routing infrastructure rather than training
   separate routing mechanisms. Not directly applicable to BitNet-2B-4T (not MoE), but
   the principle of reusing internal model signals for routing is relevant.

## Implications for Next Experiments

### The Nine-Level Proxy Chain Is Now Complete

This experiment adds the ninth level:

| # | Proxy | Predicts | Result | Finding |
|---|-------|----------|--------|---------|
| 1 | PPL | MMLU | r=0.08 | #236 |
| 2 | MMLU | Behavioral | Fails | #238 |
| 3 | PPL improvement | Specialization | Fails | #240 |
| 4 | Cosine similarity | Functional disagreement | Fails | #240 |
| 5 | Domain classification | Composition quality | Partial | #243 |
| 6 | Adapter orthogonality | Contrastive value | Fails | #245 |
| 7 | PPL-optimal scale | Behavioral-optimal scale | Fails | #249 |
| 8 | Math phase transition | Universal phase transition | Fails | #252 |
| 9 | Embedding similarity | Adapter effectiveness | r=0.234 | #255 |

**Meta-lesson:** Every proxy metric tested has failed to predict the next level in the
chain. Only direct behavioral evaluation on generation tasks is reliable. However, the
TWO-SIGNAL DECOMPOSITION shows this is not entirely bleak: domain SELECTION is trivially
solved (96%), and effectiveness prediction may not be needed if selection quality is high
enough.

### For the P0 Deployment Track

Routing is solved for 5 well-separated domains:
- Sentence-embedding centroid routing: 96% accuracy, 0% incoherent output
- Zero additional parameters (centroids are precomputed from validation data)
- ~1ms per-query overhead (sentence-transformer encoding + 5 cosine similarities)

The open question is SCALING: will this hold at 25 domains (P0 target) or 50+ domains?
The legal-finance centroid overlap (cos=0.825) suggests degradation is likely with more
overlapping domains. The PHATGOOSE module-level approach may be needed at scale.

## Recommended Follow-Up

1. **Centroid routing at N=24 (exp_embedding_routing_n24, already in backlog):**
   Test whether 96% accuracy degrades with 24 overlapping domains. If it drops below 80%,
   switch to learned routing (CLONE/LD-MoLE approach).
   *Motivation:* Finding #255 shows 96% at N=5 but inter-centroid overlap (0.825) predicts
   degradation. *Literature:* LoRAuter tested on 48 tasks with broader separation.

2. **MoLoRA per-token composition (if centroid routing holds at N=24):**
   Replace per-query routing with per-token adapter mixing. The r=0.234 effectiveness gap
   may be a granularity artifact — per-token routing addresses within-query variation.
   *Motivation:* MoLoRA (arXiv:2603.15965) shows per-token routing enables 1.7B > 8B.
   *Literature:* MoLoRA, PHATGOOSE (arXiv:2402.05859).

3. **Generation quality existential test (P0 track, highest priority):**
   Use LLM-as-judge at scale to answer THE question: does routed composition produce
   BETTER TEXT than base? All routing work is moot if the composed output isn't useful.
   *Motivation:* Nine-level proxy chain shows metrics don't predict behavioral quality.
   Only direct evaluation on generation tasks is reliable.

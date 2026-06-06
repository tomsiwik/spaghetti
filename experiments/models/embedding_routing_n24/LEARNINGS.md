# LEARNINGS: Embedding Routing at N=24 (KILLED)

## Core Finding
Raw embedding-layer mean-pooling produces near-identical centroids (cos 0.986) for all 24 domains because Zipf-distributed common words dominate the mean, making domain-distinctive vocabulary invisible. Transformer layers ADD discriminative signal (cos 0.716), refuting the "contextual mixing destroys domain signal" hypothesis. TF-IDF (35%) beats neural embeddings (28.3%) because IDF naturally downweights shared vocabulary. Sixth routing kill at N=24.

## Why This Happened

**The disease: Zipf's law + mean-pooling = centroid collapse.**

In any natural language corpus, a small set of high-frequency words (articles, prepositions, pronouns) account for the vast majority of tokens. When mean-pooling over all tokens in an embedding lookup table, these shared words dominate the average, collapsing all domain centroids toward a single point. Domain-distinctive vocabulary (e.g., "plaintiff" for legal, "photosynthesis" for science) constitutes a tiny fraction of total tokens and cannot overcome the shared-vocabulary mass.

This is a structural property of autoregressive LM embeddings. The embedding layer is a lookup table with uniform weighting per token occurrence -- it has no mechanism equivalent to IDF weighting to suppress common terms. LoRAuter (arxiv 2601.21795) explicitly avoids this by using a SupCon-trained sentence encoder that compresses semantic content into discriminative embeddings, not raw token lookups.

**Why hidden states are better:** Transformer layers perform contextual mixing that, contrary to our hypothesis, CREATES domain-discriminative signal. Self-attention allows the model to weight informative tokens over uninformative ones -- a learned attention-pooling that implicitly solves the Zipf-domination problem. Hidden-state mean cosine (0.716) is dramatically lower than embedding mean cosine (0.986), confirming that transformer processing improves domain separability by 27 percentage points.

**The MATH.md failure:** Theorem 1 was a proof sketch with an unbounded shared-vocabulary assumption. The critical condition p_k(w) ~ p_j(w) for shared words was stated as an approximation without bounding the residual. Under Zipf's law, this residual dominates, making the bound vacuous. The LoRAuter citation was misleading -- it was cited as support when its conditions (SupCon training) were explicitly violated.

## Confirming Evidence

- **Representational homogeneity in MoE** (NotebookLM): Expert representations can converge to >99% similarity, the same "centroid collapse" pattern we observe. Standard mitigation: auxiliary load-balancing losses (not applicable to our centroid routing).
- **Pooling and Attention (arxiv 2409.02727)**: Systematic study of LLM-based embedding pooling strategies finds no one-size-fits-all solution. Mean-pooling over last-layer hidden states outperforms EOS-token pooling on average, but Multi-Layer Trainable Pooling (cross-attention over ALL hidden layers) is statistically superior for similarity/retrieval. This confirms mean-pooling is suboptimal.
- **vLLM Semantic Router** (blog.vllm.ai, Oct 2025): Production semantic routing at scale uses LoRA-adapted classification with shared base model computation, NOT raw embedding centroids. Their Signal-Decision Architecture scales from 14 fixed categories to unlimited routing via multi-dimensional signal extraction.
- **SOLE experiment history** (Findings #189, #191, #192): Five prior routing kills at N=24 all hit the same ~40% ceiling with mean-pooled representations, confirming this is a representation quality bottleneck, not an architecture choice.

## Contradicting Evidence

- **LoRAuter (arxiv 2601.21795)** scales to 1500+ adapters using mean-pooled embeddings from a SupCon-trained sentence encoder, achieving oracle-level routing (101.2%). Mean-pooling CAN work, but ONLY when the encoder is explicitly trained with contrastive loss to produce discriminative embeddings. Raw LM embeddings lack this training.
- **Task-Aware LoRA Adapter Composition (Adsul et al. 2026, NotebookLM)**: Uses frozen all-MiniLM-L6-v2 embeddings (no fine-tuning) to route across 22 datasets / 6 task families. MiniLM is a distilled sentence encoder, not a raw LM embedding layer -- but shows that a well-pretrained encoder works without contrastive fine-tuning on the specific routing task.
- **Evaluating Embedding Generalization (arxiv 2511.21703)**: Studies how LoRA and SLERP shape representational geometry. Suggests SLERP model-merging can mitigate over-specialization, implying embedding geometry is more nuanced than our centroid collapse implies.

## Alternative Approaches (Paper-Backed)

1. **LoRAuter-style external sentence encoder** (arxiv 2601.21795): Proven at N=48 and 1500+ noisy adapters. Uses mean-pooled SupCon-trained embeddings. Requires loading a second model (~22M params for MiniLM). Strongest evidence base.

2. **Per-token learned routing / MoLoRA** (arxiv 2603.15965): Per-token routing with 1.7B base beats 8B single model. Avoids sequence-level mean-pooling entirely. Higher computational cost O(NL) per layer.

3. **Multi-Layer Trainable Pooling** (arxiv 2409.02727): Cross-attention over all hidden layers rather than mean-pooling the last layer. Could replace our mean-pooling bottleneck with a learned attention mechanism. Requires training.

4. **Hierarchical routing / MixER** (NotebookLM): Coarse-to-fine multi-stage routing with K-means clustering. Reduces effective N per routing decision. Compatible with our proven N=5 mechanisms.

5. **vLLM Signal-Decision Architecture** (blog.vllm.ai, Nov 2025): Multi-dimensional signal extraction + flexible decision logic. Production-proven for unlimited routing categories. Uses LoRA-adapted classifiers, not raw embeddings.

6. **LoRA-LEGO rank-wise clustering** (NotebookLM): Disassembles adapters into Minimal Semantic Units, clusters by rank. Avoids per-query routing entirely -- merges at the parameter level.

## Implications for Next Experiments

1. **Mean-pooling over raw LM features is a dead end for N>10.** Six kills confirm this. The ~40% ceiling is Zipf-law-driven centroid collapse, not an architecture or layer-selection issue.

2. **Transformer layers help, not hurt.** The "contextual mixing destroys signal" hypothesis is refuted. Future routing should use transformer outputs, not bypass them.

3. **IDF-like weighting is the minimum viable fix.** TF-IDF (35%) beats all neural embedding methods because it solves the exact problem (common-word domination). Any future embedding approach needs this property built in.

4. **The LoRAuter path is the strongest next candidate.** It uses a SupCon-trained encoder that inherently has IDF-like discriminative properties from contrastive training. Proven at 1500+ adapters. The only question is whether loading a second model (~22M) fits the M5 Pro deployment constraint (it should -- MiniLM-L6-v2 is tiny).

5. **Hierarchical routing deserves testing.** Cluster 24 domains into ~5 groups (our proven N=5 mechanisms work perfectly), then route within groups. This sidesteps the N>10 scaling problem entirely.

## Recommended Follow-Up

**Option A: LoRAuter-style external encoder routing** (STRONGEST)
- Motivation: LoRAuter (arxiv 2601.21795) achieves oracle routing at N=48 and 1500+ adapters
- Literature: arxiv 2601.21795 (proven), all-MiniLM-L6-v2 (22M params, fits M5 Pro easily)
- Why it fixes the failure: Contrastive-trained encoder produces discriminative embeddings by design, bypassing Zipf-driven centroid collapse
- Risk: Requires loading a second model alongside BitNet-2B-4T; may add latency

**Option B: Hierarchical domain clustering + proven N=5 routing**
- Motivation: Finding #179 (100% routing accuracy at N=5), Finding #189 (8.3% at N=24 = scaling failure)
- Literature: MixER (NotebookLM, coarse-to-fine routing), HMoRA (ICLR 2025, hierarchical token+task routing)
- Why it fixes the failure: Reduces effective N per routing decision to ~5, where our existing mechanisms are proven
- Risk: Requires discovering natural domain clusters; cluster assignment may be as hard as full routing

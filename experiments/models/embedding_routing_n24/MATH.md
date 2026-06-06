# MATH.md: Embedding-Based Routing at N=24

## Type: Frontier Extension
**Proven result being extended:** LoRAuter (arxiv 2601.21795) proves that routing via cosine similarity between query embeddings and task-centroid embeddings achieves oracle-level performance (101.2%) at 1500+ adapters, using a dedicated sentence embedding model.

**Gap:** LoRAuter uses an external sentence encoder (SupCon-trained). We test whether the base LLM's OWN embedding layer provides sufficient domain signal for routing at N=24, without any external model.

**New math needed:** Bound on domain separability in the base model's embedding space as a function of vocabulary overlap between domains.

## A. Failure Mode Identification

**Disease:** Mean-pooled hidden states from transformer output lack domain signal for 18/24 domains, creating a ~40% accuracy ceiling regardless of routing architecture (Finding #192/193).

**Root cause (not symptom):** Transformer layers perform contextual mixing across all token positions. By the final layer, domain-specific lexical signal has been absorbed into distributed representations optimized for next-token prediction, not domain discrimination. Mean-pooling over these contextually-mixed representations further destroys any residual domain signal.

**Why this is a stable failure mode:** The transformer is trained to predict the next token. For overlapping domains (cooking/marketing, education/code), the output hidden states converge to similar distributions because the prediction task is similar. The routing head receives representations where 18 domains are already mapped to overlapping regions.

## B. The Right Question (Reframe)

**Wrong question:** "How do we build a better classifier on top of mean-pooled hidden states?"

**Right question:** "At what representation layer is domain identity maximally preserved, and can we route there instead?"

**Answer from information theory:** The embedding layer is a lookup table mapping token IDs to vectors. No contextual mixing has occurred. Domain-specific vocabulary (e.g., "plaintiff" for legal, "photosynthesis" for science, "quarter-back" for sports) maps to UNIQUE embedding vectors. The embedding-space representation preserves the full lexical identity that transformer layers then distribute.

## C. Prior Mathematical Foundations

**Theorem (LoRAuter, 2601.21795):** For a set of tasks T = {t_1, ..., t_K} with centroid embeddings {c_1, ..., c_K} computed from m validation samples each, and a sentence encoder E trained with SupCon loss, the cosine similarity between a query embedding E(x) and the correct task centroid c_k achieves top-1 retrieval accuracy matching oracle performance when:
1. Tasks have distinct semantic signatures in embedding space
2. Intra-task variance < inter-task distance (cluster separability)

**Theorem (Johnson-Lindenstrauss, 1984):** Random projections from R^V to R^d preserve pairwise distances with distortion at most (1 +/- epsilon) when d >= O(log(K)/epsilon^2). The embedding layer is such a projection from V-dimensional one-hot space to d-dimensional embedding space.

**Fact (Mean embedding as sufficient statistic):** For a bag-of-words model with i.i.d. token draws from domain-specific distributions p_k, the mean embedding converges to the expectation E_k = E_{w~p_k}[emb(w)]. By the law of large numbers, with T tokens: ||mean_emb - E_k|| = O(sigma_k / sqrt(T)) where sigma_k is the intra-domain embedding variance.

## D. Proof of Guarantee (Conditional)

**Theorem 1 (Centroid Separability).** Let emb: V -> R^d be the base model's embedding layer. For K domains with token distributions p_1, ..., p_K, define centroid c_k = E_{w~p_k}[emb(w)]. If domains k and j have vocabulary overlap ratio alpha_{kj} = sum_w min(p_k(w), p_j(w)), then:

||c_k - c_j|| >= ||E_k^{unique} - E_j^{unique}|| * (1 - alpha_{kj})

where E_k^{unique} = sum_{w in V_k \ V_j} p_k(w) * emb(w) is the contribution from domain-unique vocabulary.

*Proof sketch.* Decompose each centroid into shared and unique components:
c_k = sum_w p_k(w) emb(w) = sum_{w in shared} p_k(w) emb(w) + sum_{w in unique_k} p_k(w) emb(w)

The shared component cancels in the difference c_k - c_j when p_k(w) ~= p_j(w) for shared words. The unique component provides the separation signal. The bound follows from triangle inequality.

QED (sketch -- full proof would require bounding the shared-vocabulary contribution).

**Corollary.** Domains with highly distinctive vocabulary (finance: "dividends", "portfolio"; legal: "defendant", "statute"; medical: "diagnosis", "symptom") will have large unique components and thus large centroid separation. Domains with generic vocabulary (education, cooking with common English) will have smaller separation.

**Prediction 1:** The 6 domains that already route correctly at 40% baseline (finance, health_fitness, legal, math, medical, psychology) should achieve near-100% with embedding routing, as they have distinctive vocabulary.

**Prediction 2:** Some of the 18 currently-failing domains should improve IF they have distinct instruction-level vocabulary patterns that get washed out by transformer contextual mixing but are preserved in embeddings.

**Prediction 3:** Domains with truly overlapping vocabulary (e.g., education/sociology) may remain hard to separate even in embedding space. The floor is set by vocabulary overlap.

## E. Assumptions & Breaking Conditions

1. **A1: Domain vocabulary is partially distinctive.** If all 24 domains use identical vocabulary distributions, all centroids collapse and routing accuracy = 1/24 = 4.2%. BREAKING: accuracy near chance would confirm this.

2. **A2: Mean-pooling over instruction tokens preserves domain signal.** If instructions are too short (< 5 tokens), the mean is noisy. BREAKING: if accuracy increases with instruction length, mean-pooling noise is the bottleneck.

3. **A3: Embedding dimensionality (d=2560) is sufficient for 24-class separation.** By JL-lemma, d >= O(log(24)/eps^2) = O(3.2/eps^2). For eps=0.5, d >= 13. d=2560 is far above this bound.

4. **A4: No external sentence encoder needed.** LoRAuter uses a SupCon-trained encoder that compresses sentences into semantically meaningful embeddings. The base model's embedding layer is a raw token-to-vector lookup with no contextual compression. If contextual compression is essential, this kills the approach.

## F. Worked Example (d=4, K=3)

Consider 3 domains with vocabulary V = {a, b, c, d, e} and embeddings in R^4:
- emb(a) = [1, 0, 0, 0], emb(b) = [0, 1, 0, 0], emb(c) = [0, 0, 1, 0]
- emb(d) = [0, 0, 0, 1], emb(e) = [0.5, 0.5, 0, 0]

Domain 1 (finance): uses {a, b, e} with probs [0.3, 0.3, 0.4]
  c_1 = 0.3*[1,0,0,0] + 0.3*[0,1,0,0] + 0.4*[0.5,0.5,0,0] = [0.5, 0.5, 0, 0]

Domain 2 (legal): uses {c, d, e} with probs [0.4, 0.4, 0.2]
  c_2 = 0.4*[0,0,1,0] + 0.4*[0,0,0,1] + 0.2*[0.5,0.5,0,0] = [0.1, 0.1, 0.4, 0.4]

Domain 3 (cooking): uses {a, c, e} with probs [0.3, 0.3, 0.4]
  c_3 = 0.3*[1,0,0,0] + 0.3*[0,0,1,0] + 0.4*[0.5,0.5,0,0] = [0.5, 0.5, 0.3, 0]

cos(c_1, c_2) = (0.05+0.05)/(0.707*0.583) = 0.24
cos(c_1, c_3) = (0.25+0.25)/(0.707*0.768) = 0.92
cos(c_2, c_3) = (0.05+0.05+0.12)/(0.583*0.768) = 0.49

Finance and cooking overlap heavily (share vocab a, e). Legal is well-separated.
A finance query using words {a, b} would have mean_emb = [0.5, 0.5, 0, 0] = c_1. Perfect routing.
A cooking query using {a, e} would have mean_emb = [0.75, 0.25, 0, 0]. cos with c_1=0.97, cos with c_3=0.94. Close but correct.

## G. Complexity & Architecture Connection

**Centroid computation (one-time, offline):**
- Per domain: tokenize T training texts, look up embeddings O(T * L * d), mean-pool
- Total: O(K * T * L * d) where K=24, T=50, L~128, d=2560
- One-time cost: ~seconds on MLX

**Routing (per query):**
- Tokenize input: O(L)
- Look up embeddings + mean: O(L * d)
- Cosine similarity with K centroids: O(K * d)
- Total: O(L * d + K * d) = O((L + K) * d)
- At L=128, K=24, d=2560: ~390K FLOPs. Sub-millisecond.

**Comparison with prior approaches:**
- Mean-pooled hidden states: O(L * d * n_layers) for full transformer forward pass (~58ms)
- Embedding routing: O(L * d) for embedding lookup only (~0.1ms estimated)
- Speedup: ~500x cheaper than hidden-state routing

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Embedding lookup preserves lexical identity without contextual mixing; domain-distinctive vocabulary maps to unique embedding vectors that cannot be collapsed by downstream processing.

2. Which existing theorem(s) does the proof build on?
   LoRAuter (arxiv 2601.21795) -- task-centroid cosine routing; JL-lemma (1984) -- dimensionality sufficient for separation.

3. What specific numbers does the proof predict?
   P1: 6 already-successful domains maintain >90% accuracy. P2: Overall accuracy significantly >39.4% baseline. P3: Overhead <1ms (vs 58ms for hidden states). Domains with distinctive vocabulary improve most.

4. What would FALSIFY the proof?
   If embedding-space centroids are NOT more separable than hidden-state centroids (i.e., accuracy does not improve significantly), then contextual mixing is not the bottleneck and the issue is fundamental domain overlap.

5. How many hyperparameters does this approach add?
   Count: 0. Centroid computation is a simple mean. Routing is argmax cosine. No learnable parameters.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is replacing the feature extraction method (embedding vs hidden state), not adding a fix on top of routing architecture. Zero parameters, zero training.

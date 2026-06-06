# MATH.md: LoRAuter Task-Representation Routing

## Experiment Type: Guided Exploration

### Proven Framework
LoRAuter (arXiv:2601.21795) demonstrates that routing among LoRA adapters via
semantic task embeddings achieves 101.2% of oracle performance on Llama2-7B.
The method:
1. Encodes validation samples per adapter using a sentence-embedding model E
2. Computes centroid embedding per adapter: e_i = (1/m) sum_j E(v_j)
3. Routes query x by cosine similarity: s_i = (e_x . e_i) / (||e_x|| ||e_i||)
4. Selects top-K adapters, weights via softmax(s/tau)

### Unknown
Does this method predict adapter effectiveness (not just domain identity) on our
5-domain BitNet-2B-4T setup? Finding #253 proved TF-IDF has zero predictive power
(r=-0.079) because bag-of-words space has no isometry to model representation space.
The hypothesis: sentence-embedding space IS a better proxy for representation-space
alignment, because sentence transformers are trained to capture semantic similarity
which correlates with task structure.

---

## A. Failure Mode Identification

**Disease:** The routing signal (TF-IDF) operates in a space that is structurally
disconnected from the space where adapter effectiveness is determined (model
hidden-state space). No monotone mapping exists between bag-of-words similarity
and adapter perturbation utility (Finding #253).

**Symptom vs. disease:** Previous work treated this as "wrong features" (a symptom).
The disease is: **the routing embedding space must be semantically aligned with the
task structure that determines adapter utility.** TF-IDF captures lexical overlap,
not task semantics. A sentence-embedding model trained on semantic similarity
captures the task-level structure that determines which adapter helps.

## B. The Right Question

**Wrong:** "How do we improve TF-IDF routing accuracy?"
**Right:** "What embedding space has the property that proximity correlates with
adapter effectiveness, not just domain membership?"

LoRAuter's answer: a sentence-transformer embedding space, because:
- Sentence transformers are trained to group semantically similar texts
- Adapter training data defines a task manifold in this space
- Validation-set centroids approximate the task manifold center
- Query proximity to a centroid measures semantic task alignment

## C. Prior Mathematical Foundations

**Theorem (Johnson-Lindenstrauss, 1984):** For n points in R^d, there exists a
projection into R^k (k = O(log n / eps^2)) preserving pairwise distances within
(1 +/- eps). Sentence transformers project from token-space to a low-dimensional
embedding space (~384-768d) that approximately preserves semantic distances.

**Centroid Nearest-Neighbor Classification (Cover & Hart, 1967; Tibshirani et al.,
1996):** For well-separated class distributions, nearest-centroid classification
achieves near-Bayes-optimal error rates. The error rate depends on the ratio of
inter-class to intra-class variance (Fisher's discriminant ratio).

**LoRAuter Result (arXiv:2601.21795, Table 2):** On 48 tasks with Llama2-7B,
cosine-similarity routing to task centroids achieves 101.2% of oracle single-adapter
selection, demonstrating that sentence-embedding centroids capture adapter utility,
not just topic.

## D. Predictions

### Behavioral Predictions

**P1 (Domain routing accuracy):** Sentence-embedding routing should achieve >= 80%
domain classification accuracy. Justification: TF-IDF already achieves 90%
(Finding #247); sentence embeddings capture superset of TF-IDF's discriminative
information.

**P2 (Effectiveness correlation):** Cosine similarity between query embedding and
correct-domain centroid should correlate with adapter effectiveness (behavioral
score improvement). Predicted r >= 0.3. Justification: LoRAuter achieves
101.2% oracle on Llama2-7B; if similarity had no predictive power, random
selection would yield ~1/K = ~20% oracle (far below 101.2%).

**P3 (Behavioral quality):** Embedding-routed composition should match or exceed
TF-IDF-routed composition on at least 1/5 domains. Justification: if the routing
signal better predicts adapter utility, behavioral outcomes must improve.

**P4 (Coherence):** Incoherent output rate <= 20%. Sentence-embedding routing
selects at most 1 adapter per query; single-adapter composition preserves
coherence (Finding #238: oracle routing produces < 5% incoherent output).

### Quantitative Predictions (Kill Criteria)

| Prediction | Metric | Threshold | Source |
|------------|--------|-----------|--------|
| P1 | Routing accuracy | >= 80% | Finding #247 (TF-IDF = 90%) |
| P2 (K1) | Embedding-effectiveness correlation | r > 0.3 | LoRAuter 101.2% oracle |
| P3 (K2) | Behavioral improvement vs TF-IDF | >= 1 domain | Finding #253 (TF-IDF r=-0.079) |
| P4 (K3) | Incoherent output rate | <= 20% | Finding #238 (oracle < 5%) |

## E. Assumptions & Breaking Conditions

**A1:** Sentence-embedding space captures task-relevant semantic structure.
If violated: embeddings cluster by surface features (length, style) rather
than task content. Would manifest as high domain classification but low
effectiveness correlation (same failure as TF-IDF).

**A2:** 20 validation samples per adapter are sufficient to estimate centroids.
LoRAuter uses up to 200. If violated: centroids are noisy, routing degrades.
Would manifest as high variance in routing accuracy across runs.

**A3:** Top-1 routing is sufficient (single adapter selection).
LoRAuter uses top-K with softmax weighting. If violated: some queries need
blended adapters. Would manifest as queries near centroid boundaries getting
wrong adapter.

**A4:** BitNet-2B-4T ternary weights + LoRA adapters exhibit similar task
structure to Llama2-7B. If violated: the ternary quantization changes the
effective task manifold. Would manifest as lower routing accuracy than
expected.

## F. Worked Example (d=384, 5 domains)

Sentence-transformer output dimension: 384 (typical for MiniLM-class models).

Suppose centroids for 5 domains:
- e_math = [0.3, 0.1, -0.2, 0.5, ...] (384-dim, unit normalized)
- e_code = [-0.1, 0.4, 0.3, -0.2, ...]
- e_medical = [0.2, -0.3, 0.1, 0.4, ...]
- e_legal = [0.1, 0.2, -0.4, 0.1, ...]
- e_finance = [-0.2, 0.3, 0.2, -0.1, ...]

Query: "What is the derivative of x^2?"
e_query = [0.28, 0.12, -0.18, 0.48, ...]

Cosine similarities:
- cos(e_query, e_math) = 0.92 (high — math query, math centroid)
- cos(e_query, e_code) = 0.31
- cos(e_query, e_medical) = 0.45
- cos(e_query, e_legal) = 0.28
- cos(e_query, e_finance) = 0.15

Route to: math adapter (highest similarity). Correct.

For effectiveness prediction: if behavioral score correlates with similarity,
then queries with cos > 0.8 should show higher adapter benefit than queries
with cos ~ 0.5. This is the r > 0.3 prediction.

## G. Complexity & Architecture Connection

**Offline (one-time):** Compute 5 centroids from 20 validation samples each.
Cost: 100 forward passes through sentence-transformer (~0.1s total).

**Online (per-query):** One sentence-transformer encoding + 5 cosine similarities.
Cost: ~1ms. Negligible compared to LLM generation.

**Memory:** 5 x 384 = 1920 floats = 7.5 KB for centroids. Negligible.

**Comparison to TF-IDF:** TF-IDF requires vocabulary-sized vectors (~50K sparse).
Sentence embeddings are dense 384-dim. Both are negligible at 5 domains.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Sentence-embedding space is trained to preserve semantic task structure, making
   proximity correlate with task identity and utility, unlike bag-of-words space.

2. Which existing theorem(s) does the proof build on?
   Johnson-Lindenstrauss lemma (distance preservation in projections), Cover & Hart
   nearest-centroid classification, LoRAuter empirical result (arXiv:2601.21795 Table 2).

3. What specific numbers does the proof predict?
   P1: routing accuracy >= 80%, P2: effectiveness correlation r > 0.3,
   P3: behavioral improvement on >= 1 domain vs TF-IDF, P4: coherence >= 80%.

4. What would FALSIFY the proof (not just the experiment)?
   If sentence-embedding similarity has zero correlation with adapter effectiveness
   (r ~ 0), it would mean semantic similarity does not predict task-level adapter
   utility — the same structural disconnect as TF-IDF but in a different space.

5. How many hyperparameters does this approach add?
   Count: 2 (sentence-transformer model choice, number of validation samples m).
   Model choice: use LoRAuter's recommended model (Styxxxx/lora_retriever) or
   a standard sentence-transformer. m: LoRAuter uses up to 200; we use 20
   (constrained by validation set size).

6. Hack check: Am I adding fix #N to an existing stack?
   No. This replaces TF-IDF routing entirely with a different embedding space.
   It is a single mechanism (centroid-based routing in sentence-embedding space),
   not a stack of fixes.

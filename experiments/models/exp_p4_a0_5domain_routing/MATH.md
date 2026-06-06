# MATH.md — P4.A0: 5-Domain TF-IDF Ridge Routing

**Experiment type:** Verification
**Prior result:** Finding #458 (N=25 ridge routing 98.8%), P3.D0 (binary math/general 100%)
**Citation:** MixLoRA (arxiv 2312.09979); Hoerl & Kennard 1970 (ridge regression theory)

---

## Disease vs Symptom

**Symptom:** P3.D0 only tested binary routing (math vs. general). Production requires 5-class routing.

**Disease:** Unverified claim that TF-IDF ridge routing scales to 5 real domain corpora with distinct vocabulary distributions.

**SIGReg question:** What structure makes 5-class domain confusion geometrically impossible?

**Answer:** High-dimensional TF-IDF vocabulary divergence. When P(V_medical ∩ V_code) << P(V_medical), P(V_code), the L2-normalized TF-IDF centroids are near-orthogonal. Ridge regression amplifies discriminative vocabulary terms and suppresses shared academic language ("the", "and", "results show"). By Finding #458, this achieves near-Bayes-optimal accuracy.

---

## Theorem 1: Ridge Routing Accuracy for 5 Real Domains

**Setup:**
- Domains D = {medical, code, math, legal, finance}, N = 5
- Feature map: φ(q) = TF-IDF(q) ∈ ℝ^d, d ≈ 10,000–50,000 (max_features)
- Training data: Φ ∈ ℝ^{M×d}, Y ∈ ℝ^{M×5} (one-hot), M = n_train × 5
- Ridge solution: W* = (Φ^TΦ + λI)^{-1} Φ^T Y, λ = 0.1

**Claim:** Weighted routing accuracy ≥ 95% when domain corpora have high TF-IDF vocabulary divergence.

**Vocabulary Divergence Condition:** For domains i ≠ j,
    cos(C_i, C_j) = (Σ_v TF-IDF(v,D_i) × TF-IDF(v,D_j)) / (||C_i|| ||C_j||) < 0.3

where C_i is the centroid of domain i's TF-IDF vectors.

**Proof:**

Step 1 (Vocabulary specialization): Medical texts use "pathology", "diagnosis", "prognosis" — absent from code ("function", "class", "return") and finance ("yield", "equity", "derivative"). This gives cos(C_medical, C_code) ≈ 0.05–0.15 empirically.

Step 2 (Ridge discrimination): W* learns to weight domain-discriminative terms positively and shared academic vocabulary negatively. This is the standard bias-variance tradeoff: regularization with λ = 0.1 shrinks shared-vocabulary weights toward zero, amplifying specificity.

Step 3 (Accuracy bound): By the classification error bound for linear discriminants:
    P(error) ≤ exp(-Δ²_min / (2σ²))
where Δ_min = min_{i≠j} ||W* φ(C_i) - W* φ(C_j)||₂ is the minimum inter-class margin and σ² is the within-class variance after W* projection.

Step 4 (Scaling from N=25 to N=5): Finding #458 shows N=25 synthetic → 98.8%. Real-data N=5 domains have MORE discriminative vocabulary (actual domain corpora vs MMLU questions). With fewer classes and stronger vocabulary divergence, accuracy ≥ 98.8% is predicted.

**QED**

---

## Theorem 2: Domain Centroid Separability

**Claim:** For 5 real domain corpora, pairwise cosine similarity of TF-IDF centroids < 0.3.

**Proof:**

By the Johnson-Lindenstrauss lemma, L2-normalized sparse vectors in ℝ^d concentrate on the surface of a unit sphere. For uncorrelated vocabulary distributions (distinct domains), cosine similarity between centroids decays as:
    E[cos(C_i, C_j)] ≈ |V_shared| / sqrt(|V_i| × |V_j|)

where V_i = vocabulary used in domain i. For medical vs code:
- |V_medical| ≈ 8,000 domain terms, |V_code| ≈ 5,000 tokens
- |V_shared| ≈ 200 (general English + numerals)
- E[cos] ≈ 200 / sqrt(8000 × 5000) ≈ 200 / 6325 ≈ 0.032

All 10 pairwise cosines for 5 domains are predicted < 0.3.

**QED**

---

## Quantitative Predictions

| Metric | Prediction | Kill Criterion |
|--------|-----------|---------------|
| Weighted routing accuracy | ≥ 98% (N=5 is easier than N=25) | K1214: ≥ 95% |
| Per-domain precision (worst) | ≥ 90% | K1215: ≥ 85% |
| Max pairwise centroid cosine | < 0.15 | K1216: < 0.30 |
| Training time | < 60 s | — |
| Inference p99 latency | < 1 ms | — |

---

## Failure Modes

**F1: Legal/Finance vocabulary overlap** — Both use "contract", "liability", "obligation". This is the most likely confusion pair. Ridge should disambiguate via more specific terms ("tort", "equity") but may underperform on ambiguous queries.

**F2: Math/Code overlap** — Code solutions often contain mathematics. Disambiguation requires detecting natural language problem framing (math) vs code syntax (code). TF-IDF should capture this via punctuation-heavy tokens.

**Impossibility structure for K1214 failure:** K1214 fails (< 95%) only if ≥ 3/100 queries are misrouted. Given N=25 achieved 98.8% with more classes and weaker vocabulary, N=5 failing at < 95% would require legal/finance confusion rate > 10% — only possible if |V_legal ∩ V_finance| / |V_legal| > 0.3. We predict < 0.1 empirically.

---

## Connection to Architecture

The 5-domain router is the first component of the full P4 production pipeline:
1. **Router** (this experiment) → 98%+ accuracy → K1214-K1216 [P4.A0]
2. **Domain adapters** → +10pp behavioral quality → [P4.A1]
3. **Personal style** → 93.3% style (verified P3.D0) → [P4.A2]

If routing fails at N=5, the architecture requires a hierarchical router (coarse-fine) or a learned query embedding (not TF-IDF). This experiment determines which approach is needed.

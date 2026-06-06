# TF-IDF Routing on Real NLP: Domain Separability Theorem

## Motivation

**Finding #354** established TF-IDF routing achieves 95% accuracy on 5 toy domains
(arithmetic/sort/reverse/repeat/parity) but flagged: *"Toy scale (128 vocab, 5 simple domains)"*.

**Finding #386** showed that a wrong-domain adapter causes Q_wrong ≈ -0.58 (58% relative
harm) — making routing accuracy directly tied to product quality.

**Critical gap:** Does TF-IDF routing transfer to real NLP domains (math word problems,
Python code instructions, general text)?

**Structural concern:** Toy domains have near-disjoint vocabularies by construction.
Real NLP domains share common words (articles, prepositions, verbs) — does TF-IDF's
IDF weighting suppress these shared tokens enough to maintain separability?

---

## Theorem 1 — TF-IDF Centroid Separability for Real NLP Domains

**Setup:** Let $\mathcal{V}$ be a vocabulary of size $V$. Each document $d$ from domain $k$
is represented as a TF-IDF vector:
$$v_d^{(w)} = \text{tf}(w, d) \cdot \log\frac{N}{1 + \text{df}(w)}$$

where $\text{tf}(w, d)$ is term frequency in $d$, $N$ is corpus size, and $\text{df}(w)$ is
the number of documents containing $w$. Sub-linear TF is used: $\text{tf}(w,d) = 1 + \log(\text{count}(w,d))$.

Define the domain centroid: $\mathbf{c}_k = \frac{1}{|D_k|}\sum_{d \in D_k} \mathbf{v}_d$

**Claim:** For domains $k \in \{\text{math, code, text}\}$ with distinct high-IDF vocabularies:
$$\cos(\mathbf{c}_i, \mathbf{c}_j) < 0.3 \quad \forall i \neq j$$

and a nearest-centroid classifier achieves routing accuracy $\geq 95\%$ on all domains.

**Proof:**

TF-IDF assigns near-zero weight to high-frequency shared terms (common English words
have $\log(N/\text{df}(w)) \approx 0$). The centroid $\mathbf{c}_k$ is dominated by
domain-specific high-IDF terms.

Let $\mathcal{V}_k^*$ = {words with TF-IDF weight in top-$P$ percentile for domain $k$}.

For real NLP domains:
- **Math**: "solve", "calculate", "many", "total", "equation", "value", "equals" → $\mathcal{V}_\text{math}^*$
- **Code**: "function", "def", "return", "class", "list", "implement", "parameter" → $\mathcal{V}_\text{code}^*$
- **Text**: proper nouns, verbs describing events, location names → $\mathcal{V}_\text{text}^*$

By domain distinctness: $\mathcal{V}_\text{math}^* \cap \mathcal{V}_\text{code}^* \approx \emptyset$,
and similarly for other pairs.

The inner product:
$$\mathbf{c}_i^\top \mathbf{c}_j = \sum_{w \in \mathcal{V}_i^* \cup \mathcal{V}_j^*} c_i^{(w)} c_j^{(w)} + \underbrace{\sum_{w \notin \mathcal{V}_i^* \cup \mathcal{V}_j^*} c_i^{(w)} c_j^{(w)}}_{\approx 0 \text{ (low IDF weight)}}$$

Since $\mathcal{V}_i^* \cap \mathcal{V}_j^* \approx \emptyset$, the first sum is also small.
Therefore $\cos(\mathbf{c}_i, \mathbf{c}_j) \approx 0$.

**QED**

**Corollary (Routing Accuracy Lower Bound):**
The composition quality floor from Finding #354 is:
$$Q_\text{composed} \geq \rho_\text{routing} \times Q_\text{per-adapter}$$

where $\rho_\text{routing} \geq 0.95$ (predicted) → composition quality floor $\geq 0.95 \times Q_\text{single}$.

---

## Theorem 2 — IDF Suppression of Shared Vocabulary

Let $S = \mathcal{V}_\text{math} \cap \mathcal{V}_\text{code}$ be shared vocabulary (common English words).
For shared word $w$: $\text{df}(w) \approx N$ → $\text{IDF}(w) \approx 0$.

Therefore shared vocabulary contributes $\approx 0$ to TF-IDF centroid coordinates.
Real NLP domains are as separable as toy domains once IDF suppresses shared tokens.

**QED**

---

## Quantitative Predictions

| Metric | Predicted | Kill if |
|--------|-----------|---------|
| cos(math, code) | < 0.20 | — |
| cos(math, text) | < 0.30 | — |
| cos(code, text) | < 0.30 | — |
| Routing acc. math | ≥ 95% | < 80% = K950 FAIL |
| Routing acc. code | ≥ 95% | < 80% = K950 FAIL |
| Routing acc. text | ≥ 90% | < 80% = K950 FAIL |

Text is predicted slightly lower because general news text has more diverse vocabulary
(proper nouns from different domains) — harder to build a tight centroid.

## Kill Criteria

- **K950**: Nearest-centroid routing accuracy ≥ 80% on ALL 3 domains.
  - PASS → TF-IDF routing is production-ready for real NLP domains
  - FAIL → must identify which domain pair confuses and why (derive impossibility structure)

## Prior Work

- Salton & Buckley (1988), "Term-weighting approaches in automatic text retrieval" — TF-IDF
- Finding #354: TF-IDF routing 95% accuracy on 5 toy domains (exp_m2p_tfidf_routing_n5)
- Finding #386: Q_wrong = -0.58 → routing is critical (exp_q_wrong_real_domains)
- Aghajanyan et al. (2021), arxiv 2012.13255 — intrinsic dimensionality for domain adaptation

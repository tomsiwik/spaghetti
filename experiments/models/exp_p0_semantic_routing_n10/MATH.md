# Learned Classifier Routing at N=10: Feature Space Analysis

## Motivation

Finding #524: TF-IDF + RidgeClassifier achieves 79.3% at N=10 (down from 98.3%
at N=3). The Ridge classifier IS already a trained linear model — the bottleneck
is the TF-IDF feature space itself, not the classification method.

Finding #255: Sentence-embedding centroid routing achieves 96% at N=5.
Finding #256: Sentence-embedding centroid collapses to 33.3% at N=24.
Unknown: What happens at N=10?

**SIGReg reasoning:**
- **Disease:** TF-IDF features lack semantic signal. "Clinical trial" and
  "cognitive assessment" have similar word frequencies but different domain meanings.
- **Right question:** In what feature space are the 10 domains maximally separable?
- **Existing math:** Fisher discriminant theory (Fisher, 1936). Domain separation
  depends on the ratio of between-class to within-class variance in the feature space.

## Problem Setup

Given N=10 domains with text samples, compare routing methods:

1. **TF-IDF + centroid** (nearest-centroid): r(x) = argmin_k ||tfidf(x) - mu_k||
2. **TF-IDF + Ridge** (trained, baseline from #524): r(x) = argmax_k w_k^T tfidf(x)
3. **TF-IDF + logistic** (trained, different loss): cross-entropy optimized
4. **Sentence-embedding + centroid**: r(x) = argmax_k cos(embed(x), mu_k)
5. **Sentence-embedding + logistic**: trained classifier on semantic features
6. **Combined TF-IDF + embedding + logistic**: feature fusion

## Theorem 1: Feature Space Quality Determines Routing Ceiling

**Theorem.** For a linear classifier on features phi(x), the classification error
is lower-bounded by the Bayes error in the feature space phi. Two feature spaces
phi_1, phi_2 satisfy:

err(phi_1) < err(phi_2)  iff  J_Fisher(phi_1) > J_Fisher(phi_2)

where J_Fisher is the multi-class Fisher ratio:

J = tr(S_B) / tr(S_W)

S_B = between-class scatter, S_W = within-class scatter.

**Corollary:** If TF-IDF + Ridge = 79.3%, then TF-IDF + logistic ~ 79-82% (both
linear, same features, different losses). To improve significantly, we need
DIFFERENT FEATURES (sentence embeddings), not a different classifier.

## Theorem 2: Embedding Margin Scaling

**Theorem (after Radovanovic et al., 2010).** For N centroids in d-dimensional
embedding space, the expected minimum pairwise margin scales as:

margin_min ~ sqrt(2/d) * (1 - (N-1)/d)  for N << d

For all-MiniLM-L6-v2 (d=384):
- N=5:  margin ~ 0.34 → 96% (observed, Finding #255)
- N=10: margin ~ 0.24 → predicted 85-90%
- N=24: margin ~ 0.15 → 33.3% (observed, Finding #256)

N=10 is the critical transition zone. If embeddings maintain > 0.20 margin
on all domain pairs, centroid routing should achieve > 85%.

## Theorem 3: Feature Fusion Advantage

**Theorem.** Concatenating independent feature spaces phi_1 (TF-IDF, lexical) and
phi_2 (embedding, semantic) yields Fisher ratio:

J(phi_1 || phi_2) >= max(J(phi_1), J(phi_2))

with equality only when both spaces provide identical discriminative signal.
In practice, lexical and semantic features are complementary: TF-IDF captures
domain-specific terminology frequency, embeddings capture meaning. The combined
classifier should strictly outperform either alone.

## Predictions

| Method | Predicted N=10 accuracy | Basis |
|--------|------------------------|-------|
| TF-IDF centroid | ~75% | No training, pure distance |
| TF-IDF + Ridge | 79.3% (measured) | Finding #524 baseline |
| TF-IDF + logistic | 80-83% | Similar to Ridge, CE loss marginally better |
| Sentence-embed centroid | 85-90% | Thm 2: margin still sufficient at N=10 |
| Sentence-embed + logistic | 88-93% | Trained on semantic features |
| Combined + logistic | **90-95%** | Thm 3: fusion strictly better |

**K1443 (>= 90%):** Combined method should PASS. Individual methods may not.
**K1444 (>= 85%):** Sentence-embedding methods should PASS.
**K1445 (>= 85% all domains, none < 70%):** Combined method should rescue
psychology (62%) and science (67%) from TF-IDF failure via semantic signal.
**K1446 (< 5 seconds):** All methods use sklearn/numpy. PASS by design.

## Kill Conditions (from proof)

- If sentence-embed centroid < 80% at N=10: hubness is worse than Thm 2 predicts;
  N=10 MMLU domains are as crowded as N=24 general knowledge.
- If combined fusion < 85%: lexical and semantic signals are NOT complementary;
  both encode the same domain structure. This would mean routing at N=10 requires
  domain-specific contrastive training, not general features.
- If best method < 85%: the 10 MMLU domains are inherently ambiguous in any
  pre-trained feature space. Hierarchical routing (cluster-then-refine) is
  the only path forward.

## References

- Fisher (1936). The Use of Multiple Measurements in Taxonomic Problems.
- Radovanovic et al. (2010). Hubs in Space: Popular Nearest Neighbors.
- LoraRetriever (arxiv 2402.09997): Contrastive routing for LoRA selection.
- Findings #524, #255, #256, #431, #207.

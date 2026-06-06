# Embedding Router at N=25: Scaling Learned Classifier from N=10

## Motivation

Finding #525: Combined TF-IDF+embedding logistic achieves 89.9% at N=10 (grouped
meta-domains). Embedding Fisher ratio (0.133) is 4.9x better than TF-IDF (0.027).
Finding #431: TF-IDF nearest-centroid achieves 86.1% at N=25 (individual MMLU subjects).
Finding #524: TF-IDF routing degrades sub-linearly with N.

**Disease:** TF-IDF features saturate at N=25 because topically adjacent domains
(astronomy/physics, prehistory/history, finance/statistics) share vocabulary.
**Right question:** Does embedding-space separation persist at N=25 where lexical
features fail?

## Problem Setup

25 domains: 5 real (math, code, medical, legal, finance) + 20 MMLU subjects.
Same 6 methods as N=10 experiment:
1. TF-IDF centroid (baseline from Finding #431: 86.1%)
2. TF-IDF + Ridge (trained linear)
3. TF-IDF + Logistic Regression
4. Sentence-embedding centroid (all-MiniLM-L6-v2, d=384)
5. Sentence-embedding + Logistic Regression
6. Combined TF-IDF + embedding + Logistic Regression

## Theorem 1: Fisher Ratio Scaling with N

**Theorem.** For N classes in d-dimensional feature space, the multi-class Fisher
ratio J = tr(S_B)/tr(S_W) scales as:

J(N) ~ J(N_0) * f(N/N_0)

where f is determined by class structure. For uniformly distributed class centroids,
J grows with N (more between-class scatter). For clustered centroids (topically
adjacent domains), J grows sub-linearly because new classes add within-cluster
variance.

**At N=25 vs N=10:**
- TF-IDF: Fisher ratio likely decreases because 25 individual MMLU subjects have
  more lexical overlap than 10 grouped meta-domains. Predicted: J_tfidf(25) ~ 0.015-0.025
  (vs 0.027 at N=10).
- Embedding: Fisher ratio may increase because individual subjects are more
  semantically distinct than meta-groups. "Astronomy" vs "high_school_physics" is
  more separable than "science" (which merges both). Predicted: J_embed(25) ~ 0.10-0.15
  (vs 0.133 at N=10).

## Theorem 2: Embedding Margin at N=25

**Theorem (after Radovanovic et al., 2010).** For N centroids in d-dimensional space
(d=384 for MiniLM), the minimum pairwise margin scales as:

margin_min ~ sqrt(2/d) * (1 - (N-1)/d)

- N=10: margin ~ sqrt(2/384) * (1 - 9/384) = 0.072 * 0.977 = 0.070 (theoretical)
  Observed at N=10: 0.100 (legal-philosophy). Theory is conservative.
- N=25: margin ~ sqrt(2/384) * (1 - 24/384) = 0.072 * 0.9375 = 0.068 (theoretical)

The margin barely changes from N=10 to N=25 because d=384 >> N=25. This predicts
that embedding centroid routing should NOT collapse at N=25 (unlike Finding #256's
33.3%, which used a different experimental setup and data loading).

## Theorem 3: Logistic Regression vs Centroid Gap

**Theorem.** For a logistic classifier with C classes:
- Centroid routing: equivalent to naive Bayes with spherical covariance assumption
- Logistic regression: finds optimal linear boundaries, corrects for class overlap

The gap grows with the number of overlapping class pairs. At N=10, the gap was:
- TF-IDF: +8.4pp (Ridge) over centroid
- Embedding: +3.6pp (logistic) over centroid

At N=25, more domain pairs overlap, so the trained-vs-centroid gap should GROW:
- TF-IDF: predicted +8-12pp over centroid
- Embedding: predicted +4-8pp over centroid

## Predictions

| Method | Predicted N=25 | Basis |
|--------|---------------|-------|
| TF-IDF centroid | 82-86% | Finding #431 = 86.1% (different params) |
| TF-IDF + Ridge | 86-90% | Centroid + ~6pp training gain |
| TF-IDF + logistic | 86-90% | Similar to Ridge |
| Embed centroid | 78-83% | Margin still OK but more confusion pairs |
| Embed + logistic | 84-88% | +4-8pp over centroid |
| Combined + logistic | **88-93%** | Feature fusion, best method |

**K1473 (overall >= 90%):** Combined method borderline — depends on fusion quality.
**K1474 (embed-only >= 85%):** Logistic should PASS if embedding separation holds.
**K1475 (combined >= 92%):** Ambitious — requires strong feature complementarity.
**K1476 (worst-domain >= 70%):** Should PASS if no domain collapses completely.
**K1477 (latency < 5ms):** Pure sklearn/numpy, PASS by design.

## Kill Conditions (from proof)

- If embed centroid < 70% at N=25: embedding space has collapsed; d=384 is
  insufficient for 25-domain separation. The hubness effect dominates.
- If combined logistic < 85% at N=25: neither lexical nor semantic features
  can separate 25 domains without contrastive fine-tuning.
- If trained classifiers don't improve over centroids by >= 3pp: the domains
  are linearly separable and training adds nothing — the ceiling is in the
  features, not the classifier.

## References

- Finding #525: Embedding routing 89.9% at N=10
- Finding #431: TF-IDF centroid 86.1% at N=25
- Finding #524: TF-IDF degrades sub-linearly with N
- Finding #256: Embedding centroid collapse at N=24 (different setup)
- arXiv:1908.10084: Sentence-BERT (MiniLM backbone)
- arXiv:2402.09997: LoraRetriever (contrastive routing)
- Radovanovic et al. (2010): Hubs in Space

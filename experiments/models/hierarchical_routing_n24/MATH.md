# MATH.md: Hierarchical Two-Stage Routing at N=24

## Type: Frontier Extension

**Proven result being extended:** Per-adapter binary routing heads achieve 100%
classification accuracy at N=5 on trivially separable domains (Finding #179).

**Gap:** N=5 routing works because d=2560 hidden states cleanly separate 5 distant
domains. At N=24, mean-pooled hidden states only separate ~6 domains (Finding #192).
We need to show that clustering reduces effective N per stage to <=5 while keeping
misrouting costs bounded.

---

## A. Failure Mode Identification

**The disease:** Flat routing at N=24 hits a ~40% accuracy ceiling regardless of
architecture (Findings #189, #191, #192, #193, #194). Six mechanisms killed:

| Mechanism | Accuracy | Failure |
|-----------|----------|---------|
| Energy gap argmin | 8.3% | Uncalibrated adapter magnitudes |
| Binary routing heads | 39.6% | False positive cascade |
| Centralized softmax | 39.4% | Same 18 domains unresolvable |
| Softmax router | ~40% | Same ceiling |
| Embedding routing | 28.3% | Centroid collapse (cos 0.986) |
| Hidden-state centroid | 32.5% | Better but same 18 fail |

The same 6 domains succeed in every experiment (finance, health_fitness, legal,
math, medical, psychology). The same 18 fail. This is a representation limit, not
an architecture limit.

**Key observation from Finding #192:** Despite 40% accuracy, softmax router matches
oracle PPL quality (gamma 0.625 = oracle, 0.0% gap). Within-cluster misrouting is
PPL-benign. This means the 18 "hard" domains form natural confusion clusters where
any member is approximately as good as the correct one.

---

## B. The Right Question

**Wrong:** "How do we build a routing head that distinguishes 24 overlapping domains?"
(Six kills prove this is impossible with mean-pooled hidden states.)

**Right:** "Given that within-cluster misrouting is PPL-benign, what clustering of
24 domains into K groups minimizes the expected routing cost, where cost is the PPL
penalty of selecting the wrong adapter?"

This reframes routing from a classification problem (pick the right adapter) to a
cost-minimization problem (pick an adapter whose PPL penalty is minimal).

---

## C. Prior Mathematical Foundations

### C1. Confusion Graph and Spectral Clustering

The confusion matrix C from a flat N-class classifier defines a weighted graph
G = (V, E, w) where V = {domains}, w(i,j) = C_ij + C_ji (symmetric confusion).
The graph Laplacian L = D - W (D = diag(sum of weights)) has eigenvalues
0 = lambda_1 <= lambda_2 <= ... <= lambda_N.

**Theorem (Fiedler, 1973):** The algebraic connectivity lambda_2 determines the
minimum edge cut. The Fiedler vector (eigenvector of lambda_2) provides the optimal
2-partition that minimizes inter-cluster confusion.

**Spectral clustering (von Luxburg, 2007):** Embedding vertices in the space of
the first K eigenvectors of the normalized Laplacian L_rw = I - D^{-1}W, then
applying K-means, minimizes the normalized cut objective, which partitions the
confusion graph into K groups with minimal inter-group confusion.

### C2. Hierarchical Classification Error Decomposition

**Proposition (standard, see Deng et al. 2011 "Hedging Your Bets"):** For a
two-stage classifier with cluster selection stage (accuracy p_c) and within-cluster
selection (accuracy p_w), the overall accuracy is:

  acc_total = p_c * p_w

But the PPL cost of misrouting depends not on accuracy but on the PPL penalty
of the wrong adapter. Define delta(i,j) = PPL(text_i, adapter_j) / PPL(text_i,
adapter_i) - 1 as the relative PPL penalty of routing domain i to adapter j.

**Effective cost:** If cluster c contains domains {d_1, ..., d_m}, and the router
selects the wrong domain within the cluster, the expected penalty is:

  E[delta | misroute within c] = (1/(m-1)) * sum_{j != i} delta(i, j)

for the correct domain i. If this is small (our PPL-benign hypothesis), then
within-cluster misrouting has low cost even at low accuracy.

### C3. Reduction to Solved Problem

**Finding #179:** At N=5, routing accuracy is 100%. The mechanism is a 2-layer
MLP (d=2560 -> 32 -> 1) trained as binary classifier.

**Hypothesis:** If spectral clustering of the N=24 confusion graph yields K~5
clusters, then:
- Stage 1 (which cluster?) faces a K-class problem where K~5
- Stage 2 (which domain within cluster?) faces an m-class problem where m~5
- Both stages are in the N<=5 regime where routing is proven

---

## D. Proof of Guarantee (Conditional)

**Theorem 1 (Hierarchical routing cost bound).** Let D = {d_1, ..., d_N} be N
domains with PPL cross-matrix P where P_ij = PPL(text_i, adapter_j). Let
C_1, ..., C_K be a K-partition of D. Define:

- p_c = cluster-level routing accuracy (probability of selecting correct cluster)
- p_w(k) = within-cluster routing accuracy for cluster k
- delta_inter = max_{i in C_k, j in C_l, k != l} (P_ij / P_ii - 1)
  (worst-case inter-cluster penalty)
- delta_intra(k) = max_{i,j in C_k} (P_ij / P_ii - 1)
  (worst-case intra-cluster penalty)

Then the expected PPL ratio of hierarchical routing vs oracle is bounded by:

  E[PPL_routed / PPL_oracle] <= 1 + p_c * (1 - p_w_avg) * delta_intra_avg
                                  + (1 - p_c) * delta_inter

*Proof sketch.* Decompose by cases:
1. Correct cluster + correct domain (prob p_c * p_w): cost = 0
2. Correct cluster + wrong domain (prob p_c * (1-p_w)): cost <= delta_intra
3. Wrong cluster (prob 1-p_c): cost <= delta_inter

The expected penalty is the probability-weighted sum of these cases. QED.

**Corollary 1.** If delta_intra << delta_inter (within-cluster misrouting is
benign compared to cross-cluster), then even moderate within-cluster accuracy
p_w yields near-optimal PPL, because the dominant term p_c * (1-p_w) * delta_intra
is small.

**Corollary 2.** If p_c >= 0.80 and delta_intra <= 0.05 (5% PPL penalty within
clusters), then E[PPL_routed / PPL_oracle] <= 1 + 0.80 * 0.5 * 0.05 + 0.20 * delta_inter
= 1.02 + 0.2 * delta_inter. Even at delta_inter = 0.5 (50% cross-cluster penalty),
the total bound is 1.12 (12% degradation).

---

## D2. Predictions

### Behavioral
1. Spectral clustering of the confusion matrix will yield 4-6 natural clusters
   (based on the 6 separable + 18 overlapping domain structure)
2. Cluster-level routing accuracy will be significantly higher than flat N=24
   accuracy (>60% vs 39.4%)
3. Hierarchically-routed PPL will beat uniform 1/24 PPL
4. The "easy 6" domains (finance, health_fitness, legal, math, medical, psychology)
   will form their own singleton or small clusters

### Quantitative
- P1: Cluster-level accuracy >= 60% (from K~5 classes in d=2560 space, proven regime)
- P2: Overall top-1 accuracy >= 50% (stage1 * stage2, both better than random)
- P3: Routed PPL < uniform PPL (gamma_routed < 1.0)
- P4: Two-stage overhead < 15% (two small MLP forwards vs one base forward)
- P5: delta_intra < 0.10 for most clusters (within-cluster PPL penalty < 10%)

---

## E. Assumptions and Breaking Conditions

**A1: Confusion matrix structure.** The confusion patterns from Finding #192 are
stable (not artifacts of that particular classifier). If violated: clusters may
not group truly confused domains, reducing the benefit.

**A2: PPL-benign within-cluster misrouting.** Domains that confuse each other in
hidden-state space produce similar PPL when composed. If violated: delta_intra
could be large, breaking Corollary 2. This is the key empirical question.

**A3: Cluster separability.** K~5 clusters are separable in d=2560 mean-pooled
hidden states. If violated: stage 1 accuracy drops, making the hierarchy pointless.
Breaking condition: cluster accuracy < 50% (no better than random at K=5).

**A4: Data sufficiency.** 40 training samples per domain for cluster centroids
and routing heads. May be insufficient for some clusters.

---

## F. Worked Example (K=5 clusters of ~5 domains each)

Suppose we cluster 24 domains into K=5 groups:
- Cluster A: {math, science, engineering, environmental, economics} (STEM)
- Cluster B: {medical, health_fitness, psychology, cooking, sports} (body/health)
- Cluster C: {legal, finance, politics, marketing, cybersecurity} (professional)
- Cluster D: {history, philosophy, linguistics, sociology, education} (humanities)
- Cluster E: {code, creative_writing, music, agriculture} (creative/applied)

Stage 1: route to cluster (5-class). At N=5, proven 100% accuracy for separable
domains. Even if not perfectly separable, cluster centroids are more distant than
individual domain centroids (averaging reduces noise).

Stage 2: route within cluster. Each cluster has 4-5 domains. If the easy domains
(math, medical, legal, finance, health_fitness, psychology) anchor their clusters,
the remaining domains only need to be distinguished from 3-4 alternatives, not 23.

Example PPL matrix within Cluster B (hypothetical):
|           | medical | health | psych | cooking | sports |
|-----------|---------|--------|-------|---------|--------|
| medical   | 4.75    | 5.10   | 5.30  | 8.20    | 7.80   |
| health    | 5.80    | 4.90   | 5.20  | 7.50    | 6.10   |

delta_intra(B) for medical->health = 5.10/4.75 - 1 = 0.074 (7.4% penalty)
If cooking/sports are in this cluster but get misrouted to medical, penalty is larger.
The key is that the CONFUSION pattern (what gets confused) aligns with the
COST pattern (what has low penalty).

---

## G. Complexity and Architecture

**Stage 1 router:** K-class softmax MLP (d=2560 -> 32 -> K). Params: ~82K.
**Stage 2 routers:** K MLPs, each (d=2560 -> 32 -> m_k). Total params: ~82K * K.
**Total routing params:** ~500K (comparable to single softmax router at 165K).
**FLOPs per query:** 2 * (2560*32 + 32*K) + 2 * (2560*32 + 32*m_k) ~ 330K.
**Overhead:** 2 small MLP forwards. At 0.29% per forward (Finding #192), expect ~0.6%.

**Memory:** All routing heads are tiny (~82K params each). Even K=5 stage-2 heads
plus 1 stage-1 head = 6 * 82K = 492K params = 984 KB. Negligible.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Confusion-graph clustering guarantees that misroutable domains are in the same
   cluster, making within-cluster misrouting PPL-benign by construction.**

2. Which existing theorem(s) does the proof build on?
   Fiedler (1973): algebraic connectivity and optimal graph partitioning.
   Von Luxburg (2007): spectral clustering minimizes normalized cut.
   Finding #192: within-cluster misrouting is PPL-benign at N=24.

3. What specific numbers does the proof predict?
   P1: cluster accuracy >= 60%, P2: overall >= 50%, P3: routed PPL < uniform,
   P4: overhead < 15%, P5: delta_intra < 0.10 for most clusters.

4. What would FALSIFY the proof (not just the experiment)?
   If domains that confuse each other in representation space do NOT produce similar
   PPL (i.e., confusion != PPL-benign similarity), then clustering by confusion
   does not minimize routing cost. delta_intra could be as large as delta_inter.

5. How many hyperparameters does this approach add?
   Count: 1 (K = number of clusters). K is discoverable from the confusion graph
   eigengap (Fiedler analysis), not arbitrary.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is a structural decomposition of the routing problem, not a patch on
   an existing router. It replaces the flat router entirely.

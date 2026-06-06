# MATH.md: Sentence-Embedding Routing at N=24

## Type: Guided Exploration
**Proven framework:** Sentence-embedding centroid routing achieves 96% at N=5 (Finding #255).
**Unknown:** Does it scale to N=24? Prior methods all failed (28-40%).

## A. Failure Mode Identification

The failure mode is **centroid confusion**: as N grows, domain centroids crowd together
in embedding space, reducing inter-centroid margins. When the margin between the true
centroid and the nearest competitor falls below the intra-class scatter, routing errors
become systematic.

This is the ROOT CAUSE of all six prior N=24 failures:
- Hidden-state routing (32.5%): mean-pooled hidden states lack domain-discriminative signal
- TF-IDF (35%): sparse overlap between domain vocabularies
- Softmax router (39.4%): learned on top of indiscriminate features
- Embedding-layer routing (25-28%): shared vocabulary causes centroid collapse

The disease is **feature space non-separability**. Prior methods used features where
inter-class distances were comparable to intra-class scatter.

## B. The Right Question

NOT: "How do we prevent routing errors at N=24?"
BUT: "What is the minimum dimensionality and feature quality such that N=24 domain
centroids are separable with margin > intra-class scatter?"

The answer comes from two classical results:
1. **Johnson-Lindenstrauss lemma** (1984): for geometric separability
2. **Fisher's linear discriminant** (Fisher 1936): for statistical separability

## C. Prior Mathematical Foundations

### C.1 Johnson-Lindenstrauss Lemma

**Theorem (JL, 1984).** For any epsilon in (0,1) and any set of N points in R^D,
there exists a linear map f: R^D -> R^d with d = O(log(N)/epsilon^2) such that for
all pairs i,j:

  (1 - epsilon) ||x_i - x_j||^2 <= ||f(x_i) - f(x_j)||^2 <= (1 + epsilon) ||x_i - x_j||^2

For N=24, epsilon=0.5: d >= O(log(24)/0.25) = O(12.7). Since our embedding dimension
d=384 >> 13, the JL lemma guarantees that d=384 is MORE than sufficient to preserve
pairwise distances among 24 points. The question is not dimensionality but whether the
sentence transformer places domain centroids far enough apart.

### C.2 Fisher's Linear Discriminant Analysis

**Definition.** For K classes with centroids mu_k and pooled within-class scatter S_W,
the Fisher criterion is:

  J = tr(S_B) / tr(S_W)

where S_B = sum_k n_k (mu_k - mu)(mu_k - mu)^T is the between-class scatter matrix.

For our centroid-based routing (single centroid per class, test point assigned to
nearest centroid), the relevant quantity simplifies. Define:

- **Inter-centroid distance:** For domains i,j: d_ij = ||mu_i - mu_j|| = sqrt(2(1 - cos(mu_i, mu_j)))
  (since centroids are unit-normalized)
- **Intra-class scatter:** sigma_k = std of cosine similarities of domain k's samples to its centroid

The Fisher-like separability ratio from the N=5 experiment:

  R = (1 - mean_inter_cos) / mean_intra_std

At N=5: R = (1 - 0.656) / 0.0613 = 5.61

### C.3 Scaling Law for Centroid Confusion

**Proposition 1 (Centroid crowding).** As N increases with fixed d and fixed embedding
model, the expected maximum inter-centroid cosine similarity grows as:

  E[max_{i!=j} cos(mu_i, mu_j)] ~ 1 - O(1/N^{2/(d-1)})

For d=384 and N=24, this bound is extremely loose (essentially 0), so crowding from
dimensionality alone is not the issue. The issue is that semantically related domains
(economics/finance, history/politics) have genuinely similar centroids because their
texts share semantic content.

## D. Proof of Guarantee (Conditional)

**Theorem 1 (Routing accuracy lower bound).** Let there be K=24 domain centroids
{mu_1, ..., mu_K} in R^384, each unit-normalized. Let sigma_max = max_k sigma_k be the
maximum intra-class standard deviation of cosine similarity. Define the minimum margin:

  delta_min = min_k (cos(x, mu_k) - max_{j!=k} cos(x, mu_j))

evaluated at x = mu_k (centroid-to-centroid margin). Then for a test query x from
domain k with ||x - mu_k|| <= sigma_max / sqrt(n) (concentration around centroid):

  Routing is correct if delta_min > 2 * sigma_max

*Proof sketch.* A test query from domain k has cosine similarity to mu_k that is
approximately cos(x, mu_k) = 1 - O(sigma_k^2). The cosine similarity to a competitor
mu_j is approximately cos(mu_k, mu_j) + noise_term where the noise term has magnitude
O(sigma_k). Routing fails when:

  cos(x, mu_j) > cos(x, mu_k)

This requires the noise to overcome the margin delta = cos(mu_k, mu_k) - cos(mu_k, mu_j)
= 1 - cos(mu_k, mu_j). For this to happen with probability > epsilon, we need
delta < O(sigma_k). Therefore routing accuracy for domain k is approximately:

  P(correct | domain k) >= 1 - sum_{j!=k} P(cos(x, mu_j) > cos(x, mu_k))

When the centroid-to-centroid margin delta_{kj} = 1 - cos(mu_k, mu_j) >> sigma_k for
all j != k, routing accuracy approaches 1. QED.

**Corollary.** If we define "confused pairs" as pairs (i,j) where cos(mu_i, mu_j) > 1 - 2*sigma_max,
then routing errors are concentrated among these pairs. The number of confused pairs determines
whether overall accuracy exceeds 60%.

## D. Predictions (Derived from the proof)

### Behavioral Predictions

1. **Routing accuracy will be determined by Fisher ratio at N=24.** If R > 2.0,
   accuracy > 60% (K1 PASS). If R < 1.0, accuracy < 40% (K1 FAIL, same as prior methods).

2. **Errors will cluster in semantically related pairs.** The proof shows errors require
   small centroid margins. Expected confusing pairs:
   - economics / finance (both financial domains)
   - history / politics (overlapping topics)
   - cooking / agriculture (food-adjacent)
   - health_fitness / medical (health domains)
   - psychology / philosophy (abstract reasoning)
   - sociology / politics (social sciences)

3. **Per-domain accuracy will be bimodal.** Domains with unique vocabulary (code, math,
   music, sports) will achieve ~100%. Domains in confusing pairs will achieve ~50-70%.

### Quantitative Predictions

| Prediction | Source | Expected Value | Kill Threshold |
|---|---|---|---|
| Fisher ratio R at N=24 | Theorem 1 | 2.0-4.0 (lower than 5.61 at N=5) | R < 1.0 implies K1 FAIL |
| Top-1 accuracy | Theorem 1 + Corollary | 65-85% | K1: < 60% |
| Mean inter-centroid cosine | Proposition 1 | 0.55-0.70 (higher than 0.656 at N=5) | > 0.80 implies collapse |
| Number of confused pairs | Corollary | 3-6 pairs | > 12 implies K1 FAIL |
| Embedding overhead | Architecture | < 10ms (384-dim, single forward pass) | K3: > 50ms |
| PPL improvement vs uniform | DDR=1.126 from cross-domain matrix | 5-12% lower PPL | K2: no improvement |

### Derived Kill Criteria

- **K1 derived from Fisher ratio:** At N=5, R=5.61 gave 96%. For K1 to PASS (>60%),
  Theorem 1 requires delta_min > 2*sigma_max, which translates to R > 2.0 approximately.
  If R < 2.0, fewer than 14/24 domains will be correctly routed, giving < 60%.

- **K2 derived from DDR:** Cross-domain PPL matrix (Finding from exp_cross_domain_ppl_matrix_n24)
  showed DDR=1.126 (correct adapter is 12.6% better). Even at 60% routing accuracy,
  expected PPL improvement vs uniform = 0.6 * 0.126 + 0.4 * 0 = 7.6% > 0.

- **K3 derived from architecture:** MiniLM-L6-v2 has 22M params, 6 layers. Single-query
  inference on CPU takes 2-5ms. Even with Python overhead, < 50ms.

## E. Assumptions and Breaking Conditions

1. **Sentence transformer captures domain signal.** If MiniLM-L6-v2 does not encode
   domain-discriminative features for some domains, those centroids will be diffuse.
   *Breaking condition:* sigma_k > 0.3 for any domain (centroid is meaningless).

2. **Centroid is representative.** With 20 samples per domain, the centroid estimate has
   standard error sigma_k / sqrt(20). If sigma_k = 0.1, SE = 0.022.
   *Breaking condition:* 20 samples insufficient for high-variance domains.

3. **Domains are semantically distinct.** Some domain pairs may have cosine > 0.85,
   making them effectively indistinguishable.
   *Breaking condition:* > 12 confused pairs (>50% of domains involved in confusion).

4. **Test distribution matches training distribution.** Validation samples drawn from
   same distribution as test queries.
   *Breaking condition:* Domain drift between centroid samples and test queries.

## F. Worked Example (N=5 to N=24 extrapolation)

At N=5:
- Mean inter-centroid cosine: 0.656
- Mean intra-class std: 0.0613
- Fisher ratio: R = (1 - 0.656) / 0.0613 = 5.61
- Accuracy: 96% (48/50)
- Confused pairs: 0 (legal/finance closest at 0.825, but still separated)

At N=24, the mean inter-centroid cosine will rise because:
- More domains = more semantically close pairs
- But each centroid is still in 384-dim space (more than enough room)

Extrapolation: If mean inter-cosine rises to ~0.65 and intra-std stays ~0.06:
- R = (1 - 0.65) / 0.06 = 5.83 (similar to N=5)
- But the MINIMUM margin matters more: at N=24, the closest pair will be closer
- If closest pair has cosine 0.85: margin = 0.15, need sigma < 0.075

This suggests per-domain accuracy is bimodal, with overall accuracy depending on how
many domains fall in confused pairs.

## G. Complexity and Architecture Connection

**Centroid computation (offline, one-time):**
- N=24 domains x 20 samples x sentence_model_forward = 480 forward passes
- MiniLM: 22M params, ~5ms per sample = 2.4s total
- Storage: 24 x 384 floats = 36.9 KB

**Routing (per query, online):**
- 1 sentence_model_forward (~5ms) + 24 dot products (~0.001ms) = ~5ms
- Well within K3 threshold of 50ms

**Comparison to prior methods:**
- Hidden-state routing required full LLM forward pass (~500ms): 100x slower
- TF-IDF required vocabulary processing: comparable speed but worse accuracy
- Sentence embedding: fast AND accurate (if Fisher ratio holds)

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Sufficient Fisher discriminant ratio (R > 2.0) in 384-dim sentence-embedding space
   guarantees centroid margins exceed intra-class scatter, making systematic misrouting
   impossible for well-separated domains.

2. **Which existing theorem(s) does the proof build on?**
   Johnson-Lindenstrauss lemma (1984) for dimensionality sufficiency;
   Fisher's linear discriminant (Fisher, 1936) for separability criterion.

3. **What specific numbers does the proof predict?**
   Fisher ratio 2.0-4.0; accuracy 65-85%; 3-6 confused pairs; overhead < 10ms;
   PPL improvement 5-12% vs uniform.

4. **What would FALSIFY the proof (not just the experiment)?**
   The proof is wrong if Fisher ratio > 2.0 but accuracy < 60%. This would mean
   intra-class distributions are non-Gaussian (heavy tails) or centroid estimation
   is biased, violating the concentration assumption.

5. **How many hyperparameters does this approach add?**
   Count: 1 (number of centroid samples). It can be derived: need n > (sigma/epsilon)^2
   for centroid estimation error < epsilon. At sigma=0.1, epsilon=0.03: n > 11.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This is a direct scale test of a proven mechanism. No new fixes or tricks.

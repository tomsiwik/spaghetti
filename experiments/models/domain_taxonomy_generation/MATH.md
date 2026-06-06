# Domain Taxonomy Generation: Mathematical Foundations

## Problem Statement

Given a frozen base model B and a composition budget of N LoRA experts, select
N leaf domains from a hierarchical taxonomy such that:
1. Each domain produces a DISTINCT expert (measurably different from base)
2. Domains have MINIMAL OVERLAP (maximizing coverage per expert slot)
3. The taxonomy is SCALABLE (principled addition from 50 to 500 to 5000)

## Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| D | Set of all possible domains | |D| = N_max |
| d_i | Domain i with description text t_i | |
| e(t_i) | Embedding of description t_i | R^d_emb (d_emb=384 for MiniLM) |
| S_ij | Cosine similarity between embeddings | [-1, 1] |
| T = (V, E) | Taxonomy tree | V = domains, E = parent-child |
| L(T) | Leaf set of taxonomy | L subset V |
| cat(d_i) | Category of domain d_i | level-2 node in T |
| PPL_B(d_i) | Base model perplexity on domain i | R+ |
| PPL_E(d_i) | Expert perplexity on domain i | R+ |
| Delta_i | Improvement: 1 - PPL_E/PPL_B | [0, 1] typically |

## Overlap Metric

For two domains d_i, d_j with description embeddings e(t_i), e(t_j):

    S_ij = cos(e(t_i), e(t_j)) = (e(t_i) . e(t_j)) / (||e(t_i)|| * ||e(t_j)||)

The OVERLAP FRACTION at threshold tau is:

    Overlap(tau) = |{(i,j) : i < j, S_ij > tau}| / C(N, 2)

where C(N, 2) = N(N-1)/2 is the total number of unique pairs.

## Kill Criteria Formalization (Revised)

Original criteria were vacuous (3000x margins). Tightened per adversarial review:

**K1 (tightened)**: Overlap(0.5) <= 0.05
- At most 5% of domain pairs may have embedding cosine > 0.5
- Old: Overlap(0.7) <= 0.30 (passed with 3000x margin -- useless)

**K2 (tightened)**: |{d_i : max_{j != i} S_ij > 0.7}| / N <= 0.05
- At most 5% of domains may have their most-similar neighbor above 0.7
- Old: threshold 0.85, limit 20% (passed with infinite margin -- useless)

**Negative control validation**: A good metric must FAIL on a deliberately
bad taxonomy (270 paraphrases of 30 domains). If both pass, the metric has
no discriminative power. K2 discriminates (bad taxonomy: 72.2% FAIL);
K1 does not (bad taxonomy: 2.79% PASS).

## Taxonomy Design Principles

The taxonomy is a rooted tree T = (V, E) with:
- Level 0: Root (1 node)
- Level 1: Supercategories (k_1 nodes, k_1 = 6)
- Level 2: Categories (k_2 nodes, k_2 = 35)
- Level 3: Leaf domains (N nodes)

**Uniformity constraint**: For each category c at level 2,
    |children(c)| in [5, 10]
This prevents extremely fine-grained categories (1-2 leaves) and
extremely broad ones (>10 leaves that should be split).

**Distinguishability within category**: For siblings d_i, d_j under same category:
    S_ij < 0.85 (must be semantically distinguishable)

**Cross-category gap**: Expected within-category similarity should exceed
cross-category similarity:
    E[S_ij | cat(d_i) = cat(d_j)] > E[S_ij | cat(d_i) != cat(d_j)]

This ratio quantifies how well the taxonomy captures semantic structure.

## Scaling Analysis

Number of pairs scales as O(N^2). For N domains:
- N=50: 1,225 pairs
- N=270: 36,315 pairs
- N=500: 124,750 pairs
- N=5000: 12,497,500 pairs

If overlap fraction is O(k^2/N^2) where k is the number of naturally
confusable concept clusters, then overlap fraction DECREASES as N grows
(adding distinct domains dilutes the confusable-pair density).

## Prediction of Expert Distinctness

A domain d_i is predicted to produce a DISTINCT expert if:
1. max_{j != i} S_ij < 0.7 (not too similar to any existing domain)
2. The domain description is specific enough to generate focused training data

**Empirical validation against pilot-50 (honest assessment):**

The correlation between embedding overlap and expert PPL improvement was
tested against pilot-50 ground truth:
    r(max_sibling_cos, improvement) = 0.034 (p=0.813)
    r(max_any_cos, improvement) = 0.028 (p=0.846)

This near-zero correlation means embedding similarity does NOT predict
expert quality. The proxy validates only that domain NAMES are semantically
distinct -- not that domain EXPERTS will be distinct or useful. The
embedding proxy is a necessary-but-not-sufficient condition: domains
identical in embedding space would produce similar experts, but distinct
embeddings do not guarantee distinct or useful experts.

## Worked Example

For N=270 domains at d_emb=384:
- Total pairs: 270*269/2 = 36,315
- Mean cosine: 0.133

Tightened criteria:
- K1: 0.45% of pairs > 0.5 -> PASS (threshold 5%, margin 11x)
- K2: 3.0% of domains NN > 0.7 -> PASS (threshold 5%, margin 1.7x)

Negative control (30 x 9 paraphrases):
- K1: 2.79% of pairs > 0.5 -> PASS (K1 does not discriminate)
- K2: 72.2% of domains NN > 0.7 -> FAIL (K2 discriminates, 24.4x separation)

Within-category mean cosine: 0.338 (n=943 pairs)
Cross-category mean cosine: 0.128 (n=35,372 pairs)
Ratio: 2.65x (taxonomy captures semantic structure)

## Assumptions

1. Sentence-transformer embeddings (all-MiniLM-L6-v2) capture domain
   semantic similarity relevant to LoRA expert distinction
2. Description-based similarity is a NECESSARY but NOT SUFFICIENT proxy for
   actual weight-space cosine between trained experts. The r=0.034 correlation
   with pilot-50 shows it is not sufficient. It remains useful as a filter:
   domains identical in embedding space would produce similar experts.
3. The taxonomy can be extended by adding sibling domains to existing
   categories without disrupting the existing structure. However, K2 margin
   is only 1.7x, so adding fine-grained domains risks pushing it over.
4. Expert distinctness from base is primarily determined by domain
   specificity and base model weakness, not by inter-domain embedding similarity.

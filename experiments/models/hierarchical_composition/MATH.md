# Hierarchical Expert Composition: Mathematical Foundations

## 1. Setup

We have N domain experts organized into C clusters. Each cluster c contains
n_c domains (sum n_c = N). The question: does a two-level hierarchy
(foundation + specialist) outperform flat composition for within-cluster
cross-domain queries?

**Notation:**
- W_base: frozen base model weights (d x d per layer)
- Delta_k: flat expert k's weight delta (rank r_flat)
- Delta_c^F: foundation delta for cluster c (rank r_f)
- Delta_k^S: specialist delta for domain k within cluster c (rank r_s)
- r_flat = r_f + r_s (equalized total rank budget per domain)
- L(W, Q): NTP loss on query Q
- C: number of clusters
- n_c: domains in cluster c

## 2. Composition Architectures

### 2.1 Flat Composition (Baseline)

Each domain k has a single rank-r_flat delta:

  W_flat(S) = W_base + sum_{k in S} w_k * Delta_k

where S is the set of active experts and w_k are weights (equal or PPL-probe).

Total rank budget per domain: r_flat.
Total parameters per domain-expert: O(r_flat * d) per weight matrix.

### 2.2 Hierarchical Composition

Each domain k in cluster c has two components:
- Foundation: Delta_c^F (rank r_f), shared across all domains in cluster c
- Specialist: Delta_k^S (rank r_s), specific to domain k

Composition for a query involving domains i, j in the same cluster c:

  W_hier(i, j) = W_base + Delta_c^F + (w_i * Delta_i^S + w_j * Delta_j^S)

For a cross-cluster query involving domain i in cluster c1 and j in cluster c2:

  W_hier(i, j) = W_base + alpha_1 * Delta_{c1}^F + alpha_2 * Delta_{c2}^F
                 + w_i * Delta_i^S + w_j * Delta_j^S

where alpha_1, alpha_2 are cluster weights.

Total rank budget per domain: r_f + r_s = r_flat (equalized).

### 2.3 Training Protocol

**Flat:** Train each expert from base on domain-specific data, SVD-truncate
to rank r_flat.

**Hierarchical:**
1. Train full-rank expert on each domain (same as flat)
2. For each cluster c, compute shared subspace:
   - Stack deltas from all domains in cluster c
   - SVD of concatenated deltas to extract top-r_f shared directions
   - This is the foundation delta: Delta_c^F
3. For each domain k in cluster c, compute residual:
   - Delta_k^S = project(Delta_k - Delta_c^F, rank=r_s)
   - The specialist captures what the foundation missed

## 3. Why Hierarchy Might Help

### 3.1 Shared Subspace Hypothesis

Prior result (exp_orthogonality_by_domain_type): within-cluster |cos| is
7.84x higher than cross-cluster. This means domains in the same cluster
share a significant subspace.

If Delta_i and Delta_j (same cluster) share a subspace U of dimension r_shared:

  Delta_i = U * S_i + V_i   (shared + unique)
  Delta_j = U * S_j + V_j   (shared + unique)

Flat composition averages everything:
  (Delta_i + Delta_j) / 2 = U * (S_i + S_j)/2 + (V_i + V_j)/2

The shared component is preserved at half strength. The unique components
interfere.

Hierarchical composition separates shared and unique:
  Delta_c^F + (w_i * Delta_i^S + w_j * Delta_j^S)

The foundation captures U at full strength. Specialists capture V_i, V_j
with weights. No dilution of the shared component.

### 3.2 Rank Efficiency Argument

For K same-cluster experts with shared subspace of dimension r_shared:

**Flat:** Each expert uses r_flat ranks. Shared subspace is represented
K times (wasteful). Effective unique capacity: r_flat - r_shared per expert.

**Hierarchical:** Foundation uses r_f >= r_shared once. Each specialist
uses r_s for unique content. Effective unique capacity: r_s per expert.

If r_shared > 0, hierarchy is more rank-efficient: it represents the shared
subspace once instead of K times.

### 3.3 When Hierarchy Hurts

For cross-cluster queries, the hierarchy adds the wrong foundation. If
domains i (cluster 1) and j (cluster 2) are queried together, applying
both foundations may inject noise from the irrelevant foundation.

Prediction: hierarchy helps within-cluster, hurts across-cluster.

## 4. Kill Criteria Formalization

**K1:** Let G_flat and G_hier be the mean cross-domain loss gaps vs base.

  K1_pass iff G_hier < G_flat  (hierarchy improves quality)

Specifically, measure on within-cluster cross-domain queries:

  G_within_flat = mean over within-cluster pairs of [(L_flat - L_base) / L_base]
  G_within_hier = mean over within-cluster pairs of [(L_hier - L_base) / L_base]

  K1_pass iff G_within_hier < G_within_flat

**K2:** Complexity overhead:

  Overhead = (training_time_hier - training_time_flat) / training_time_flat

  K2_pass iff Overhead <= 0.30 OR quality_improvement >= 5%

  where quality_improvement = G_within_flat - G_within_hier

## 5. Experimental Design

- d=64, H=4, L=2 (micro transformer)
- 5 domains: arithmetic, parity (cluster "symbolic"), reverse, repeat, sort (cluster "string")
- r_flat = 8 per expert
- r_f = 4 (foundation), r_s = 4 (specialist) -> same total rank = 8
- 5 seeds, 200 train / 50 test per domain
- Cross-domain types: 10 pairs (4 within-cluster, 6 across-cluster)
- Composition strategies: equal_weight and ppl_probe for both flat and hierarchical
- Additional controls:
  - flat_rank4: 5 experts at rank-4 (lower bound, same as specialist-only)
  - monolithic_rank8: single expert trained on all data at rank-8

## 6. Worked Example (d=64, r_f=4, r_s=4)

Consider cluster "string" with domains reverse and sort.

Training phase:
1. Train reverse expert: Delta_reverse (full rank, then truncate to rank-8 for flat)
2. Train sort expert: Delta_sort (full rank, then truncate to rank-8 for flat)
3. Foundation: SVD of [flatten(Delta_reverse); flatten(Delta_sort)] -> top-4 directions
4. Specialist_reverse: project(Delta_reverse - reconstruct(foundation), rank=4)
5. Specialist_sort: project(Delta_sort - reconstruct(foundation), rank=4)

Inference on cross-domain query "dcba>abcd>abcd" (reverse + sort):

Flat: W_base + (Delta_reverse_r8 + Delta_sort_r8) / 2
  - Shared "string manipulation" signal diluted by 0.5x
  - Unique signals also diluted

Hierarchical: W_base + Delta_string^F + (0.5 * Delta_reverse^S + 0.5 * Delta_sort^S)
  - Shared "string manipulation" at full strength via foundation
  - Only unique parts diluted

Expected: hierarchy recovers ~r_shared/(2*r_flat) more signal, or ~25% if r_shared=4.

## 7. Assumptions

1. Within-cluster domains share a meaningful subspace (supported by exp_orthogonality_by_domain_type: 7.84x cos within vs across)
2. SVD of stacked deltas can extract this shared subspace
3. Rank-4 + rank-4 hierarchy captures comparable information to rank-8 flat
4. Foundation trained on cluster data generalizes to unseen within-cluster combinations
5. Micro-scale results (d=64, synthetic data) are directional for macro (d=4096, real data)

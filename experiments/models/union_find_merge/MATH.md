# Union-Find Expert Merging: Mathematical Foundations

## Variables and Notation

| Symbol | Shape/Type | Description |
|--------|-----------|-------------|
| P | scalar | Number of capsules per layer |
| P_total | scalar | Total capsules across all layers |
| d | scalar | Model embedding dimension (d=64 at micro) |
| N | scalar | Number of composed domain pools |
| a_i | (d,) | Detector vector for capsule i (row of A) |
| b_i | (d,) | Expansion vector for capsule i (column of B) |
| A | (P, d) | Detector matrix |
| B | (d, P) | Expansion matrix |
| J(i,j) | scalar in [0,1] | Co-activation Jaccard similarity |
| rho(i,j) | scalar in [-1,1] | Output correlation |
| tau_J | scalar | Jaccard threshold for union |
| tau_rho | scalar | Output correlation threshold for union |
| UF | data structure | Union-find with path compression |

## Union-Find Data Structure (Tarjan 1975)

The union-find (disjoint-set) maintains a forest of trees where each
tree represents a set of equivalent capsules. Two operations:

**Find(x)**: Return the canonical representative (root) of x's tree.
With path compression, all nodes on the path to root are linked directly
to root:

```
Find(x):
  if parent[x] != x:
    parent[x] = Find(parent[x])   // path compression
  return parent[x]
```

**Union(x, y)**: Merge the sets containing x and y. Union by rank
keeps trees balanced:

```
Union(x, y):
  rx = Find(x), ry = Find(y)
  if rx == ry: return False
  if rank[rx] < rank[ry]: swap(rx, ry)
  parent[ry] = rx
  if rank[rx] == rank[ry]: rank[rx] += 1
  return True
```

**Complexity**: Both operations run in O(alpha(n)) amortized time, where
alpha is the inverse Ackermann function (effectively constant, alpha(n) <= 4
for all practical n).

## Similarity Metrics (from behavioral_dedup)

**Co-activation Jaccard**:
$$J(i,j) = \frac{|\text{fire}_i \cap \text{fire}_j|}{|\text{fire}_i \cup \text{fire}_j|}$$

where fire_i is the set of input positions where capsule i activates (h_i > 0).

**Output correlation**:
$$\rho(i,j) = \frac{\sum_n h_{n,i} h_{n,j} \cdot (b_i \cdot b_j)}{\sqrt{\sum_n h_{n,i}^2 \|b_i\|^2} \sqrt{\sum_n h_{n,j}^2 \|b_j\|^2}}$$

## Merging Protocol

For a cluster C = {i_1, i_2, ..., i_k} identified by union-find:

**Detector merge (a-average)**:
$$a_C = \frac{1}{k} \sum_{j \in C} a_j$$

**Expansion merge (b-sum)**:
$$b_C = \sum_{j \in C} b_j$$

Rationale: The composed output is $\sum_{j \in C} h_j \cdot b_j$. If capsules
in C fire on similar inputs (high Jaccard), their activations h_j are
correlated. Summing b vectors approximates the total contribution. Averaging a
vectors finds the "centroid" detector.

## Key Difference: Transitive Closure

**Greedy pairing** (behavioral_dedup): Each capsule merges at most once.
If A~B and B~C, only one pair merges (whichever has higher similarity).
Maximum compression per layer: P/2.

**Union-find transitive closure**: If A~B and B~C, all three merge even
if sim(A,C) < threshold. The transitivity can chain: if A~B, B~C, C~D, ...,
the entire chain merges. Maximum compression per layer: P -> 1.

## Computational Cost

- Profiling: O(P^2 * n_batches * B * T) for Jaccard and correlation matrices
- Union-find construction: O(P^2 * alpha(P)) ~ O(P^2)
- Merging: O(k * d) per cluster where k is cluster size
- Total: Dominated by profiling, same as behavioral_dedup

## Worked Example (d=64, P=512, N=2)

Given 2 composed domains with P=256 each (P_total = 512 per layer):

1. Profile Jaccard on 20 batches of 32 sequences
2. In Layer 0: mean Jaccard J=0.527 (from behavioral_dedup findings)
3. At threshold tau_J=0.3: nearly all capsules in Layer 0 are transitively
   connected (475/512 merge into one giant cluster)
4. This destroys Layer 0's representation: one merged capsule cannot
   reproduce the function of 475 individual capsules

## Why Transitive Closure Fails Here

The critical failure mode: Layer 0 capsules are loosely similar to many
neighbors (mean Jaccard ~0.5). Greedy pairing merges only the most similar
pairs. But union-find chains transitivity: A~B at J=0.35, B~C at J=0.32,
C~D at J=0.31, ... The entire layer becomes one connected component.

The end-to-end similarity between A and the final element in the chain may
be very low (J < 0.1), but transitive closure still merges them. This is
mathematically correct for equivalence relations but WRONG for approximate
similarity, which is not transitive:

$$\text{sim}(A,B) > \tau \text{ and } \text{sim}(B,C) > \tau \not\Rightarrow \text{sim}(A,C) > \tau$$

This is a well-known problem in clustering: single-linkage clustering
(which union-find implements) produces "chaining" artifacts.

## Assumptions

1. **Similarity is transitive** -- FALSIFIED. Co-activation Jaccard is not
   transitive. Two capsules can both be similar to a third without being
   similar to each other.
2. **Weight merging preserves function** -- PARTIALLY TRUE. For small clusters
   (2-3 capsules), a-average/b-sum preserves quality within 3%. For large
   clusters (400+ capsules), the merged capsule cannot reproduce the
   original function.
3. **Layer 0 redundancy enables compression** -- TRUE but NOT via transitive
   closure. Layer 0 has high redundancy (J=0.527 mean) but needs per-pair
   merging, not connected-component merging.

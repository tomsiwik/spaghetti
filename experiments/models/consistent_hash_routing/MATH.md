# Consistent Hash Routing: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| N | scalar | Number of expert groups |
| d | scalar | Embedding dimension |
| k | scalar | Number of experts selected per token (top-k) |
| V | scalar | Number of virtual nodes per expert on the ring |
| x | (B, T, d) | Token hidden states |
| p | (d,) | Fixed random projection vector (frozen, not trainable) |
| s | scalar | Ring position: hash(x @ p) mapped to [0, 2^32) |
| R | set of (pos, idx) | Hash ring: sorted set of (ring_position, expert_index) tuples |
| w_i | scalar | Routing weight for expert i |

## Hash Ring Construction

Each expert i in {0, ..., N-1} is placed on a ring [0, 2^32) at V virtual
node positions. For expert i, virtual node v:

    ring_pos(i, v) = FNV1a(i || v)

where || denotes concatenation of the 4-byte big-endian representations,
and FNV1a is the 32-bit Fowler-Noll-Vo hash variant 1a.

The ring R is the sorted set of all (ring_pos(i,v), i) tuples:

    R = sort({ (ring_pos(i, v), i) : i in [0,N), v in [0,V) })

Total ring size: |R| = N * V.

## Token-to-Ring Mapping

For a token with hidden state x in R^d:

1. **Project**: s_float = x @ p, where p ~ N(0, 1/sqrt(d)) is fixed
2. **Quantize**: s_int = FNV1a(float_bytes(s_float)) in [0, 2^32)
3. **Lookup**: Find insertion point in R via binary search on s_int

## Expert Selection (Top-k Clockwise)

Starting from s_int, walk clockwise on the ring collecting distinct experts:

    selected = {}
    for offset in 0, 1, 2, ...:
        idx = (insertion_point + offset) mod |R|
        expert = R[idx].expert_idx
        if expert not in selected:
            selected[expert] = R[idx].ring_pos
        if |selected| >= k:
            break

This selects the k nearest distinct experts clockwise from s_int.

## Routing Weights

For selected experts {e_1, ..., e_k} with ring distances {d_1, ..., d_k}:

    d_j = (R[e_j].ring_pos - s_int) mod 2^32

    w_j = exp(1/(d_j + 1)) / sum_{j'} exp(1/(d_{j'} + 1))

This gives higher weight to closer experts on the ring.

## Displacement Analysis

### Theorem (Karger et al. 1997, adapted)

When adding the (N+1)-th expert to a ring with N experts and V virtual
nodes each, the expected fraction of tokens that change their primary
expert assignment is:

    E[displacement] = 1/(N+1)

**Proof sketch**: The new expert's V virtual nodes partition the ring
into V arcs. Each arc steals tokens from the expert that previously
owned that arc segment. With V sufficiently large, the total arc length
owned by the new expert approaches 1/(N+1) of the ring circumference.
Tokens are mapped to ring positions via a hash function with
approximately uniform distribution, so ~1/(N+1) of tokens are
remapped.

### For top-k routing

With top-k selection, a token is "displaced" if any of its k selected
experts changes. The displacement for top-k is bounded by:

    E[displacement_topk] <= k * E[displacement_top1] = k/(N+1)

For our parameters (k=2, N=8): E[displacement] <= 2/9 = 22.2%.

The actual displacement is lower because the events are not independent:
a token's top-2 selection only changes if the new expert is closer than
one of its current top-2, and the two events are correlated.

## Computational Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Ring construction | O(N * V * log(N * V)) | O(N * V) |
| Token routing (per token) | O(d + log(N * V) + k * V) | O(1) |
| Add expert | O(V * log(N * V)) | O(V) |
| Remove expert | O(N * V) | O(0) |

For our parameters (d=64, N=8, V=150, k=2):
- Ring size: 1200 entries
- Per-token: 64 MADs (projection) + 11 comparisons (binary search) + ~300 ring walk
- Add expert: 150 sorted insertions

Comparison with softmax routing:
- Softmax: O(N * d) = 512 MADs per token (dot product with all N expert keys)
- Consistent hash: O(d) + O(log(NV)) ~ 64 + 11 = 75 operations

At large N, consistent hash is asymptotically cheaper: O(d + log N) vs O(N * d).

## Worked Example (d=4, N=4, V=3, k=2)

Ring construction:
```
Expert 0, virtual 0: FNV1a(00000000|00000000) = 0x050C5D1F -> pos 84,860,191
Expert 0, virtual 1: FNV1a(00000000|00000001) = 0x41B5A127 -> pos 1,102,840,103
Expert 0, virtual 2: FNV1a(00000000|00000002) = 0x5B70F52F -> pos 1,534,227,759
Expert 1, virtual 0: FNV1a(00000001|00000000) = 0x4F1E6D18 -> pos 1,327,267,096
...
```
Sorted ring (12 entries):
```
pos: [84M, 341M, 512M, 789M, 1.1B, 1.3B, 1.5B, 2.1B, 2.4B, 2.8B, 3.1B, 3.9B]
exp: [0,    2,    3,    1,    0,    1,    0,    3,    2,    1,    2,    3   ]
```

Token x = [0.5, -0.3, 0.8, 0.1], p = [0.25, -0.5, 0.1, 0.3]:
- s_float = 0.5*0.25 + (-0.3)*(-0.5) + 0.8*0.1 + 0.1*0.3 = 0.385
- s_int = FNV1a(bytes(0.385)) ~ 1.8B (hypothetical)
- Binary search finds insertion between pos 1.5B and 2.1B
- Walk clockwise: expert 3 (pos 2.1B), expert 2 (pos 2.4B)
- Selected: {3, 2} with distances {300M, 600M}
- Weights: softmax([1/300M, 1/600M]) ~ [0.67, 0.33]

Now add Expert 4 with virtual nodes at positions {1.7B, 2.3B, 3.5B}:
- Token s_int=1.8B: walk clockwise from 1.8B
- First distinct: Expert 4 at 2.3B (was expert 3 at 2.1B... wait, 2.1B < 2.3B)
- Actually expert 3 still at 2.1B, then expert 4 at 2.3B instead of expert 2 at 2.4B
- Only the 2nd-choice expert changed: displacement of the top-2 set.

## Assumptions

1. **Hash uniformity**: FNV1a produces approximately uniform distribution
   over [0, 2^32). Validated empirically in tests (chi-squared on 10K samples).

2. **Projection preserves some locality**: Similar hidden states produce
   similar scalar projections, so similar tokens route to similar experts.
   This is a weaker guarantee than LSH (no collision probability bound),
   but sufficient for routing where quality is not sensitivity to routing
   precision (proven by LSH experiment).

3. **Virtual nodes sufficient for balance**: V=150 per expert with N=8
   gives 1200 ring entries. With uniform hashing, each expert gets
   ~1/N fraction of ring arcs. Balance improves with V.

4. **Deterministic routing**: The hash function is deterministic, so the
   same hidden state always routes to the same experts. This eliminates
   stochasticity in routing (unlike dropout-based approaches).

# Bloom Filter Pre-Filtering: Mathematical Foundations

## 1. Standard Bloom Filter

### 1.1 Definition

A Bloom filter is a probabilistic data structure for approximate set
membership testing. It consists of:

```
m     = number of bits in the bit array
k     = number of independent hash functions
S     = the set of inserted elements
n     = |S|, number of elements inserted
```

Each hash function h_i: U -> {0, 1, ..., m-1} maps elements from a
universe U to bit positions.

### 1.2 Operations

**Insert(x):** For each i in 1..k, set bit[h_i(x)] = 1.

**Query(x):** Return True iff bit[h_i(x)] = 1 for ALL i in 1..k.

### 1.3 Guarantees

- **Zero false negatives:** If x was inserted, Query(x) = True (all k bits
  were set during insertion, bits are never cleared).

- **Bounded false positives:** If x was NOT inserted, Query(x) may still
  return True (all k bit positions happen to be set by other insertions).

The false positive rate (FPR) after n insertions into m bits with k hash
functions:

```
FPR = (1 - (1 - 1/m)^(k*n))^k
    approx (1 - e^(-k*n/m))^k
```

### 1.4 Optimal k

For given m and n, the FPR is minimized at:

```
k_opt = (m/n) * ln(2)
```

giving FPR_opt = (1/2)^k = (0.6185)^(m/n).

### 1.5 Notation for Expert Routing

```
d         = embedding dimension (default 64)
G         = number of expert groups (default 8)
m         = bits per Bloom filter (parameter to sweep)
k         = hash functions per filter (default 4)
n_g       = number of tokens inserted into filter g
d_h       = dimensions used for hashing (default 8)
tau       = activation threshold for profiling
```

---

## 2. Application to Expert Routing

### 2.1 Profiling Phase

After training, profile token-expert activation patterns:

```
For each profiling batch (x, _):
    For each layer l:
        h = post-norm hidden state at layer l
        For each group g in 0..G-1:
            act_g = mean(|CapsuleGroup_g(h)|)    (L1 activation magnitude)
            If act_g > tau:
                Insert quantize(h) into bloom_g
```

Where quantize(h) maps the continuous hidden state to an integer key:

```
quantize(x) = concat_j( floor((clip(x_j, -4, 4) + 4) / 8 * 256) )
              for j in 0..d_h-1
```

This produces a discrete key from the first d_h dimensions of x.

### 2.2 Inference Phase

```
For each token x:
    survivor_mask = []
    For each group g:
        key = quantize(x)
        If bloom_g.query(key):
            survivor_mask[g] = True   (group MAY be relevant)
        Else:
            survivor_mask[g] = False  (group DEFINITELY irrelevant)

    # Only route over survivors
    scores = W_r[survivors] @ x
    selected = top_k(softmax(scores), k=2)
```

### 2.3 Computational Cost

**Profiling:** O(n_profile * G * d) for computing group activations, plus
O(n_profile * G * k) for Bloom filter insertions.

**Inference per token:** O(G * k) for Bloom filter queries (independent of d),
plus O(|survivors| * d) for softmax routing over survivors only.

**Memory:** m * G bits per layer. At m=100K, G=256: 3.2 MB per layer.
At m=256, G=256: 8 KB per layer (negligible).

---

## 3. The Fundamental Problem: Exact vs Approximate Membership

### 3.1 Why Bloom Filters Fail for Neural Routing

Bloom filters provide **exact membership testing** -- they answer "was this
exact key ever inserted?" Neural network hidden states are **continuous-valued**
and live in a smooth representation space. Two tokens that should route
to the same expert produce **similar but not identical** hidden states.

The quantization function:

```
quantize(x) = concat_j( floor((x_j + 4) / 8 * 256) )
```

Maps a continuous vector to a discrete key. Two vectors x and x' that differ
by epsilon in any dimension may produce **different keys** (if they fall in
different quantization bins). This means:

- Token x profiled during training: quantize(x) = key_1, inserted into bloom_g
- Token x' from validation set: quantize(x') = key_2, NOT in bloom_g
- Even if x and x' should route to the same expert

This is NOT a false negative of the Bloom filter (the filter correctly
reports that key_2 was never inserted). It is a **semantic gap** between
exact membership testing and the approximate similarity needed for routing.

### 3.2 Quantification

With d_h=8 dimensions quantized to 256 bins each, the key space has
256^8 = 1.8 * 10^19 possible keys. Even with aggressive quantization
to 8 bins (3 bits per dimension), the space is 8^8 = 16.7M keys.

After profiling n_profile = 20,480 tokens (20 batches x 32 x 32),
we insert at most 20,480 distinct keys per group. The coverage of the
key space is:

```
coverage = n_profiled / |key_space| = 20480 / 16.7M = 0.12%
```

This means 99.88% of possible quantized representations have NEVER been
profiled. For any validation token that falls into an unprofiled
quantization bin, the Bloom filter correctly returns "not seen" -- even
if a very similar token WAS profiled.

### 3.3 The Saturation Regime

With small m (e.g., m=256 bits), each insertion sets k=4 bits. After
inserting n keys, the expected number of set bits is:

```
E[bits_set] = m * (1 - (1 - 1/m)^(k*n))
```

At n=20000, k=4, m=256: E[bits_set] = 256 * (1 - (1-1/256)^80000) = 256.
The filter is **completely saturated** -- all 256 bits are set, and the
filter returns True for every query (FPR = 100%).

This explains the initial experimental result: at m=256, elimination = 0%.

At m=100,000: E[bits_set] = 100000 * (1 - e^(-80000/100000)) = 55,067.
Fill ratio = 55%, FPR = (0.55)^4 = 9.1%. The filter is functional.

At m=1,000,000: E[bits_set] ~ 76,883. Fill ratio = 7.7%, FPR ~ 0%.
The filter is effective but oversized.

### 3.4 The Dilemma

Even with sufficient capacity (m=100K), the Bloom filter eliminates
expert groups based on **exact key matching**:

- If profiling covers all possible key patterns: elimination reflects
  true expert specialization (groups that genuinely never fire for
  certain patterns are eliminated)
- If profiling coverage is sparse (as it always is): elimination also
  catches tokens that are **similar to profiled patterns but quantize
  differently**, incorrectly excluding them

Empirically, with m=100K and threshold=0.5:
- 74-83% of expert-token pairs are eliminated
- 76-85% of top-k experts that SHOULD fire are eliminated (false negatives
  in the routing sense, though not in the Bloom filter sense)
- Quality degrades 6-11% -- the filter is too aggressive

---

## 4. Comparison with Alternative Approaches

### 4.1 LSH (What Works)

LSH uses random projections to hash similar vectors to the same bucket:

```
h(x) = sign(r^T x),  r ~ N(0, I_d)
P[h(x) = h(y)] = 1 - theta(x,y) / pi
```

This PRESERVES angular similarity -- similar vectors hash to the same
bucket with high probability. Bloom filters have no such locality property.

### 4.2 Why LSH Succeeds Where Bloom Fails

| Property | Bloom Filter | LSH |
|----------|-------------|-----|
| Similarity-preserving | No | Yes |
| False negative guarantee | On exact keys | Probabilistic |
| Generalization | None | Angular locality |
| Suitable for continuous data | No | Yes |

---

## 5. Parameter Count

The model parameters are identical to CapsuleMoEGPT (same architecture).
The Bloom filter bank is NOT a learned component -- it is built during
profiling and stored separately.

```
Per-layer learned params:
  Attention:  4 * d^2 = 16,384
  Router:     G * d = 512
  Capsules:   G * 2 * d * n_c = 32,768
  Layer total: 49,664

Full model (4 layers, V=28):
  All layers:   4 * 49,664 = 198,656
  Embeddings:   28*64 + 32*64 = 3,840
  LM head:      28*64 = 1,792
  Total learned: ~204,160 (identical to CapsuleMoEGPT)

Bloom filter storage (non-learned):
  Per filter: m bits
  Per layer:  G * m bits = 8 * m bits
  Total:      4 * 8 * m bits = 32m bits = 4m bytes
  At m=256:     1 KB (negligible but useless -- saturated)
  At m=100K:    400 KB (functional but quality-killing)
  At m=1M:      4 MB (minimal FPR but still quality-killing)
```

---

## 6. Assumptions (All Violated)

1. **Quantized hidden states form a finite, enumerable set.**
   VIOLATED: continuous hidden states produce an effectively infinite
   number of distinct quantized keys. Profiling cannot cover the space.

2. **Expert group activation is a deterministic function of the
   quantized hidden state.** PARTIALLY VIOLATED: the quantization
   discards information. Two tokens in the same bin may have different
   activations; two tokens in adjacent bins may have identical activations.

3. **Profiling set is representative of inference tokens.**
   VIOLATED: even with training-set profiling, quantization means
   validation tokens frequently fall in unprofiled bins.

4. **Expert groups specialize to distinct token patterns.**
   PARTIALLY MET: at micro scale, most groups fire for most tokens
   (93-100% at threshold=0.1). Deeper layers show some specialization
   (30-45% of tokens for some groups). At macro scale with diverse data,
   specialization should be stronger.

---

## 7. Worked Example

### Setup
d=8, G=4, m=32, k=2, d_h=4, threshold=0.5

### Profiling
Token x = [1.2, -0.5, 0.8, 0.3, ...]
quantize(x) = [192, 109, 153, 134] -> key = hash(192, 109, 153, 134)

Group activations: [0.8, 0.2, 1.1, 0.05]
Groups 0 and 2 are above threshold (0.5).

Insert key into bloom_0 and bloom_2:
  h_1(key) = 7, h_2(key) = 23   -> bloom_0 bits 7, 23 set
  h_1(key) = 7, h_2(key) = 23   -> bloom_2 bits 7, 23 set

### Inference
New token x' = [1.21, -0.49, 0.79, 0.31, ...]
quantize(x') = [192, 109, 153, 134] -> same key (lucky -- same bins)
bloom_0.query(key) = True (bits 7, 23 set)
bloom_2.query(key) = True
bloom_1.query(key) = False (bit 7 not set in bloom_1)
bloom_3.query(key) = False

Survivors: {0, 2}. Softmax routes over groups 0 and 2 only.

### What Goes Wrong
New token x'' = [1.22, -0.51, 0.81, 0.29, ...]
quantize(x'') = [192, 108, 153, 133] -> DIFFERENT key (dim 1 and 3 crossed bin boundary)
bloom_0.query(new_key) = False (different hash positions, not set)

Group 0 is eliminated even though x'' is essentially the same token as x.

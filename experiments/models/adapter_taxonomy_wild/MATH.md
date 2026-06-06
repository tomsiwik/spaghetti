# Adapter Taxonomy in the Wild: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 3584 (Qwen2.5-7B) |
| d_ff | FFN intermediate dimension | 18944 (= 5.29d) |
| r | LoRA rank | 16 (default) |
| L | Number of layers | 28 |
| N | Number of composed experts | variable |
| K | Number of MoE experts (MoLoRA) | variable |
| n_p | Prefix length (prefix tuning) | 10 (typical) |
| k | Bottleneck dimension (Houlsby) | 64 (typical) |

## 2. Composition Algebra

The core question is: given N adapter deltas, how do they compose?

### 2.1 Additive Composition (LoRA family)

For LoRA adapters with deltas dW_i = (alpha/r) * B_i @ A_i:

    W_composed = W_base + sum_{i=1}^{N} dW_i

This is well-defined when deltas are approximately orthogonal:

    <dW_i, dW_j> / (||dW_i|| * ||dW_j||) ~ 0    for i != j

**Proven**: cos ~ 0.0002 at d=896 (macro/ortho_scaling/).

Interference bound: when composing N rank-r adapters additively,

    ||W_composed - W_ideal|| <= sum_{i<j} |<dW_i, dW_j>| / ||dW_i||

For near-orthogonal adapters (cos ~ epsilon), this is O(N^2 * epsilon),
which is negligible for epsilon ~ 10^{-4} even at N ~ 1000.

### 2.2 Multiplicative Composition (IA3)

IA3 learns scaling vectors l_k, l_v, l_ff:

    K' = diag(l_k) @ K
    V' = diag(l_v) @ V
    FFN' = diag(l_ff) @ FFN(x)

Composing two IA3 adapters (l_1, l_2) gives:

    l_composed = l_1 * l_2    (element-wise product)

This is NOT additive. The product of N adapters is:

    l_composed = prod_{i=1}^{N} l_i

Problems:
- Values compound: if l_i[j] = 0.5 for all i, then l_composed[j] = 0.5^N -> 0
- No orthogonality property: there is no analog of cos ~ 0 for products
- Converting to log-space (log(l_composed) = sum log(l_i)) restores additivity
  but requires all l_i > 0 and changes the optimization landscape

### 2.3 Sequential Composition (Houlsby)

Houlsby adapter function:

    adapter(x) = x + W_up @ sigma(W_down @ x)

where sigma is a nonlinearity (ReLU, GELU).

Composing two adapters:

    composed(x) = adapter_2(adapter_1(x))
                = adapter_1(x) + W_up_2 @ sigma(W_down_2 @ adapter_1(x))

This is NOT equivalent to sum of adapters:

    adapter_1(x) + adapter_2(x) != adapter_2(adapter_1(x))

because of the nonlinearity. The cross-term

    W_up_2 @ sigma(W_down_2 @ W_up_1 @ sigma(W_down_1 @ x))

introduces complex interactions that prevent additive decomposition.

### 2.4 Concatenative Composition (Prefix/Prompt Tuning)

Prefix tuning prepends virtual tokens P to keys and values:

    Attn(Q, [P_K; K], [P_V; V])

Composing two prefix adapters:

    Attn(Q, [P_K_1; P_K_2; K], [P_V_1; P_V_2; V])

This consumes context window linearly: N adapters with n_p prefix tokens
each consume N * n_p positions. For context window C:

    N_max = (C - required_context) / n_p

At n_p = 10 and C = 8192: N_max = ~800. But in practice, long prefixes
degrade attention quality well before this limit.

## 3. Parameter Counting

### 3.1 Per-Expert Parameters (Qwen2.5-7B, rank-16, FFN-only)

| Adapter Type | Formula | Per Layer | Total (28L) | % of Base |
|-------------|---------|-----------|-------------|-----------|
| LoRA-XS | 3 * r^2 | 768 | 21,504 | 0.0003% |
| VeRA | 2 * d | 7,168 | 200,704 | 0.003% |
| IA3 | 3 * d | 10,752 | 301,056 | 0.004% |
| BitFit | ~5 * d | 17,920 | 501,760 | 0.007% |
| Prefix (n=10) | 2 * n_p * d | 71,680 | 2,007,040 | 0.029% |
| Houlsby (k=64) | 4 * d * k | 917,504 | 25,690,112 | 0.367% |
| LoRA (r=16) | 3 * r * (d + d_ff) | 1,081,344 | 30,277,632 | 0.433% |
| Full-rank | 3 * d * d_ff | 203,685,888 | 5,703,204,864 | 81.5% |

### 3.2 Storage per Expert

At fp16 (2 bytes/param):

| Type | Storage per Expert |
|------|-------------------|
| LoRA-XS | 43 KB |
| VeRA | 401 KB |
| IA3 | 602 KB |
| BitFit | 1 MB |
| Prefix | 4 MB |
| LoRA (r=16) | 60 MB |
| Houlsby | 51 MB |

For 5,000 experts:
- LoRA: 300 GB total, 60 MB per query (k=1)
- LoRA-XS: 215 MB total, 43 KB per query
- VeRA: 2 GB total, 401 KB per query

## 4. Capacity Analysis

### 4.1 Rank of Weight Updates

The rank of a LoRA update dW = BA is at most r. For a d x d weight
matrix, the fraction of expressible directions is:

    rank_fraction = r / min(d_in, d_out)

At d=3584, r=16: rank_fraction = 0.0045.

This means a single LoRA adapter can express 0.45% of the full weight
space. For task-specific adaptation this is sufficient (empirically,
rank-8 to rank-16 captures most task knowledge). For base-level
knowledge it is insufficient.

### 4.2 ReLoRA: Accumulating Rank

ReLoRA merges LoRA updates periodically:

    W_t = W_{t-1} + B_t @ A_t    (merge, then reinitialize A, B)

After T merge iterations with rank-r updates:

    rank(W_T - W_0) <= T * r

Full rank is achieved when T * r >= min(d_in, d_out):

    T_full = ceil(min(d_in, d_out) / r)

For d=3584, r=16: T_full = 224 iterations.

Each iteration involves training a rank-16 LoRA to convergence, then
merging. The total computational cost is T_full times the cost of
one LoRA training run. This is still cheaper than full fine-tuning
(LoRA training is ~3x more memory-efficient).

### 4.3 Information-Theoretic Capacity

For a weight matrix W in R^{m x n} at fp16:

    Full capacity: m * n * 16 bits

LoRA at rank r:

    LoRA capacity: r * (m + n) * 16 bits
    Compression ratio: r * (m + n) / (m * n)
                     = r * (1/n + 1/m)
                     ~ 2r/d for square matrices

At d=3584, r=16: compression ratio = 0.009 (111x compression).

The key insight: LoRA's capacity is sufficient for task-specific
knowledge BECAUSE the base model provides the common knowledge.
Without the base, you need full rank.

## 5. Composition Interference Bounds

### 5.1 Additive Types (LoRA, BitFit, VeRA, LoRA-XS)

For N adapters composed additively, the interference is:

    I = sum_{i<j} |<dW_i, dW_j>| / (||dW_i|| * ||dW_j||)

For random rank-r matrices in R^{m x n}, the expected absolute cosine
between their vectorized (Frobenius) inner products scales as:

    E[|cos(dW_i, dW_j)|] ~ r / sqrt(D)

where D = m * n is the total number of entries. For D = d * d_ff =
3584 * 18944 = 67,879,424 and r = 16:

    E[|cos|] ~ 16 / sqrt(67,879,424) ~ 1.9 * 10^{-3}

However, the empirical measurement from macro/ortho_scaling/ gives
cos ~ 0.0002, which is ~10x LOWER than this random expectation.
This is because trained LoRA deltas are not random -- they specialize
to different tasks and thus occupy more orthogonal subspaces than
random matrices would. The empirical bound is more reliable than
the theoretical one for assessing real interference.

The N_max formula from VISION.md:

    N_max = D / r^2 = d^2 / r^2 = 3584^2 / 16^2 = 50,176

### 5.2 Non-Additive Types (IA3, Houlsby, Prefix)

These types do not have clean interference bounds because composition
is nonlinear. The only way to assess interference is empirically.

For IA3, in log-space:

    log(l_composed) = sum log(l_i)

If we define "interference" in log-space:

    I_log = sum_{i<j} |<log(l_i), log(l_j)>| / (||log(l_i)|| * ||log(l_j)||)

This is well-defined but the optimization dynamics are different
(training in linear space, composing in log space introduces mismatch).

## 6. Composability Classification

Based on the mathematical analysis, adapter types fall into three classes:

### Class A: Directly Composable (additive, mergeable)
- LoRA, QLoRA, rsLoRA: W + sum(dW_i)
- BitFit: b + sum(db_i)
- LoRA-XS: W + sum(U_r @ R_i @ V_r^T)
- VeRA: W + sum(diag(d_i) @ B_random @ diag(b_i) @ A_random)
- Tied-LoRA: W + sum(alpha_i * B_shared @ A_shared)

Property: all merge into base at inference (zero overhead).
Property: interference bounded by O(N^2 * epsilon) where epsilon ~ 10^{-4}.

**Note on full-rank adapters**: Full-rank deltas (W_new - W_base) are
technically "additive" and thus listed as Class A, but ONLY for the
single-adapter case. Multi-adapter composition of full-rank deltas
will NOT be near-orthogonal: two full-rank matrices in R^{d x d} have
E[|cos|] ~ 1/sqrt(d), which is far from zero. The O(N^2 * epsilon)
bound requires epsilon << 1, which holds for low-rank adapters
(where epsilon ~ r/sqrt(D)) but NOT for full-rank ones. In practice,
full-rank "composition" at N > 2 will exhibit significant interference.

### Class B: Composable with Caveats
- DoRA: W = m * (W0 + BA)/||W0 + BA||. Magnitude scaling is nonlinear.
  Can be approximated as additive for small deltas (first-order Taylor).
- MoLoRA: Additive but requires router at inference (not mergeable).

### Class C: Incompatible with Additive Composition
- IA3: Multiplicative. h' = diag(l) @ h.
- Houlsby: Sequential with nonlinearity. h' = h + W_up(sigma(W_down(h))).
- Prefix/Prompt: Concatenative. Consumes context window.

## 7. Base-Freedom Analysis

### 7.1 Can LoRA Encode Base Knowledge?

A single rank-r LoRA: NO. rank(BA) = r << d.

Iterative LoRA (ReLoRA): YES. rank(sum_{t=1}^{T} B_t @ A_t) = T*r.

The base model weights W_pretrained can be expressed as:

    W_pretrained = W_random + sum_{t=1}^{T} B_t @ A_t

where T = ceil(d/r) iterations. This is exact (not approximate).

### 7.2 Practical Implications

If the base IS a sequence of merged LoRA updates, then:

1. The "base adapter" = sum of the first T LoRA iterations
2. Domain adapters = standard LoRA on top of this accumulated base
3. Base upgrade = train a new sequence of LoRA iterations
4. The entire model is composable: base adapter + domain adapters

This is theoretically sound but practically untested at scale for
composition. The key unknown: do domain adapters trained on base_v1
compose with base_v2?

## 8. Assumptions and Limitations

1. **Random orthogonality assumption**: E[|cos|] bounds assume random
   deltas. Real trained deltas may have structure that increases or
   decreases cosine. Empirically, real cos ~ 10^{-4} (BETTER than
   random expectation), but the math-medical outlier (cos=0.59)
   shows domain overlap can violate this.

2. **Capacity analysis is information-theoretic, not empirical**:
   the "bits per parameter" estimates are upper bounds. Actual
   task-relevant capacity depends on the data distribution.

3. **ReLoRA/LTE full-rank claims are for pretraining, not composition**:
   these methods achieve full rank during training but the resulting
   model is a single dense model, not a composed set of adapters.
   Using the ACCUMULATED weights as a "base adapter" is our novel
   proposal, not yet validated.

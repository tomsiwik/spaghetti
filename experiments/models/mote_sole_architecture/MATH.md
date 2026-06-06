# MoTE-SOLE Architecture: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| W | (d, d') | Base model weight matrix (FP16, frozen) |
| Delta_i | (d, d') | Expert i weight delta = B_i @ A_i |
| A_i | (d, r) | LoRA down-projection for expert i |
| B_i | (r, d') | LoRA up-projection for expert i |
| h | (B, T, d) | Hidden state input to router |
| W_r | (d, N) | Router weight matrix |
| g(h) | (B, T, N) | Router logits = h @ W_r |
| p(h) | (B, T, N) | Router probabilities = softmax(g(h)) |
| N | scalar | Number of domain experts |
| k | scalar | Top-k experts selected per token |
| d | 64 | Model hidden dimension |
| r | 4 | LoRA rank |
| L | 2 | Number of transformer layers |

## Architecture

### Equal-Weight Baseline (prior experiments)

The composed model adds all expert deltas with uniform weight 1/N:

```
W_composed = W + (1/N) * sum_{i=1}^{N} Delta_i
```

This is static -- the same weights apply regardless of input.

### MoTE-SOLE Architecture

**Shared expert**: The frozen FP16 base model acts as the "shared expert" in
MoTE terminology. It processes every token and provides the general-purpose
representation.

**Routed experts**: N ternary domain experts, each a LoRA adapter with
weights quantized to {-1, 0, 1} * alpha via QAT with STE.

**Router**: A linear layer mapping hidden states to expert scores:

```
g(h_t) = h_t @ W_r + b_r         # (d,) @ (d, N) -> (N,)
p(h_t) = softmax(g(h_t))          # (N,)
```

**Top-k selection**: Select the k experts with highest probability. For SOLE
with N=5, we test k=1, k=2, k=3.

```
S_k = top_k_indices(p(h_t))
```

**Composed output**: Only selected experts contribute, weighted by their
softmax probability (renormalized over selected set):

```
w_i = p_i / sum_{j in S_k} p_j    for i in S_k
Delta_routed = sum_{i in S_k} w_i * Delta_i
W_effective = W + Delta_routed
```

### Load-Balancing Loss

Following Switch Transformer / MoTE, we add a load-balancing auxiliary loss
to prevent expert collapse (all tokens routed to one expert):

```
L_balance = N * sum_{i=1}^{N} f_i * P_i
```

where:
- f_i = fraction of tokens routed to expert i (hard assignment from top-k)
- P_i = mean probability assigned to expert i across all tokens

If routing is perfectly balanced, f_i = k/N for all i, and L_balance = k.
If all tokens go to one expert, L_balance = N (maximum).

The total training loss is:

```
L_total = L_NTP + alpha_balance * L_balance
```

With alpha_balance = 0.01 (standard MoE coefficient).

## Router Training

The router is trained AFTER expert training, on a mixed dataset containing
all domains. The training signal comes from:

1. **Cross-entropy loss** on next-token prediction (using the routed composition)
2. **Load-balancing loss** to ensure all experts get traffic

The router learns to detect which expert(s) are most beneficial for each
input token by observing which composition produces the lowest NTP loss.

### Complexity Analysis

**Router inference**: O(d * N) per token for the linear projection. At d=64,
N=5, this is 320 FLOPs -- negligible compared to the O(d^2) attention cost.

**Expert selection**: O(N log N) for sorting (or O(N) for partial sort with k << N).

**Weight composition**: O(k * d * d') for adding k expert deltas. This is
k/N of the equal-weight composition cost.

**Router training**: One epoch over the mixed dataset, gradient through
the top-k selection (using straight-through estimator for the hard selection).

## Worked Example (d=64, N=5, k=2, r=4)

Given hidden state h_t in R^64, router weights W_r in R^(64,5):

1. Compute logits: g = h_t @ W_r -> R^5, e.g., [1.2, -0.3, 0.8, -1.1, 0.5]
2. Softmax: p = [0.38, 0.08, 0.25, 0.04, 0.19]  (shifted, approximate)
3. Top-2: S_2 = {0, 2} (arithmetic and repeat experts)
4. Renormalize: w_0 = 0.38/(0.38+0.25) = 0.60, w_2 = 0.25/(0.38+0.25) = 0.40
5. Compose: Delta = 0.60 * Delta_0 + 0.40 * Delta_2
6. Effective: W_eff = W + Delta

Compared to equal-weight: Delta_eq = 0.20 * (Delta_0 + Delta_1 + ... + Delta_4)

The routed version concentrates 100% of the expert budget on the 2 most
relevant experts, vs spreading 20% each across all 5 (including irrelevant ones).

## Key Predictions

1. **Routing should help most when domains are dissimilar**: If experts
   interfere destructively (negative cosine similarity), routing avoids
   the interference. Prior finding: within-cluster |cos| = 7.84x higher
   than cross-cluster.

2. **k=1 may be optimal at N=5**: With only 5 experts, selecting the right
   one (plus the shared FP16 base) may suffice. k=2 adds redundancy.

3. **Ternary experts should match FP16 individually within 10%**: Prior
   exp_bitnet_ternary_adapter_composition showed only 2.6% individual
   quality loss for ternary adapters.

4. **Load balancing may be unnecessary at N=5**: With only 5 experts and
   distinct domains, natural routing should be balanced.

## Assumptions

1. The router can learn domain-relevant features from hidden states at
   layer L (after the full transformer forward pass).
2. QAT with STE produces ternary experts that retain domain-specific
   knowledge comparable to FP16 experts.
3. The straight-through estimator for top-k selection provides sufficient
   gradient signal for router training.
4. Load-balancing loss coefficient 0.01 is appropriate at this scale.

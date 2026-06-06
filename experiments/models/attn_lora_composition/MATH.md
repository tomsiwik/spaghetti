# Attention LoRA Composition: Mathematical Foundations

## Setup

### Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| d | Model embedding dimension | scalar (64) |
| r_m | MLP LoRA rank | scalar (8) |
| r_a | Attention LoRA rank | scalar (4) |
| alpha | LoRA scaling factor | scalar (1.0) |
| L | Number of transformer layers | scalar (4) |
| h | Number of attention heads | scalar (4) |
| d_h | Head dimension = d/h | scalar (16) |
| N | Number of domains | scalar (2) |

### LoRA Delta Formulation

For any weight matrix W with LoRA adapter (A, B):

    W_adapted = W + (alpha / r) * A @ B

where A: (d_in, r), B: (r, d_out). The delta dW = (alpha/r) * A @ B is pure linear.

### Adapted Projections

**MLP-only LoRA** (baseline): adapts fc1 and fc2 per layer.

    MLP(x) = (W_fc2 + dW_fc2) @ relu((W_fc1 + dW_fc1) @ x)

Parameters per domain per layer: 2 * (d * r_m + r_m * 4d) = 2 * r_m * 5d
Total MLP LoRA params per domain: L * 2 * r_m * 5d = 4 * 8 * 5 * 64 = 10,240 * 2 = 20,480

**MLP + Attention LoRA** (experimental): adapts fc1, fc2, Wq, Wk per layer.

    q = (W_q + dW_q) @ x    # adapted query
    k = (W_k + dW_k) @ x    # adapted key
    v = W_v @ x              # shared value (not adapted)
    Attn(x) = W_o @ softmax(q @ k^T / sqrt(d_h)) @ v

Attention LoRA params per domain per layer: 2 * (d * r_a + r_a * d) = 2 * 2 * d * r_a
Total attention LoRA params: L * 4 * d * r_a = 4 * 4 * 64 * 4 = 4,096
Total (MLP + Attn) params: 20,480 + 4,096 = 24,576 (+20% overhead)

### Why Wq/Wk Only (Not Wv/Wo)

The control theory principle of minimal intervention at the bottleneck:
- Wq and Wk determine WHAT tokens attend to (attention routing)
- Wv and Wo determine HOW attended content is transformed (value projection)
- The bottleneck is attention routing (shared patterns force all domains
  to attend the same way), not value projection
- Adapting Wq/Wk modifies attention patterns with fewer parameters than
  adapting all four projections

### Composition via Routed Deltas

Given N domain-specific delta sets, composition routes between them:

    output = sum_{k=1}^{K} w_k(x) * f_k(x)

where w_k are routing weights from a learned softmax router, K = top-k experts,
and f_k computes the forward pass using expert k's delta-augmented weights.

For MLP+Attention composition, the routing applies to BOTH:
- Attention: q_composed = sum_k w_k * (W_q + dW_q^k) @ x
- MLP: mlp_composed = sum_k w_k * MLP_k(x)

The key insight: attention composition is LINEAR in the routing weights because
the adapted projection is a sum of linear maps. The MLP composition is NOT linear
due to the ReLU nonlinearity, but routing averages the outputs (not weights).

### Composition Gap Analysis

The composition gap measures degradation from composition vs joint training:

    gap = (L_composed - L_joint) / L_joint * 100%

Hypothesis: adapting attention closes the gap because:
1. Shared attention forces all domains to use identical attention patterns
2. Domain-specific attention patterns allow each domain to "look at" different features
3. This is a first-order correction to the proven bottleneck

### Parameter Efficiency

Attention LoRA overhead relative to MLP-only: 4,096 / 20,480 = 20%
But attention delta captures 35.9% of total adapted norm, suggesting
the attention subspace carries disproportionate information per parameter.

## Assumptions

1. **Rank 4 is sufficient for attention adaptation at d=64**: at full rank d=64,
   attention LoRA would be 4x larger. Rank 4 captures only 4/64 = 6.25% of the
   full-rank space. This may be insufficient at micro scale where the total
   information capacity is already constrained.

2. **Wq/Wk capture the bottleneck**: the bottleneck could equally be in Wv/Wo
   or in the interaction between attention and MLP. Only adapting Wq/Wk tests
   one specific hypothesis about where the bottleneck lies.

3. **Attention patterns are domain-discriminative at micro scale**: at d=64 with
   character-level tokenization, attention patterns may not differ meaningfully
   between a-m and n-z names. The prior finding that domain discrimination is
   weak at micro scale applies here too.

## Worked Example

d=64, r_a=4, single layer, single head for simplicity:

    W_q: (64, 64), dW_q = (1.0/4) * A_q @ B_q where A_q: (64, 4), B_q: (4, 64)
    W_k: (64, 64), dW_k = (1.0/4) * A_k @ B_k where A_k: (64, 4), B_k: (4, 64)

    For input x: (1, T, 64)
    q_adapted = x @ (W_q + dW_q)^T  # shape (1, T, 64)
    k_adapted = x @ (W_k + dW_k)^T  # shape (1, T, 64)

    attn = softmax(q_adapted @ k_adapted^T / sqrt(16))  # (1, h, T, T)

The adapted attention matrix differs from base by:
    delta_attn ~ softmax'(base_attn) * (dW_q @ x) @ (dW_k @ x)^T / sqrt(d_h)

This is a rank-r_a perturbation to the attention pattern, capable of
shifting attention to domain-relevant positions while preserving the
overall attention structure from pretraining.

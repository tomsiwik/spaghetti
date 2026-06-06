# P7.B1: Weighted Multi-Adapter Composition via Null-Space Projections

## Motivation

Finding #495 killed null-space projection as a routing signal: domain info lives in
range(W_v), not null(W_v). But the adapters trained in that experiment are valid —
they achieve near-zero loss per domain with exact orthogonality.

The open question: given a proven router (TF-IDF, Finding #354: 95% accuracy), does
**weighted** composition of null-space adapters outperform **exclusive** (argmax) routing
on mixed-domain queries?

**Prior art:**
- LoRAHub (Huang et al., 2310.13699): gradient-free composition of LoRA adapters via
  weighted sum, showing quality improves when relevant adapters are combined
- Finding #494: null-space LoRA preserves 98.7% quality with orthogonality guarantee
- Finding #354: TF-IDF routing achieves 95% accuracy at N=5
- Finding #495: route in range(W_v), adapt in null(W_v) — complementary subspace concerns

## Setup

- Model: Gemma 4 e4b-it-4bit
- 5 null-space adapters on v_proj layers 16-23 (reused from exp_p7_null_projection_routing)
- Domains: medical, code, math, legal, finance
- Router: TF-IDF cosine similarity → softmax weights

## Theorem (Null-Space Closure Under Convex Combination)

**Statement.** Let W_v in R^{d_out x d_in} be the value projection weight matrix with
null-space basis Q in R^{d_in x d_null}. Let {(A_i, B_i)}_{i=1}^k be k null-space LoRA
adapters where A_i in R^{d_null x r} and B_i in R^{r x d_out}.

For any weight vector w in Delta^{k-1} (probability simplex: w_i >= 0, sum w_i = 1),
the weighted composition delta matrix:

    D = sum_{i=1}^k w_i * (Q @ A_i @ B_i)

satisfies W_v @ D = 0.

**Proof.** By linearity of matrix multiplication:

    W_v @ D = W_v @ sum_{i=1}^k w_i * (Q @ A_i @ B_i)
            = sum_{i=1}^k w_i * W_v @ Q @ A_i @ B_i
            = sum_{i=1}^k w_i * 0 @ A_i @ B_i      [since W_v @ Q = 0 by construction]
            = 0

The null-space is a vector subspace, closed under linear combination. Any convex
combination of null-space adapter effects remains in the null-space. QED.

**Corollary 1 (Harmless composition).** The weighted composition preserves the base
model's value projection regardless of the weight vector. Even a poor router cannot
degrade the base model — the worst case is suboptimal adapter contribution, not damage.

**Corollary 2 (Exclusive routing as special case).** When w = e_j (one-hot vector),
the weighted composition reduces to exclusive routing with adapter j. Exclusive routing
is a degenerate case of weighted composition.

## Predictions

### P1: Single-domain queries
On single-domain test inputs, TF-IDF similarity is peaked (one domain dominates).
The softmax weights approximate a one-hot vector.

**Prediction:** Weighted composition NTP loss within 2pp of exclusive routing on
single-domain queries (they converge to the same adapter).

### P2: Mixed-domain queries  
On inputs spanning two domains (e.g., "medical device liability" = medical + legal),
the TF-IDF weight vector has entropy > 0, activating multiple adapters.

**Prediction:** Weighted composition achieves >= 3pp lower NTP loss than exclusive
routing on mixed-domain queries (the correct second adapter contributes useful signal).

### P3: Orthogonality under composition
For any weight vector, the merged adapter delta satisfies W_v @ D = 0.

**Prediction:** max|W_v @ D| < 1e-4 for all tested weight vectors.

## Kill Criteria

- K1303: Weighted composition outperforms exclusive routing by >= 3pp on mixed-domain
- K1304: Weighted composition does not degrade single-domain (< 2pp vs exclusive)  
- K1305: Cross-domain queries show measurable benefit from multi-adapter

## What Would Kill This

1. **Adapter interference in output space:** Though adapters are orthogonal to W_v,
   their B matrices map to the same output dimensions. If B_i and B_j constructively
   interfere (amplify noise), weighted composition could be worse than exclusive.
   
2. **TF-IDF weight entropy too low:** If TF-IDF similarity is already peaked on
   mixed-domain queries (one domain always dominates), weighted ≈ exclusive and
   K1303 fails trivially (no difference, not regression).

3. **Mixed-domain texts don't benefit from multiple adapters:** The adapters were
   trained on single-domain data. Mixed-domain texts may not activate useful features
   in non-primary adapters.

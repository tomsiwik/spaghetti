# Partial RoPE: Position-Free Dimensions as Natural Routing Features

## 1. Mechanism Definition

### RoPE (Rotary Position Embedding)

For a query/key vector q in R^{d_head}, RoPE applies a position-dependent rotation:

```
RoPE(q, pos) = R(pos) * q
```

where R(pos) is a block-diagonal rotation matrix. For head dimension d_head=80 with
pairs indexed by j in {0, ..., d_head/2 - 1}:

```
R(pos)_{2j, 2j}     = cos(pos * theta_j)
R(pos)_{2j, 2j+1}   = -sin(pos * theta_j)
R(pos)_{2j+1, 2j}   = sin(pos * theta_j)
R(pos)_{2j+1, 2j+1} = cos(pos * theta_j)
```

where theta_j = rope_theta^{-2j/d_head}, with rope_theta=10000 for BitNet-2B-4T.

**Key property:** RoPE is applied AFTER the linear projections (q_proj, k_proj) but
BEFORE the dot-product attention. The pre-RoPE Q/K outputs are purely semantic
(content-dependent), while post-RoPE Q/K encode both content AND position.

### BitNet-2B-4T Architecture (Actual)

- Hidden dim d=2560
- 20 attention heads, d_head=128
- 5 KV heads (GQA: 4 query heads share each KV head)
- Full RoPE on all 128 dimensions per head
- 24 transformer layers

### Partial RoPE Simulation

Since BitNet applies RoPE to all dims, we SIMULATE partial RoPE by splitting:

```
q_pre_rope in R^{B x L x H x d_head}
q_rope = q_pre_rope[:, :, :, :r_rope]     # first 25% dims: position-encoded
q_free = q_pre_rope[:, :, :, r_rope:]     # last 75% dims: position-free
```

where r_rope = d_head/4 = 32. The "position-free" dimensions are the pre-RoPE
projections, which depend only on the content of the input token, not its position.

### Routing from Position-Free Features

Given N_samples per domain, compute domain centroids from mean-pooled position-free
QK features:

```
c_d = (1/N_d) * sum_{i in domain_d} mean_pool(q_free_i)   in R^{D_free}
```

where D_free = n_heads * (d_head - r_rope) = 20 * 96 = 1920.

Route input x by: argmin_d ||mean_pool(q_free(x)) - c_d||_2

## 2. Why It Might Work

The hypothesis rests on two observations:

**Observation 1 (Parameter Golf, arXiv 2506.06105):** Partial RoPE models (25% dims
with position encoding) learn to use position-free dimensions for pure semantic
similarity. The position-free dimensions develop into content-type features.

**Observation 2 (Pre-RoPE features are position-invariant):** The q_proj output
before RoPE depends only on the linear projection W_Q applied to the hidden state.
For a given semantic concept, q_pre = W_Q * h, where h is the hidden state at that
position. The same content at different positions produces the same pre-RoPE Q.

**Connection:** If domains have distinct semantic signatures, the pre-RoPE Q
projections should cluster by domain. The Q projection is a learned linear map
that extracts "what the model wants to attend to" -- this is semantic content,
not positional structure.

## 3. What Breaks It

**Failure mode 1: Q projections are position-dominated even pre-RoPE.**
The hidden state h itself carries position information from prior layers' RoPE
applications. By the last layer, h may encode position so strongly that even
pre-RoPE Q projections are position-dependent. If position variance >> domain
variance in Q space, clustering fails.

**Failure mode 2: Domain signal is in MLP residual, not attention.**
The attention Q/K projections may not be the right place to look for domain
signal. Domain-specific features may live primarily in the MLP residual stream.
The prior softmax router experiment used FULL hidden states (post-norm, all
2560 dims) and achieved 40% accuracy with oracle-matching PPL.

**Failure mode 3: 75% of Q dims is not enough signal.**
Even if some Q dims encode semantics, 60/80 dims per head times 32 heads = 1920
features may not have enough domain-discriminative variance compared to the full
2560-dim hidden state that the existing router uses.

**Kill criteria connection:**
- K1 (silhouette < 0.3): Fails if position variance dominates domain variance
- K2 (routing < 1/N = 4.2%): Fails if Q features carry no domain signal at all
- K3 (PPL degradation): N/A -- we're not training a partial RoPE model

## 4. Assumptions

1. **Pre-RoPE Q/K projections are accessible.** We extract by hooking into the
   attention forward pass before RoPE application. Confirmed: BitNet attention
   code separates q_proj() from self.rope() calls.

2. **Mean-pooling is adequate aggregation.** Sequence-level routing requires
   reducing L tokens to one vector. Mean-pool is the simplest; max-pool or
   CLS-token extraction are alternatives. Prior work (softmax router) used
   mean-pooled full hidden states.

3. **Centroid-based routing is the correct baseline.** K-nearest centroid is the
   simplest zero-parameter router. If this fails, learned routing might still
   work on these features, but the "zero-parameter" motivation is lost.

## 5. Complexity Analysis

- Feature extraction: O(N_samples * L * d * d_head) for forward pass
- Centroid computation: O(N_domains * N_samples_per_domain * D_free)
- Routing decision: O(N_domains * D_free) per query -- negligible
- Memory: 24 centroids * 1920 floats = 184 KB -- essentially zero

vs. softmax router: 330K learned parameters, 0.46% of inference time

## 6. Worked Example (d_head=128, 20 heads, 24 domains)

- r_rope = 32 (25% of 128)
- D_free per head = 96
- Total D_free = 20 * 96 = 1920
- Extract pre-RoPE Q from last layer for 50 samples per domain
- Compute 24 centroids in R^1920
- For test sample: compute pre-RoPE Q, mean-pool, find nearest centroid
- Compare: full hidden state centroids (R^2560) vs Q-free centroids (R^1920)

## 7. Connection to Architecture

The existing softmax router uses full hidden states (R^2560) and achieves
oracle-matching quality at N=24 despite only 40% classification accuracy.
This experiment asks: can we achieve similar routing using ONLY the attention
features that are architecturally position-invariant?

If YES: position-free attention features are natural routing signals, supporting
the Partial RoPE design where 75% of dims are explicitly position-free.

If NO: domain routing signal lives in dimensions that RoPE would encode with
position, meaning Partial RoPE would lose routing-relevant information.

## References

- RoFormer (arXiv 2104.09864): original RoPE paper
- Parameter Golf / T2L scaling (arXiv 2506.06105): partial RoPE analysis
- exp_softmax_router_scaling: 40% accuracy, oracle quality, 330K params
- exp_speculative_expert_selection: router overhead 0.46%, optimization dead end

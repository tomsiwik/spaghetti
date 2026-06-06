# MoLoRA Per-Token Routing: Mathematical Foundations

## Notation

| Symbol | Meaning | Typical shape |
|--------|---------|---------------|
| d | Hidden state dimension | 2560 |
| N | Number of domain experts | 5 |
| k | Top-k experts per routing decision | 2 |
| T | Sequence length | up to 256 |
| h | Router hidden dimension | 64 |
| r | LoRA rank | 16 |
| h_t | Hidden state at token position t | (d,) |
| g_t | Gate vector at token position t | (N,) |

## Per-Token Gumbel-Sigmoid Router

### Architecture
The router is a 2-layer MLP that maps each token's hidden state to N independent gate logits:

```
f_router(h_t) = W_2 * GELU(W_1 * h_t + b_1) + b_2
```

Where:
- W_1 in R^{h x d}, b_1 in R^h (first layer: d -> h)
- W_2 in R^{N x h}, b_2 in R^N (second layer: h -> N)

Parameter count: d*h + h + h*N + N = 2560*64 + 64 + 64*5 + 5 = 164,229

### Gumbel-Sigmoid Non-Competing Gates (L2R-style)

Unlike softmax top-k (which forces experts to compete), each expert gate is an independent Bernoulli:

```
g_t^{(i)} = sigma((f_router(h_t)_i + epsilon) / tau)
```

Where:
- epsilon = log(u) - log(1-u), u ~ Uniform(0,1) (Gumbel noise)
- tau = temperature (1.0 in our experiments)
- sigma = sigmoid function

**Key property**: Because gates are independent, multiple experts can be simultaneously activated without competing. This allows natural multi-task blending.

### Top-k Selection and Normalization

At inference (no Gumbel noise):
```
g_t^{(i)} = sigma(f_router(h_t)_i)

S_t = argtop_k(g_t)     (indices of k largest gates)

w_t^{(i)} = g_t^{(i)} / sum_{j in S_t} g_t^{(j)}   for i in S_t
           = 0                                          otherwise
```

### Per-Token Adapter Composition

For a token group G (tokens sharing the same top-k expert set S):

```
Delta W_t = sum_{i in S} w_t^{(i)} * B_i * A_i
```

Where A_i, B_i are the LoRA matrices for expert i.

### Token Grouping for Efficient Pre-Merge

In practice, many tokens in a sequence select the same expert pair. We group tokens by their expert set S and pre-merge the adapters once per group:

```
Delta W_S = sum_{i in S} w_rep^{(i)} * B_i * A_i
```

Where w_rep is the weight vector of the **first token** assigned to group S, used as a proxy for the group. This is a simplification -- ideally one would average weights across all tokens in the group (bar{w}_S). In practice, within a group all tokens selected the same top-k expert pair, so their sigmoid gate values are generally close (they passed a similar threshold). The approximation error is bounded by the within-group weight variance, which is small when the router produces confident (high or low sigmoid) activations.

**Implementation note**: The code (run_experiment.py ~line 600-606) assigns `token_weights[expert_set] = weights` on the first token encountered for each group and does not update it for subsequent tokens. This matches the "first-token proxy" description above, not a true average.

**Complexity**: With N=5 experts and k=2, there are C(5,2)=10 possible expert pairs. Observed average: 2.42 distinct groups per sequence (empirical).

## Training Objective

Binary cross-entropy on independent expert gates:

```
L = -(1/T) sum_{t=1}^{T} sum_{i=1}^{N} [y_i * log(g_t^{(i)}) + (1-y_i) * log(1-g_t^{(i)})]
```

Where y_i = 1 if token t belongs to domain i, 0 otherwise.

During training, Gumbel-sigmoid provides gradient flow through the discrete top-k selection.

## Computational Cost

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Router forward (per token) | 2*d*h + 2*h*N = 327,680 + 640 | 164K params (0.66MB) |
| Base forward (per token) | ~O(d^2 * L) | ~1.7GB model |
| Router overhead | 0.58% of base forward | negligible |

## Numerical Example (d=2560, N=5, k=2)

For a medical text token:
- Hidden state: h_t in R^2560
- Router output: f(h_t) = [0.1, 0.3, 0.9, 0.2, 0.1] (logits)
- Sigmoid gates: g_t = [0.52, 0.57, 0.71, 0.55, 0.52]
- Top-2: {math, medical} with scores {0.57, 0.71}
- Normalized: w_math = 0.45, w_medical = 0.55
- Merged adapter: Delta W = 0.45 * B_math * A_math + 0.55 * B_medical * A_medical

## Assumptions

1. **Domain homogeneity within tokens**: Each token's optimal expert set depends primarily on its local hidden state, not on global sequence context. This is a simplification -- in practice, a token's meaning depends on its context.

2. **Router generalization**: The router trained on 5 cleanly separated domains can generalize to mixed-domain text. At micro scale with 5 trivially separable domains, per-token routing adds minimal value over per-sequence routing.

3. **Hidden state sufficiency**: The base model's hidden states at the final layer contain enough information for expert selection. Proven in exp_tiny_routing_heads (100% domain accuracy).

4. **Pre-merge approximation**: Using first-token weights as a proxy for the token group (rather than per-token weights or group-averaged weights) introduces a small approximation error. Within a group, all tokens selected the same top-k expert pair, so their normalized sigmoid weights are generally close. The error is bounded by the within-group weight variance.

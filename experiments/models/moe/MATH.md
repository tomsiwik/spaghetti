# MoE GPT: Mathematical Foundations

## 1. Router

For each token embedding `x` of dimension `d`, the router computes logit scores over `N` experts via a single linear projection (no bias):

```
s = x W_r^T,   W_r in R^{N x d},   s in R^N
```

These are converted to gate probabilities with softmax:

```
p_i = exp(s_i) / sum_j exp(s_j),   i = 1..N
```

`p` forms a probability distribution over experts. It is also saved (as `_gate_probs`) for the balance loss computation.

## 2. Top-k Gating

Only the top-k experts are activated per token. The implementation uses a threshold derived from the raw scores (not the probabilities):

```
threshold = min over top-k values of s
mask_i = 1  if s_i >= threshold,  else 0
```

The gating weights are then the masked softmax probabilities, renormalized to sum to 1:

```
g_i = (p_i * mask_i) / (sum_j p_j * mask_j + epsilon)
```

This means the final weights are non-negative and sum to exactly 1 over the k active experts. Masking on scores (not probs) ensures the threshold is stable; renormalizing on probs preserves the relative ranking from softmax.

## 3. Expert Computation

Each expert is an independent two-layer MLP with a 4x bottleneck and ReLU activation:

```
Expert_i(x) = W_{i,2}  ReLU( W_{i,1} x )
```

where `W_{i,1} in R^{4d x d}` and `W_{i,2} in R^{d x 4d}`.

The MoE layer output is the gating-weighted sum over all experts (with non-selected experts contributing zero via `g_i = 0`):

```
MoE(x) = sum_{i=1}^{N} g_i * Expert_i(x)
```

In this implementation all `N` experts are always evaluated and then multiplied by their gate weights. This is equivalent to sparse dispatch for small `N` (see section 5).

## 4. Load Balancing Loss

Without regularization, routers collapse: a few experts receive all tokens and the rest are never trained. The balance loss penalizes this concentration.

Let `f_i` be the mean gate probability for expert `i` across all tokens in the batch:

```
f_i = mean_{b,t} p_{b,t,i}
```

The balance loss is:

```
L_bal = N * sum_{i=1}^{N} f_i^2
```

with a coefficient of `0.01` applied at the model level:

```
L_total = L_CE + 0.01 * sum_{layers} L_bal
```

### Why This Works

By Cauchy-Schwarz (or Jensen's inequality), for a fixed `sum f_i = 1`:

```
sum f_i^2 >= (sum f_i)^2 / N = 1/N
```

Equality holds if and only if `f_i = 1/N` for all `i` (perfectly uniform load). The factor `N` normalizes the minimum to exactly 1, so the loss is at its minimum value of 1 when routing is perfectly balanced and grows as experts become more loaded.

Gradient intuition: `d/df_i (N * sum f_j^2) = 2N * f_i`. An overloaded expert (`f_i > 1/N`) receives a large positive gradient on `f_i`, which the router is trained to reduce. The loss acts directly on `p` (the pre-mask probabilities), so it penalizes the router's tendency to concentrate probability mass even for tokens that ultimately get routed elsewhere.

## 5. Parameter Count

### Dense GPT Baseline (per layer)

Each `Block` contains:
- Attention: `W_q, W_k, W_v, W_o` each `d x d` -> `4d^2` params
- MLP: `W_1` (`4d x d`) + `W_2` (`d x 4d`) -> `2 * 4d^2 = 8d^2` params

Total per layer: `12d^2`

### MoE Substitution

Replace the single MLP with `N` expert MLPs plus one router:

- `N` experts: `N * 8d^2` params
- Router: `W_r in R^{N x d}` -> `N*d` params

Total MoE per layer: `4d^2` (attention) + `N * 8d^2` (experts) + `N*d` (router)

### Full Model

```
P_GPT  = n_layer * 12d^2 + V*d + T*d + d*V
P_MoE  = n_layer * (4d^2 + N*8d^2 + N*d) + V*d + T*d + d*V
```

where `V = vocab_size`, `T = block_size`. The `norm0`, `wte`, `wpe`, `lm_head` terms are shared and identical.

Parameter overhead from MoE:

```
Delta = n_layer * ((N-1) * 8d^2 + N*d)
```

At default settings (`d=64, N=4, n_layer=4`):
- Dense MLP per layer: `8 * 64^2 = 32768`
- MoE experts per layer: `4 * 32768 = 131072` (+`4*64 = 256` router)
- Overhead: `3 * 32768 * 4 = 393216` extra params across all layers

## 6. Active Parameter Ratio

Only `top_k` of the `N` experts process each token. The fraction of expert parameters that are "active" per forward pass per token is:

```
active_ratio = top_k / N
```

At `top_k=2, N=4`: 50% of expert parameters are active per token.

Attention parameters are always fully active. Total active fraction across a full block:

```
active_frac = (4d^2 + top_k * 8d^2) / (4d^2 + N * 8d^2)
            = (4 + 8*top_k) / (4 + 8*N)
```

At `top_k=2, N=4`: `(4 + 16) / (4 + 32) = 20/36 ~= 55.6%`

This is the core MoE trade-off: parameter count scales with `N`, but compute per token scales with `top_k`.

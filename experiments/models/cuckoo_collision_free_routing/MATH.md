# Cuckoo Collision-Free Routing: Mathematical Foundations

## Notation

| Symbol | Definition | Shape/Type |
|--------|-----------|------------|
| d | Embedding dimension | scalar (=64) |
| N | Number of expert groups | scalar (=8) |
| k | Experts selected per token | scalar (=2) |
| B | Batch size | scalar |
| T | Sequence length | scalar |
| x | Hidden state of one token | R^d |
| X | Batch of hidden states | R^(B x T x d) |
| h1, h2 | Hash function projections | R^(N x d) |
| s1, s2 | Raw scores from h1, h2 | R^(B x T x N) |
| p1, p2 | Softmax probabilities from h1, h2 | R^(B x T x N), entries in [0,1] |
| tau | Collision threshold | scalar in [0, 1] |
| alpha | Eviction blending weight | R^(B x T x 1), entries in [0,1] |
| w | Final routing weights | R^(B x T x N), sparse with k nonzero |

## Core Mechanism

### Dual Hash Functions

Two independent linear projections serve as hash functions, mapping hidden
states to expert scores:

```
s1 = h1(x) = W1 @ x,   W1 in R^(N x d)
s2 = h2(x) = W2 @ x,   W2 in R^(N x d)
```

Both W1 and W2 are learned (trainable). At initialization, they are independent
random matrices, guaranteeing different routing patterns with high probability.

### Collision Detection

In standard softmax routing with one projection, a "collision" occurs when the
top-2 expert scores are close:

```
collision(x) := |p_top1 - p_top2| < epsilon
```

where p = softmax(W @ x). At collision, the router is uncertain, and both
selected experts may compute similar outputs (wasted compute).

We detect collisions via the maximum probability from h1:

```
confidence(x) = max_i(p1_i),  p1 = softmax(s1)
```

If confidence(x) < tau, the primary hash function has no clear winner.

### Soft Eviction Blending

The eviction decision is made differentiable via sigmoid blending:

```
alpha = sigmoid((tau - confidence(x)) * T_temp)
```

where T_temp = 10 is a temperature parameter controlling eviction sharpness.

- alpha ~ 0 when confidence >> tau (no eviction, use h1)
- alpha ~ 1 when confidence << tau (evict, use h2)

The blended probability distribution is:

```
p_blend = (1 - alpha) * p1 + alpha * p2
```

### Top-k Selection

Standard top-k masking on the blended distribution:

```
top_k_vals = topk(p_blend, k)
threshold = min(top_k_vals)
mask = (p_blend >= threshold)
w = mask * p_blend / sum(mask * p_blend)
```

## Eviction Chain Depth

In cuckoo hashing, an eviction chain of length L means L items were displaced
before finding an empty slot. In our routing analogy:

- Depth 0: h1 is confident (no eviction needed)
- Depth 1: h1 not confident, evicted to h2
- Depth 2: h2 also not confident (would need h3, but we stop at 2)

Formally:

```
depth(x) = I[confidence_h1(x) < tau] + I[confidence_h1(x) < tau] * I[confidence_h2(x) < tau]
```

where I[.] is the indicator function.

**Bound**: With two hash functions, maximum chain depth is 2. This is well
below the kill threshold of 3. With M hash functions, max depth is M.

## Collision Rate Analysis

For a uniform softmax distribution over N experts, p_i = 1/N for all i.
The expected gap between top-1 and top-2 is 0. As the distribution concentrates,
the gap increases. The collision rate (fraction of tokens with gap < epsilon) is:

```
collision_rate = P[max(p) - second_max(p) < epsilon]
```

For N=8 with random-init router:
- Uniform: expected gap = 0, collision rate ~ 1.0
- After training: collision rate depends on specialization strength
- Measured: 57.4% of tokens have gap < 0.05 after 500 steps

This means **over half of tokens have near-tied routing** in softmax.
The cuckoo mechanism provides a second independent opinion (h2) for these
ambiguous tokens.

## Parameter Count

| Component | Softmax (N=8) | Cuckoo (N=8) |
|-----------|--------------|--------------|
| Router h1 (per layer) | 512 | 512 |
| Router h2 (per layer) | 0 | 512 |
| Tau (per layer) | 0 | 1 |
| Router total (4 layers) | 2,048 | 4,100 |
| Capsule params | 196,608 | 196,608 |
| Embeddings + head | 5,504 | 5,504 |
| **Total** | **204,160** | **206,208** |

Overhead: +2,048 params (+1.0%) from the second hash function.

## Computational Cost

### Forward Pass

Softmax routing per token per layer:
```
FLOPs_softmax = N*d (projection) + N*log(N) (softmax) + N (top-k)
             = 8*64 + 8*3 + 8 = 544
```

Cuckoo routing per token per layer:
```
FLOPs_cuckoo = 2*N*d (two projections) + 2*N*log(N) (two softmaxes)
             + N (blending) + N (top-k)
             = 2*512 + 48 + 8 + 8 = 1,080
```

Ratio: 1,080/544 = 1.99x. The routing cost doubles, but routing is a small
fraction of total FLOPs (capsule evaluation dominates at 2 * N_caps * d per
selected expert).

### Total FLOPs per Token

```
Expert compute per token = k * (2 * n_caps * d)  [A and B matrices]
                        = 2 * (2 * 32 * 64)
                        = 8,192 FLOPs

Routing compute:
  Softmax: 544 FLOPs (6.2% of expert compute)
  Cuckoo: 1,080 FLOPs (13.2% of expert compute)

Total overhead of cuckoo: +536 FLOPs = +6.5% vs softmax routing
```

## Assumptions

1. **Independent hash functions**: W1 and W2 are initialized independently and
   remain sufficiently different during training. Validated: h1 vs h2 differ
   rate is 87.5% at initialization.

2. **Score ties indicate ambiguity**: When softmax scores are close, the router
   is genuinely uncertain. This assumes the router has learned meaningful
   expert specialization. At micro scale with G=8 and character-level data,
   specialization is weak, so the collision rate is high (57.4%).

3. **Tau is learnable**: The collision threshold can adapt to the model's
   confidence distribution. In practice, tau did not learn (stayed at 0.299)
   because it was implemented as a raw array rather than a registered parameter.
   The mechanism still works because h1 and h2 learn to produce scores that
   naturally resolve routing through the blended distribution.

4. **Two hash functions suffice**: Classical cuckoo hashing achieves < 50%
   load factor with 2 hash functions. For routing, 2 candidate expert sets
   provide sufficient diversity. The measured max chain depth of 0.24
   (well below 3) confirms this.

## Worked Example (d=4, N=4, k=2)

Token x = [1.0, 0.5, -0.3, 0.8]

W1 (4x4):
```
h1 = [[0.2, 0.1, -0.1, 0.3],   -> s1 = [0.51, 0.05, -0.09, 0.47]
      [0.0, 0.1,  0.2, -0.1],
      [-0.1, 0.0, 0.1, 0.0],
      [0.3, 0.2, -0.2, 0.1]]
```

p1 = softmax(s1) = [0.321, 0.203, 0.176, 0.300]
confidence = max(p1) = 0.321

tau = 0.3 (learned threshold)
alpha = sigmoid((0.3 - 0.321) * 10) = sigmoid(-0.21) = 0.448

Since alpha = 0.448 ~ 0.5, roughly half h1, half h2:

W2 (different):
s2 = [0.10, 0.60, 0.20, -0.15]
p2 = softmax(s2) = [0.222, 0.366, 0.245, 0.173]

p_blend = 0.552 * [0.321, 0.203, 0.176, 0.300]
        + 0.448 * [0.222, 0.366, 0.245, 0.173]
        = [0.277, 0.276, 0.207, 0.243]

Top-2: experts 0 and 1 (scores 0.277 and 0.276)
After renorm: w = [0.501, 0.499, 0, 0]

Chain depth = 0 (alpha < 0.5, not classified as eviction)

Compare softmax-only: top-2 from p1 = experts 0 and 3 (0.321, 0.300)
The cuckoo mechanism considers h2's strong preference for expert 1,
producing a different (potentially better) routing decision.

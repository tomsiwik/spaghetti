# N=50 Ternary Adapter Composition with Gumbel Routing: Mathematical Foundations

## Notation

| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| N | Number of adapters | 50 |
| d | Model hidden dimension | 2560 (BitNet-2B-4T) |
| r | LoRA rank | 16 |
| A_i | Low-rank projection (frozen, random uniform) | d x r |
| B_i | Trained ternary weights | r x d |
| Delta W_i | Adapter weight update = A_i B_i | d x d |
| alpha | STE ternary scale = mean(|W|) | scalar |
| s | LoRA scale factor | 20.0 |
| k | Top-k for routing | 2 |
| tau | Gumbel temperature | 1.0 -> 0.5 (annealed) |
| h | Hidden state (router input) | d |
| g_i | Gumbel-sigmoid gate for adapter i | [0,1] |

## A-Matrix Initialization (Frozen, Random Uniform)

Each A_i is initialized from Uniform(-s, s) where s = 1/sqrt(d), then frozen.
No Grassmannian Alternating Projection is applied. Orthogonality between
A matrices arises naturally from the high d/r ratio (2560/16 = 160:1).

By the Johnson-Lindenstrauss lemma and the FlyLoRA result (Liu et al., 2024),
random projections in R^d with d >> r produce near-orthogonal subspaces with
high probability when the number of subspaces N << d^2/r^2.

Capacity bound: N_max = d^2 / r^2 = 2560^2 / 16^2 = 25,600 >> 50.
At N=50, we use 50/25600 = 0.2% of the theoretical capacity.

### Interference Bound

||Delta W_i^T Delta W_j||_F = ||B_i^T A_i^T A_j B_j||_F
                             <= ||B_i|| * ||A_i^T A_j||_F * ||B_j||

For random A matrices at d/r = 160: E[||A_i^T A_j||_F] ~ O(r/sqrt(d)) which
is small enough that interference stays bounded.

Empirically at N=25: max |cos|(Delta W_i, Delta W_j) = 0.006259
Empirically at N=50: max |cos| = 0.010 (5x below K3 threshold of 0.05)

## Gumbel-Sigmoid Routing (L2R Framework)

### Gate Computation

For input hidden state h (mean-pooled over sequence):

z_i = W_route^T h + b_i,  i = 1..N     (N independent logits)

During training (discrete but differentiable):
g_i = sigma((z_i + G_i) / tau)

where G_i = -log(-log(U_i)), U_i ~ Uniform(0,1) is a Gumbel noise sample.

At inference (hard routing):
g_i = 1 if z_i > 0, else 0

### Why Gumbel-Sigmoid, Not Softmax

Softmax forces sum_i g_i = 1 (competition). As N grows, each adapter's
share shrinks to 1/N -- catastrophic dilution at N=50.

Gumbel-sigmoid uses independent Bernoulli gates:
- P(adapter i active) is independent of P(adapter j active)
- Multiple adapters can be ON simultaneously (multi-label, not multi-class)
- No zero-sum competition between adapters

### Top-k Selection

After computing all N gates g_i, select top-k by logit magnitude:
S = top_k(z_1, ..., z_N, k=2)

Composed output:
y = x + sum_{i in S} (s / k) * x @ A_i @ B_i

Scale by s/k (not s/N) -- only active adapters contribute.

### Router Architecture

Two-layer MLP router:
  Layer 1: W_proj in R^{d x h}, b_proj in R^h       (h = 256)
  Activation: GELU
  Layer 2: W_gate in R^{h x N}, b_gate in R^N

Parameters: d*h + h + h*N + N = 2560*256 + 256 + 256*49 + 49 = 668,977 (~0.65 MB)
Negligible vs adapter params: 49 * 21.6M = 1.06B total adapter params.

## Composition Ratio at N=50

### Uniform Composition (Baseline)
gamma_uniform = PPL(base + sum_i (1/N) * Delta W_i) / PPL(base)

At N=25: gamma = 0.982 (composed is 1.8% BETTER than base)
At N=50: gamma = 0.996 (composed is 0.4% better than base)

Gamma approaches 1.0 as N increases, consistent with 1/N dilution of
individual adapter contributions under uniform averaging. Each adapter's
weight is s/N, so as N grows, the per-adapter signal weakens toward zero
and the composition converges to the base model. The specific functional
form (e.g., 1 - c/sqrt(N) vs 1 - c/N) is not yet determined from the
available data points, as the implied constant c varies 2x across N values:

| N  | gamma | implied c (1/sqrt(N) model) |
|----|-------|-----------------------------|
| 5  | 0.920 | 0.179                       |
| 15 | 0.938 | 0.240                       |
| 25 | 0.982 | 0.090                       |
| 50 | 0.996 | 0.028                       |

Kill criterion K2: gamma > 1.5 would mean catastrophic degradation.
Observed gamma = 0.996, well within bounds.

### Routed Composition (Gumbel Top-2)
gamma_routed = PPL(base + sum_{i in S(x)} (s/k) * Delta W_i) / PPL(base)

Empirically: gamma_routed = 0.632 (37% better than base, 49/49 domains below base)

This is dramatically better than uniform (gamma = 0.996) because:
1. Only relevant adapters contribute (no dilution from irrelevant domains)
2. Scale factor s/k vs s/N (k=2 vs N=49 -> 24.5x stronger signal per adapter)
3. The router learns sensible cross-domain pairings (e.g., math+reasoning, code+sql)

## Cosine Scaling at N=50

Mean cosine between adapter pairs at various N:
| N | Mean |cos| | Max |cos| |
|---|------------|-----------|
| 5 | 0.0020 | ~0.005 |
| 15 | 0.0011 | ~0.004 |
| 25 | 0.0007 | 0.006 |
| 50 | **0.0019** | **0.010** |

With random uniform initialization at d/r = 160, the high-dimensional
geometry ensures near-orthogonality between A matrices with high probability.
Interference stays bounded well below threshold up to the capacity limit.

K3 threshold: max |cos| > 0.05 → KILL. We have 8x margin at N=25.

## Worked Example (Micro Scale)

d=2560, r=16, N=50, k=2

1. Initialize 50 A matrices via random uniform → each A_i is 2560x16
2. Train B_i with STE ternary quantization → each B_i is 16x2560
3. Total adapter storage: 50 * (2560*16 + 16*2560) * 1.58 bits / 8 ≈ 50 * 1.9KB = 95KB
4. Router: ~669K params * 4 bytes = 2.6MB
5. For input x, router computes 50 logits → top-2 → compose 2 adapters
6. Effective model: base + 2 experts, not base + 50 experts

## Computational Cost

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Router forward | 2*d*h + 2*h*N ~ 1.3M | h + N floats ~ 1.2KB |
| Top-k selection | O(N log k) | negligible |
| 2 adapter forwards | 2 * 2 * d * r = 163K | 2 * (d*r + r*d) = 327KB |
| Total overhead | ~1.5M FLOPs | ~2.9MB |

Base model forward: ~4B FLOPs per token (2B params * 2)
Routing overhead: 1.5M / 4B = 0.04% -- negligible.

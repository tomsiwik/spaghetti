# TT-LoRA Port to MLX: Mathematical Foundation

**Paper**: TT-LoRA MoE (arXiv:2504.21190)
**Target**: Gemma 4 E4B on Apple M5 Pro, MLX

## Setup

Standard LoRA adapts weight W via low-rank correction:
$$y = Wx + \alpha \cdot BAx$$
where $B \in \mathbb{R}^{n \times r}$, $A \in \mathbb{R}^{r \times m}$, total parameters $P_{\text{LoRA}} = r(m + n)$.

TT-LoRA replaces LoRA's BA with a Tensor Train decomposition of the weight correction.

## Definition (Tensor Train Decomposition)

Given $\Delta W \in \mathbb{R}^{n \times m}$, factorize dimensions:
- $m = \prod_{i=1}^{d_m} m_i$ (input factors)
- $n = \prod_{j=1}^{d_n} n_j$ (output factors)

Reshape $\Delta W$ as tensor $\mathcal{T} \in \mathbb{R}^{m_1 \times \cdots \times m_{d_m} \times n_1 \times \cdots \times n_{d_n}}$.

The TT decomposition represents $\mathcal{T}$ as a chain of cores:
$$\mathcal{T}[i_1, \ldots, i_d] = G_1[i_1] \cdot G_2[i_2] \cdots G_d[i_d]$$
where $G_k \in \mathbb{R}^{r_{k-1} \times s_k \times r_k}$ with boundary ranks $r_0 = r_d = 1$.

## Theorem 1 (Parameter Count)

**Claim**: TT-LoRA with uniform rank $r$ on a $d$-core decomposition with factor sizes $\{s_k\}$ requires:
$$P_{\text{TT}} = s_1 \cdot r + \sum_{k=2}^{d-1} r \cdot s_k \cdot r + r \cdot s_d = r \cdot s_1 + (d-2) \cdot r^2 \cdot \bar{s} + r \cdot s_d$$
where $\bar{s}$ is the average interior factor size.

**Proof**: Core $G_k$ has $r_{k-1} \times s_k \times r_k$ entries. With $r_0 = r_d = 1$ and $r_k = r$ for $1 \leq k < d$:
- $G_1$: $1 \times s_1 \times r = s_1 r$
- $G_k$ for $2 \leq k \leq d-1$: $r \times s_k \times r = r^2 s_k$
- $G_d$: $r \times s_d \times 1 = r s_d$

Total: $P_{\text{TT}} = r(s_1 + s_d) + r^2 \sum_{k=2}^{d-1} s_k$. QED.

## Predictions for Gemma 4 E4B

### q_proj: in=2560, out=2048

Factorization: $2560 = 5 \times 8 \times 8 \times 8$, $2048 = 4 \times 8 \times 8 \times 8$

TT shape: $[5, 8, 8, 8, 4, 8, 8, 8]$ (8 cores)

| Rank $r$ | $P_{\text{TT}}$ | LoRA $r_L=6$: 27,648 | Compression |
|-----------|------------------|-----------------------|-------------|
| 2 | $2(5+8) + 4 \times 6 \times 8 = 218$ | 27,648 | 126.8x |
| 4 | $4(5+8) + 16 \times 6 \times 8 = 820$ | 27,648 | 33.7x |
| 8 | $8(5+8) + 64 \times 6 \times 8 = 3,176$ | 27,648 | 8.7x |

### v_proj: in=2560, out=512

Factorization: $2560 = 5 \times 8 \times 8 \times 8$, $512 = 8 \times 8 \times 8$

TT shape: $[5, 8, 8, 8, 8, 8, 8]$ (7 cores)

| Rank $r$ | $P_{\text{TT}}$ | LoRA $r_L=6$: 18,432 | Compression |
|-----------|------------------|-----------------------|-------------|
| 2 | $2(5+8) + 4 \times 5 \times 8 = 186$ | 18,432 | 99.1x |
| 4 | $4(5+8) + 16 \times 5 \times 8 = 692$ | 18,432 | 26.6x |
| 8 | $8(5+8) + 64 \times 5 \times 8 = 2,664$ | 18,432 | 6.9x |

### Per-layer total (q + v) at rank 8

$P_{\text{layer}} = 3,176 + 2,664 = 5,840$

**Kill criterion K2 predicts PASS**: 5,840 << 40,000.

## Theorem 2 (Forward Pass Equivalence)

**Claim**: The reconstruction $\hat{\Delta W} = \text{reshape}(\text{contract}(G_1, \ldots, G_d), [n, m])$ satisfies:
$$y = Wx + \alpha \cdot \hat{\Delta W} x$$
which is identical to standard linear + correction.

**Proof**: The contraction $\text{contract}(G_1, \ldots, G_d)$ produces a tensor $\mathcal{T}$ identical to the original TT representation. Reshape to matrix form gives $\hat{\Delta W} = \text{reshape}(\mathcal{T}, [n, m])$. The forward pass $y = Wx + \alpha \hat{\Delta W} x$ is exact up to floating-point precision. QED.

**Self-consistency test**: Reconstruct $\hat{\Delta W}$, compute $y_{\text{recon}} = \text{base}(x) + \alpha (x \cdot \hat{\Delta W}^T)$. Also compute $y_{\text{direct}}$ via sequential core contraction. Check $\|y_{\text{recon}} - y_{\text{direct}}\|_\infty < 10^{-5}$.

## Latency Analysis

The reconstruction approach has two phases:
1. **Core contraction**: $O(d \cdot \max(s_k) \cdot r^2 \cdot \prod s_j)$ — small since total params are ~3K
2. **Matrix-vector product**: $O(B \cdot S \cdot m \cdot n)$ — same as standard linear

The contraction approach avoids materializing $\hat{\Delta W}$ but requires $d$ sequential small-tensor operations, each with kernel launch overhead on Metal.

**Prediction**: On MLX/Metal, reconstruction will be faster than contraction because:
- Core contraction is cheap (3K params → microseconds)
- A single large matmul is more efficiently scheduled than $d$ small operations
- Metal kernel launch overhead dominates for tiny operations

**K3 prediction**: Latency within 2x of LoRA because the dominant cost is x @ W (same for both), and the correction term adds one extra matmul of the same size. Overhead from core reconstruction is negligible.

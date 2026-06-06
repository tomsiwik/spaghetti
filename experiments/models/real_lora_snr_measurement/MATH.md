# Real LoRA SNR Measurement: Mathematical Foundation

## Purpose

Determine whether the adaptive rank selection fallback (designed for low-SNR
conditions in `micro/models/adaptive_rank_snr_fallback/`) has practical benefit
by measuring the spectral profile of real LoRA deltas from the pilot-50
distillation pipeline.

## Setup

**Adapters:** 5 LoRA experts trained on Qwen2.5-7B (d=3584 for most modules)
at rank r=16, all-modules (q, k, v, o, gate, up, down projections).

**Domains:** bash, math, medical, python, sql.

## Definitions

### LoRA Delta

For each (expert, layer, module), the LoRA delta is:

$$\Delta W = B A$$

where $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$,
and the scaling factor $\alpha/r = 16/16 = 1.0$ (no additional scaling needed).

### Singular Value Decomposition

$$\Delta W = U \Sigma V^T$$

where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)$ with
$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$.

Since $\text{rank}(\Delta W) \leq r = 16$, at most 16 singular values are non-zero.

### Signal-to-Noise Ratio (SNR)

$$\text{SNR} = \frac{\sigma_1}{\sigma_r}$$

This is the condition number of the LoRA delta within its rank-r subspace.
High SNR means the delta is dominated by a few directions; low SNR means
all r directions contribute comparably.

**Interpretation:**
- SNR ~ 1: All singular values roughly equal (flat spectrum). Full rank is useful.
- SNR ~ 10: Moderate concentration. Some rank reduction possible.
- SNR >> 100: Highly concentrated. Only a few directions matter; adaptive rank
  could save significant computation.

### Effective Rank Metrics

Cumulative variance fraction:

$$f(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$$

- $r_{95}$: smallest $k$ such that $f(k) \geq 0.95$
- $r_{99}$: smallest $k$ such that $f(k) \geq 0.99$

### Rank Diversity Ratio

$$\rho = \frac{r_{99}}{r_{95}}$$

This measures how spread the tail energy is:
- $\rho \approx 1$: 95% and 99% thresholds need essentially the same rank
  (sharp spectral cutoff, adaptive rank has little to gain)
- $\rho > 1.5$: significant energy in the tail between 95% and 99%
  (different experts may benefit from different rank allocations)

### Kill Criteria

- **K1:** If ALL experts have SNR >= 10 across all layers/modules, the fallback
  is correct but vacuous (no low-SNR conditions exist in practice).
- **K2:** If the $r_{99}/r_{95}$ ratio varies less than 1.5x across experts
  (max ratio / min ratio < 1.5), there is no spectral diversity to exploit.

## Expected Analysis Dimensions

- 5 experts x 28 layers x 7 modules = 980 LoRA deltas
- Per-expert aggregation: mean/std/min/max of SNR, r_95, r_99, rho
- Per-module aggregation: do attention vs MLP modules differ systematically?
- Per-layer aggregation: does SNR vary with depth?
- Cross-expert variance: is there enough diversity for adaptive rank to matter?

## Computational Cost

Each delta requires one SVD of the r x r inner product (since rank <= 16,
we compute SVD of $A^T B^T B A$ or equivalently SVD of B@A truncated).
Total: 980 SVDs of matrices with at most 16 non-zero singular values.
Expected runtime: < 30 seconds on CPU.

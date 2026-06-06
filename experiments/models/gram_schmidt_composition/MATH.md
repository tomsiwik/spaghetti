# Gram-Schmidt Orthogonalization for LoRA Composition: Mathematical Foundations

## 1. Setup

We have a pretrained base model with parameters W_base. For each domain k in
{1, ..., N}, we fine-tune LoRA adapters on the MLP layers (fc1 and fc2) while
freezing all base weights.

Each LoRA adapter produces a delta:
  dW_k = (alpha / r) * A_k @ B_k  in R^{d_in x d_out}

For composition, we flatten each delta dict into a single vector:
  d_k = flatten(dW_k)  in R^D

where D is the total number of parameters across all LoRA-adapted layers.

## 2. The Interference Problem

When composing N experts via additive merging:
  W_merged = W_base + sum_k dW_k

If two deltas d_i and d_j have non-trivial cosine similarity:
  cos(d_i, d_j) = (d_i . d_j) / (||d_i|| * ||d_j||) >> 0

then the overlapping component is counted multiple times. The overlapping
subspace receives N times the intended perturbation, which can degrade quality.

The severity scales with both the cosine similarity and the number of experts.

## 3. Gram-Schmidt Orthogonalization

Given flattened deltas d_1, ..., d_N, the classical Gram-Schmidt process
produces orthogonalized deltas d_1', ..., d_N':

  d_1' = d_1  (first expert is unchanged)
  d_k' = d_k - sum_{i=1}^{k-1} proj(d_k, d_i')  for k = 2, ..., N

where the projection operator is:
  proj(u, v) = (u . v / v . v) * v

### 3.1 Properties

**Orthogonality guarantee**: For all i != j:
  d_i' . d_j' = 0

This is exact (to numerical precision).

**Signal retention**: The fraction of expert k's signal preserved is:
  rho_k = ||d_k'|| / ||d_k||

For expert k, the removed component is the projection onto the subspace
spanned by d_1', ..., d_{k-1}'. By Pythagoras:
  ||d_k||^2 = ||d_k'||^2 + sum_{i=1}^{k-1} |proj(d_k, d_i')|^2

Therefore:
  rho_k = sqrt(1 - sum_{i=1}^{k-1} cos^2(d_k, d_i'))

When pairwise cosines are small (cos << 1), signal retention is approximately:
  rho_k ~ 1 - (1/2) * sum_{i<k} cos^2(d_k, d_i')

### 3.2 Order Dependence

Gram-Schmidt is order-dependent: the first expert retains 100% of its signal,
while later experts lose their projections onto earlier ones. The final
orthogonal basis spans the same subspace regardless of order, but the
allocation of signal across experts changes.

For the SUM merge (d_1' + d_2' + ... + d_N'), order affects the result because
each d_k' depends on the ordering. For the AVERAGE merge (1/N * sum d_k'),
order affects each term but the overall quality impact is smaller.

## 4. Merging Strategies

### 4.1 Naive Sum
  delta_merged = sum_k d_k
No orthogonalization. Overlapping components are double-counted.

### 4.2 Simple Average (1/N)
  delta_merged = (1/N) * sum_k d_k
Scales down by N, which controls the perturbation magnitude but does not
remove interference in the overlapping subspace.

### 4.3 GS Sum
  delta_merged = sum_k d_k'
Orthogonalizes first, then sums. Each expert contributes only its novel
component. However, the total perturbation magnitude scales with N.

### 4.4 GS Average (1/N)
  delta_merged = (1/N) * sum_k d_k'
Orthogonalizes first, then averages. This is the principled combination:
remove interference AND control perturbation magnitude.

## 5. Computational Cost

**Gram-Schmidt**: O(N^2 * D) where D is the flattened delta dimension.
  - N inner products (each O(D)) for each of N experts
  - At micro scale: D ~ 4 layers * 2 sublayers * 64 * 64 = 32,768
  - At macro scale: D ~ 32 layers * 2 sublayers * 4096 * 64 = 16,777,216

**Merge overhead**: < 5ms at micro scale (N=5, D=32K). Negligible compared
to training time. At macro scale with N=100, still < 1 second.

## 6. Worked Example (Micro Scale)

Setup: d=64, n_layer=4, rank=8, N=5 experts

Observed pairwise cosines (seed=42):
  max |cos| = 0.031 (k_o vs a_e)

Signal retention after GS:
  a_e: 1.0000 (first, unchanged)
  f_j: 0.9998
  k_o: 0.9994
  p_t: 0.9995
  u_z: 0.9994

Post-GS cosines: all < 1e-6 (machine precision orthogonality)

Expected retention from formula:
  rho_5 ~ 1 - (1/2) * 4 * 0.031^2 = 1 - 0.0019 = 0.9981

Observed: 0.9994, consistent with the bound (actual overlaps are smaller
than the max).

## 7. Key Insight: GS is Unnecessary When Deltas Are Already Near-Orthogonal

The critical finding: at d=64, r=8, pairwise cosines are already 0.01-0.06.
Signal retention is >99.6% for all experts. The Gram-Schmidt projection
removes <0.4% of any expert's signal.

This means:
1. The deltas are already near-orthogonal (confirming cos~0.0002 at d=896)
2. GS adds no benefit: the interference it removes is negligible
3. GS adds no harm: it preserves >99% of all signals

The GS average and simple average produce nearly identical results because
the input deltas are already nearly orthogonal. GS is a solution looking
for a problem that does not exist in the LoRA composition regime.

## 8. Assumptions

1. **Linear delta path**: LoRA deltas compose via matrix addition without
   intervening nonlinearities. Validated in prior experiments (lora_procrustes).
2. **Flattened vector space**: Treating all parameters as a single vector
   is valid for measuring interference. Layer-wise GS would be an alternative.
3. **Cosine similarity as interference metric**: High cosine implies
   overlapping function in weight space. This is a sufficient but not
   necessary condition for functional interference.

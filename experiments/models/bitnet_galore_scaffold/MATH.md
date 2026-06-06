# GaLore Scaffold: Mathematical Foundations

## Notation

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| W | Weight matrix | (m, n) |
| G | Full gradient dG/dW | (m, n) |
| P | Left projection matrix (top-r left singular vectors of G) | (m, r) |
| r | GaLore projection rank | scalar |
| T | SVD update frequency (steps between re-computing P) | scalar |
| G_proj | Projected gradient P^T G | (r, n) |
| m_t, v_t | Adam first/second moments in projected space | (r, n) |
| alpha | Learning rate | scalar |
| d | Model hidden dimension | scalar |
| N | Number of LoRA adapters | scalar |

## GaLore Algorithm

### Core Idea

GaLore maintains optimizer states in a low-rank projected space rather than
full parameter space. At each step:

1. Compute full gradient G in R^(m x n)
2. Project: G_proj = P^T G in R^(r x n) where P in R^(m x r)
3. Run Adam on G_proj (states m_t, v_t are (r, n) -- memory savings!)
4. Reconstruct: delta_W = P * Adam_update(G_proj) in R^(m x n)
5. Every T steps: recompute P from SVD(G)

### Memory Analysis

Standard Adam for a (m, n) weight matrix stores:
- W: m*n parameters
- m_t: m*n first moments
- v_t: m*n second moments
- Total optimizer state: 2*m*n

GaLore stores:
- W: m*n parameters (unchanged -- full-rank model)
- m_t: r*n first moments
- v_t: r*n second moments
- P: m*r projection matrix
- Total optimizer state: 2*r*n + m*r

Memory ratio: (2rn + mr) / (2mn) = r/m + r/(2n) approx r/m for large n.

For d=256, r=64: ratio = 64/256 = 0.25 (75% optimizer memory savings).

### SVD Projection Update

Every T steps, compute thin SVD of current gradient:
  G = U S V^T
where U in R^(m x min(m,n)), S diagonal, V in R^(n x min(m,n)).

Take P = U[:, :r] (top-r left singular vectors).

Cost: O(m*n*min(m,n)) per SVD, amortized over T steps.

## Ternary Quantization

### Absmean Quantization

For weight matrix W in R^(m x n):
1. Compute threshold: tau = mean(|W|)
2. Ternary map: Q(w) = sign(w) * I(|w| > tau)
3. Scale: s = mean(|w| : |w| > tau) (mean of non-zero entries)
4. Quantized: W_q = s * Q(W)

### Quantization-Friendliness

The key finding: GaLore-trained weights are LESS quantization-friendly than
standard-trained weights. After ternary quantization:

| | Standard | GaLore | Ratio |
|--|----------|--------|-------|
| Pre-quant PPL | 15.92 | 13.31 | 0.836 |
| Post-quant PPL | 17.34 | 34.26 | 1.975 |
| Quant degradation | 1.09x | 2.57x | 2.36x |

Hypothesis for the gap: GaLore's low-rank gradient projection produces weights
with higher effective rank (more information distributed across singular values),
making the ternary threshold less effective. Standard training with full-rank
gradients naturally produces weights closer to ternary-friendly distributions.

## Composition Theory

### LoRA Composition with 1/N Scaling

Given N domain adapters {delta_W_i = B_i A_i}_i=1^N, composed model:
  W_composed = W_base + (1/N) sum_i B_i A_i

Composition ratio = PPL(composed) / PPL(best individual)

### Adapter Orthogonality

Pairwise cosine between flattened adapter parameter vectors:
  cos(i,j) = <vec(theta_i), vec(theta_j)> / (||vec(theta_i)|| * ||vec(theta_j)||)

where theta_i = {A_i, B_i} concatenated.

### Observed Composition Quality

| Metric | Standard | GaLore |
|--------|----------|--------|
| Composition ratio | 1.077 | 1.155 |
| Mean |cos| | 0.00295 | 0.00322 |
| Max |cos| | 0.00721 | 0.00974 |

Both are excellent (comp ratio near 1.0, cosines near 0). The GaLore scaffold
produces slightly less orthogonal adapters but within noise of the standard.

## Worked Example (Micro Scale)

Model: d=256, 6 layers, 4 heads. Total 6.4M params.

GaLore parameters per layer:
- q/k/v/o_proj: (256, 256) -> 4 matrices, GaLore rank 64
- gate/up_proj: (256, 1024) -> 2 matrices, GaLore rank 64
- down_proj: (1024, 256) -> 1 matrix, GaLore rank 64
- Total: 7 matrices per layer, 42 total

Optimizer memory per GaLore matrix (256 x 256):
- Standard: 2 * 256 * 256 = 131,072
- GaLore: 2 * 64 * 256 + 256 * 64 = 49,152
- Savings: 62.5%

For larger matrices (256 x 1024):
- Standard: 2 * 256 * 1024 = 524,288
- GaLore: 2 * 64 * 1024 + 256 * 64 = 147,456
- Savings: 71.9%

LoRA adapter overhead:
- Per layer: 7 * (256*16 + 16*256) = 7 * 8192 = 57,344
- Total: 6 * 57,344 = 344,064 params
- As fraction of model: 344,064 / 6,395,648 = 5.4%
- (Note: larger than typical because model is very small)

## Assumptions

1. Character-level tokenization (vocab=133) is a proxy for BPE tokenization
2. 2000 pretraining steps is sufficient to show GaLore convergence behavior
3. Absmean ternary quantization is representative of production STE quantization
4. 5 domains is sufficient to test composition quality
5. The mechanism (GaLore -> quantize -> LoRA -> compose) transfers from d=256 to d=2560
6. **GaLore moment-persistence deviation**: Our implementation does NOT reset
   Adam moments (m_t, v_t) when the projection matrix P is recomputed every
   T=200 steps. The original GaLore paper (Zhao et al., 2024) discusses that
   when P changes, the existing moments are in the OLD projected space and may
   not be compatible with the NEW projected space. The paper proposes resetting
   or projecting moments. With T=200 over 2000 total steps, there are only 10
   SVD updates, so the impact is limited. This deviation weakly biases AGAINST
   GaLore (slightly worse optimization), making the 0.834x pre-quantization PPL
   advantage more impressive rather than less. If anything, correcting this
   deviation would strengthen the GaLore result.

## Limitation: Scale Gap

This experiment uses d=256 with 6.4M params. BitNet-2B-4T has d=2560 with 2.4B params.
The ternary quantization gap may behave differently at larger scale because:
- Larger weight matrices have more stable statistical properties
- GaLore rank can be proportionally larger (e.g., r=256 at d=2560)
- STE-aware GaLore training (not tested here) may eliminate the quantization gap

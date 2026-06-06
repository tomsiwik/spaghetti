# Attention LoRA Cosine as Domain Similarity Predictor: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | FFN intermediate dimension | 256 |
| d_head | Per-head dimension | 16 |
| n_h | Number of attention heads | 4 |
| L | Number of layers | 2 |
| r | LoRA rank | 8 |
| alpha | LoRA scaling factor | 8 |
| N | Number of domain experts | 12 |
| C | Number of domain clusters | 4 |
| n_c | Domains per cluster | 3 |

## 2. Domain Similarity Ground Truth

### 2.1 Cluster Structure

Twelve domains are organized into 4 clusters:

    code:      {python, javascript, rust}
    reasoning: {math, logic, physics}
    knowledge: {medical, law, history}
    creative:  {poetry, fiction, comedy}

### 2.2 Graduated Similarity Matrix

The ground-truth semantic similarity S in [0, 1]^{N x N} is defined as:

    S(i, j) = 1.0                         if i = j
    S(i, j) = 0.7                         if cluster(i) = cluster(j), i != j
    S(i, j) = CLUSTER_SIM(c_i, c_j)      otherwise

Inter-cluster similarities (motivated by real semantic relationships):

| Pair | Similarity | Rationale |
|------|-----------|-----------|
| code-reasoning | 0.5 | logical structures shared |
| reasoning-knowledge | 0.4 | statistical/scientific overlap |
| knowledge-creative | 0.2 | narrative overlap |
| code-knowledge | 0.15 | largely unrelated |
| code-creative | 0.1 | very different |
| reasoning-creative | 0.1 | largely unrelated |

This gives N(N-1)/2 = 66 unique pairs for correlation analysis.

The ground truth has 5 distinct similarity levels: {0.1, 0.15, 0.2, 0.4, 0.5, 0.7},
ensuring Spearman correlation can discriminate ranked ordering, not just binary.

## 3. LoRA Delta Decomposition

### 3.1 Per-Layer Delta Structure

Each layer l has LoRA on 6 weight matrices:
- Attention: W_q, W_k, W_v, W_o (shape d x d each)
- FFN: W_1 (d x d_ff), W_2 (d_ff x d)

For each matrix, the LoRA delta is:

    dW = (alpha/r) * A @ B      where A: (d_in, r), B: (r, d_out)

### 3.2 Module-Specific Delta Vectors

Attention delta vector (per expert):

    delta_attn = concat[ vec(A_q[l] @ B_q[l]), vec(A_k[l] @ B_k[l]),
                          vec(A_v[l] @ B_v[l]), vec(A_o[l] @ B_o[l])
                          for l in 0..L-1 ]

    dim(delta_attn) = L * 4 * d * d = 2 * 4 * 64 * 64 = 32,768

FFN delta vector (per expert):

    delta_ffn = concat[ vec(A_1[l] @ B_1[l]), vec(A_2[l] @ B_2[l])
                        for l in 0..L-1 ]

    dim(delta_ffn) = L * (d * d_ff + d_ff * d) = 2 * 2 * 64 * 256 = 65,536

Full delta:

    delta_full = concat[ delta_attn, delta_ffn ]
    dim(delta_full) = 98,304

### 3.3 Cosine Similarity Matrices

For N experts with delta vectors {delta_i}:

    C_attn(i, j) = <delta_attn_i, delta_attn_j> / (||delta_attn_i|| * ||delta_attn_j||)
    C_ffn(i, j)  = <delta_ffn_i,  delta_ffn_j>  / (||delta_ffn_i||  * ||delta_ffn_j||)
    C_full(i, j) = <delta_full_i, delta_full_j> / (||delta_full_i|| * ||delta_full_j||)

## 4. Statistical Analysis

### 4.1 Primary Metric: Spearman Rank Correlation

We compute rho_s between the upper triangle of |C_module| and the upper
triangle of S (ground truth), both flattened to vectors of length N(N-1)/2 = 66.

    rho_s = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))

where d_i is the rank difference for pair i.

Kill threshold: rho_s >= 0.3 AND p < 0.05 to PASS.

### 4.2 Secondary Metric: Within/Cross Ratio

    R_module = mean(|C_module(i,j)| for i,j in same cluster)
             / mean(|C_module(i,j)| for i,j in different clusters)

R > 1 means within-cluster deltas are more similar, indicating the module
captures cluster structure.

### 4.3 Random Baseline

For random unit vectors in D dimensions:

    E[|cos|] = sqrt(2 / (pi * D))

    Attn: E[|cos|] = sqrt(2 / (pi * 32768)) = 0.0044
    FFN:  E[|cos|] = sqrt(2 / (pi * 65536)) = 0.0031

Expected ratio R for random vectors: ~1.0 (no structure).

## 5. SPSA Training

### 5.1 Algorithm

We use Simultaneous Perturbation Stochastic Approximation (Spall 1992) to
train all B matrices simultaneously:

    1. Sample perturbation Delta_i ~ Rademacher(+1, -1) for each B parameter
    2. Evaluate L(theta + eps * Delta) and L(theta - eps * Delta)
    3. Gradient estimate: g_i = (L+ - L-) / (2 * eps * Delta_i)
    4. Update: theta <- theta - lr * g

SPSA convergence: E[g] = nabla L(theta) (unbiased). Converges with
O(2 forward passes per step) regardless of parameter count.

### 5.2 Cost

Per domain: 300 steps * 2 forward passes = 600 forward passes.
Total: 12 domains * 600 = 7,200 forward passes.
Runtime: ~120s on CPU (Apple Silicon).

## 6. Key Limitation: Training Signal Strength

The training losses remain near log(V) = log(32) = 3.466 throughout,
indicating the LoRA deltas reflect gradient direction (early stochastic
movement) rather than converged domain-specific features. The delta norms
are O(0.01), which is ~100x smaller than what converged macro-scale
training produces.

This means:
- Cosine similarities are dominated by initialization noise and random
  walk artifacts, not learned domain structure.
- The SPSA gradient estimate adds additional noise that further obscures
  any domain signal.
- At macro scale (d=4096, real data, converged training), both attention
  and FFN deltas would be dominated by learned features, not noise.

## 7. Worked Example

With N=12, C=4, n_c=3:

    Total pairs: 12 * 11 / 2 = 66
    Within-cluster pairs: 4 * (3 * 2 / 2) = 12
    Cross-cluster pairs: 66 - 12 = 54

Ground truth S for one within-cluster pair (python-javascript):
    S = 0.7

Ground truth S for one cross-cluster pair (python-math):
    S = CLUSTER_SIM(code, reasoning) = 0.5

If attention captures domain structure perfectly:
    |C_attn(python, javascript)| >> |C_attn(python, math)| >> |C_attn(python, poetry)|

Observed (seed 42):
    |C_attn(python, javascript)| = random noise, no systematic ordering

## 8. Assumptions

1. **Graduated similarity is well-defined**: The cluster similarity values
   (0.5 for code-reasoning, 0.4 for reasoning-knowledge, etc.) are
   researcher-assigned, not empirically measured. Different assignments
   would change correlation values.

2. **Synthetic Markov chains approximate domain structure**: Real domains
   differ in complex semantic ways; our synthetic domains differ only in
   character-level transition probabilities. The domain signal at micro
   scale may be too weak for any module to capture.

3. **SPSA training is sufficient**: While SPSA converges in expectation,
   300 steps may be insufficient for the gradient estimate to dominate
   noise, especially for attention parameters which interact through the
   softmax nonlinearity.

4. **Delta cosine equals interference**: We assume cosine similarity of
   expanded deltas (A@B products) measures composition interference.
   This is the standard SOLE assumption (see structural_orthogonality_proof).

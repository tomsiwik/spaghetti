# MI Expert Independence: Mathematical Foundations

## Notation

| Symbol | Type | Description |
|--------|------|-------------|
| d | scalar | Embedding dimension (64 at micro scale) |
| G | scalar | Number of capsule groups per layer |
| P | scalar | Capsules per group (64 at micro scale) |
| N | scalar | Number of calibration samples |
| x_t | R^d | Hidden state at position t |
| f_g(x) | R^d | Output of capsule group g |
| a_g(x) | R^P | Activation vector of group g (post-ReLU) |
| k | scalar | KSG neighbor count |

## Expert Output Model

Each capsule group g produces output:

$$f_g(x) = B_g \cdot \text{ReLU}(A_g \cdot x)$$

where A_g in R^{P x d} (detectors), B_g in R^{d x P} (expansions).

The activation vector is:

$$a_g(x) = \text{ReLU}(A_g \cdot x) \in \mathbb{R}^P$$

## Independence Metrics

### Cosine Similarity (Baseline)

Given N calibration samples, collect group output matrices
O_g in R^{N x d} where row t is f_g(x_t).

$$\text{cos}(g_i, g_j) = \frac{\text{vec}(O_i)^T \text{vec}(O_j)}{||\text{vec}(O_i)|| \cdot ||\text{vec}(O_j)||}$$

This flattens all N*d values into a single vector and computes the angle.
Cosine captures only linear correlation between the flattened output vectors.

**Complexity**: O(N * d * G^2) -- one pass over outputs for each pair.

### Mutual Information via KSG (Activation-Level)

For each group, compute the mean activation scalar:

$$\bar{a}_g(x_t) = \frac{1}{P} \sum_{p=1}^{P} a_{g,p}(x_t)$$

This gives a 1D summary of "how active was group g on sample t".

MI between groups i and j:

$$\hat{I}(\bar{a}_i; \bar{a}_j) = \psi(k) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle + \psi(N)$$

where psi is the digamma function, n_x and n_y count neighbors within
epsilon of each point in the marginal spaces, and epsilon is the distance
to the k-th neighbor in joint space. (Kraskov et al. 2004, Algorithm 1)

**Why mean activation?** Operating on 1D scalars makes KSG reliable with
small N (640 samples). The mean activation captures the overall firing
intensity of the group, which is the natural summary statistic for
"how much is this expert contributing?"

**Complexity**: O(N * P * G + N * log(N) * G^2)
- First term: computing mean activations
- Second term: KD-tree construction and k-NN queries for each pair

### Mutual Information via KSG (PCA-Reduced Outputs)

Project outputs to d_pca dimensions via joint PCA:

$$\tilde{O}_g = (O_g - \mu) V_{:d_{pca}}^T$$

where V comes from SVD of concatenated centered outputs.

Then apply KSG to the d_pca-dimensional projected outputs.

**Complexity**: O(N * d * G + N * d^2 + N * log(N) * G^2)
- Higher due to SVD and higher-dimensional KD-trees

## Predictive Power Analysis

For each metric M in {cosine, MI-act, MI-PCA}, compute the mean
pairwise value across all G*(G-1)/2 group pairs:

$$\bar{M} = \frac{2}{G(G-1)} \sum_{i<j} M(g_i, g_j)$$

Then compute r-squared between the metric values and validation loss
across different routing configurations:

$$r^2(M, \mathcal{L}) = \left(\frac{\text{Cov}(\bar{M}, \mathcal{L})}{\sigma_{\bar{M}} \cdot \sigma_{\mathcal{L}}}\right)^2$$

A metric with higher r-squared is a better predictor of composition quality.

## Worked Example (d=64, G=4, P=64, N=640)

**Cosine computation**:
- Each group output: 640 x 64 = 40,960 values
- Per pair: 2 * 40,960 multiplications + norms = ~120K FLOPs
- G*(G-1)/2 = 6 pairs
- Total: ~720K FLOPs

**MI-activation computation**:
- Mean activation per group: 640 * 64 = 40,960 ops
- KD-tree build (2D, N=640): O(640 * log(640)) ~ 5,900
- k-NN queries (N=640, k=3): O(640 * log(640)) ~ 5,900
- Neighbor counting: O(640 * log(640)) ~ 5,900
- Per pair: ~17K ops + overhead
- 6 pairs: ~102K ops
- Plus: digamma evaluations (640 * 2 * 6 = 7,680)
- Total: much smaller in FLOPs but dominated by k-NN constant factors

**Cost ratio** (empirical): MI-act takes ~96x cosine wall-clock time,
dominated by KD-tree overhead and Python loop over N=640 points.

## Assumptions

1. **Mean activation is a sufficient statistic for group-level dependence.**
   Losing per-capsule detail may miss fine-grained interactions, but the
   mean captures the overall "contribution magnitude" which is what the
   router optimizes.

2. **KSG is reliable at N=640 for 1D inputs.** Theory requires N >> k.
   At N=640, k=3, this is well-satisfied. For d_pca=4, reliability
   degrades (empirically confirmed by higher variance in MI-PCA).

3. **Routing configuration affects both independence metrics and quality.**
   If all configs produce near-identical quality (narrow loss range),
   the correlation test has low statistical power.

4. **Cosine on flattened vectors is the fair baseline.** Alternative:
   per-sample cosine averaged across positions. The flattened version
   matches the behavioral_dedup convention in this project.

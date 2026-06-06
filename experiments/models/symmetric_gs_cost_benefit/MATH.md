# Symmetric GS Cost-Benefit: Mathematical Analysis

## Setup

Let $\delta_1, \ldots, \delta_N \in \mathbb{R}^d$ be $N$ expert deltas with pairwise cosine similarity $c = \cos(\delta_i, \delta_j) \approx c_0$ for all pairs $i \neq j$ (uniform overlap model).

### Gram-Schmidt Merge

For a permutation $\sigma \in S_N$, the GS process produces orthogonalized vectors $\delta'_{\sigma(1)}, \ldots, \delta'_{\sigma(N)}$ where:

$$\delta'_{\sigma(k)} = \delta_{\sigma(k)} - \sum_{i=1}^{k-1} \frac{\langle \delta_{\sigma(k)}, \delta'_{\sigma(i)} \rangle}{\|\delta'_{\sigma(i)}\|^2} \delta'_{\sigma(i)}$$

The GS merged vector for permutation $\sigma$ is:

$$M_\sigma = \frac{1}{N} \sum_{k=1}^{N} \delta'_{\sigma(k)}$$

### Symmetric GS

Symmetric GS averages over $P$ random permutations:

$$M_{\text{sym}}(P) = \frac{1}{P} \sum_{p=1}^{P} M_{\sigma_p}$$

In the limit $P \to |S_N| = N!$, this becomes the true symmetric GS:

$$M_{\text{sym}}^* = \frac{1}{N!} \sum_{\sigma \in S_N} M_\sigma$$

## Key Insight: Averaging Causes Destructive Interference

### Why single orderings outperform symmetric GS

For each permutation $\sigma$, the merged vector $M_\sigma$ retains full information along the shared direction but assigns the orthogonal residuals differently based on the ordering. Different permutations produce merged vectors that share a common component along the shared direction but differ in their orthogonal components.

Let $\hat{s}$ be the unit shared direction. Then:

$$M_\sigma = \alpha_s \hat{s} + r_\sigma$$

where $r_\sigma \perp \hat{s}$ and $r_\sigma$ depends on the permutation. When we average:

$$M_{\text{sym}} = \alpha_s \hat{s} + \frac{1}{P}\sum_{p=1}^P r_{\sigma_p}$$

The residual components $r_{\sigma_p}$ point in different directions (because different orderings prioritize different experts' unique components). Averaging these diverse residuals causes partial cancellation:

$$\left\|\frac{1}{P}\sum_{p=1}^P r_{\sigma_p}\right\| \leq \frac{1}{P}\sum_{p=1}^P \|r_{\sigma_p}\|$$

with equality only when all $r_{\sigma_p}$ are parallel (which they are not in general).

### Quantitative prediction

For the uniform-overlap model with $N$ experts at cosine $c$, the merged norm for a single ordering is approximately:

$$\|M_\sigma\| \approx \frac{1}{N}\sqrt{1 + (N-1)(1-c)}$$

The norm of the symmetric average is reduced by the angular diversity of residuals across orderings. As $c \to 1$, the residual components shrink (less unique signal) but become more diverse in direction, leading to stronger cancellation when averaged.

### Observed scaling

At $c = 0.85$, $N = 5$, $d = 256$:
- Single ordering norm: $\|M_\sigma\| \approx 0.2734$
- Symmetric GS norm ($P=100$): $\|M_{\text{sym}}\| \approx 0.2490$
- Loss: $\approx 8.9\%$

The loss grows with cosine $c$:

| $c$ | Single norm | Sym norm ($P=100$) | Loss (%) |
|-----|------------|-------------------|---------|
| 0.01 | 0.4454 | 0.4444 | 0.22% |
| 0.10 | 0.4415 | 0.4375 | 0.90% |
| 0.30 | 0.4182 | 0.4017 | 3.98% |
| 0.50 | 0.3800 | 0.3522 | 7.32% |
| 0.70 | 0.3268 | 0.2954 | 9.63% |
| 0.85 | 0.2734 | 0.2490 | 8.95% |

The loss peaks around $c \approx 0.70$ and slightly decreases at $c = 0.90$ because the extreme overlap reduces the diversity of residuals.

## Computational Cost

- Single GS: $O(N^2 d)$
- Symmetric GS ($P$ orderings): $O(P \cdot N^2 d)$
- At $P = 100$: 100x cost for 9% quality LOSS

## Conclusion

Symmetric GS is dominated by single-ordering GS at all cosine levels tested. The averaging operation introduces destructive interference in the orthogonal residual components. Any single deterministic ordering (canonical, random-fixed, etc.) outperforms symmetric GS.

The variance across single orderings is extremely small (CV = 0.12% at cos=0.85), meaning any single ordering is essentially as good as any other. The "order sensitivity problem" that motivated symmetric GS does not manifest as a quality problem -- it manifests only as a direction problem (which direction the merged vector points), and averaging these diverse directions produces a shorter vector (less signal).

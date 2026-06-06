# Content-Aware Routing: Mathematical Foundations

## 1. Problem Setup

Given:
- Base model $f_\theta: \mathbb{R}^{B \times T} \to \mathbb{R}^{B \times V}$ (frozen MLP, $d=64$, $d_{ff}=256$, 4 layers)
- $N = 15$ LoRA experts $\{\Delta_i\}_{i=1}^{N}$, each rank $r=8$
- 3 semantic clusters $\{C_k\}_{k=1}^{3}$, each containing 5 domains
- Query $x \in \mathbb{R}^T$ (token sequence, $T=16$, $V=32$)

**Goal:** Find routing function $R: \mathbb{R}^T \to \{1, \dots, N\}$ that minimizes:

$$\mathcal{L}_{route} = \mathbb{E}_{(x,y) \sim \mathcal{D}_i}\left[-\log p(y \mid x; \theta + \Delta_{R(x)})\right]$$

where $R(x)$ selects the expert and $\mathcal{D}_i$ is the true domain of query $x$.

## 2. Routing Strategies

### 2.1 Hash Ring (baseline)

$$R_{hash}(x) = \arg\min_{j \in \{1,\dots,N\}} d_{ring}(h(x), v_j)$$

where $h: \{0,\dots,V-1\}^T \to [0, 2^{32})$ is MD5 hash of the byte representation,
$v_j$ are virtual node positions (150 per expert), and $d_{ring}$ is circular distance.

- **Complexity:** $O(\log(N \cdot V_{nodes}))$ via binary search on sorted ring positions
- **Parameters:** 0 (no learned parameters)
- **Properties:** Content-agnostic, uniform distribution, 1/N displacement on expert add/remove

### 2.2 Cosine Similarity Router

$$R_{cos}(x) = \arg\max_{j} \frac{\text{emb}(x) \cdot \mu_j}{\|\text{emb}(x)\| \cdot \|\mu_j\|}$$

where $\text{emb}(x) = \frac{1}{T}\sum_{t=1}^{T} W_E[x_t] \in \mathbb{R}^d$ (bag-of-words embedding)
and $\mu_j = \frac{1}{|\mathcal{D}_j|}\sum_{x \in \mathcal{D}_j} \text{emb}(x)$ is the centroid of expert $j$'s training data.

- **Complexity:** $O(Nd)$ per query (matrix-vector product)
- **Parameters:** $N \times d = 15 \times 64 = 960$ stored centroids (not learned)
- **Training:** None (centroids computed from training data embeddings)

### 2.3 MLP Classifier Router

$$R_{MLP}(x) = \arg\max_{j} \text{softmax}(W \cdot \text{emb}(x) + b)_j$$

where $W \in \mathbb{R}^{N \times d}$, $b \in \mathbb{R}^N$.

- **Complexity:** $O(Nd)$ per query
- **Parameters:** $N \times d + N = 15 \times 64 + 15 = 975$ (learned)
- **Training:** Cross-entropy on $(\text{emb}(x_i), \text{domain}_i)$ pairs, SGD, 500 steps

### 2.4 Keyword Frequency Router

$$R_{kw}(x) = \arg\min_{j} \|f(x) - p_j\|_2^2$$

where $f(x) \in \Delta^{V-1}$ is the normalized character frequency histogram of query $x$,
and $p_j \in \Delta^{V-1}$ is the aggregate frequency profile of expert $j$'s training data.

- **Complexity:** $O(NV)$ per query
- **Parameters:** $N \times V = 15 \times 32 = 480$ stored profiles (not learned)

## 3. Theoretical Analysis

### 3.1 Random Baseline

With 15 experts, random routing achieves accuracy $1/N = 1/15 \approx 0.067$.
Cluster-level random accuracy = $1/3 \approx 0.333$.

### 3.2 Routing Quality Bound

Let $\mathcal{L}^*_i = \mathcal{L}(x; \theta + \Delta_i)$ be the loss with the correct expert
and $\mathcal{L}_{wrong} = \mathcal{L}(x; \theta + \Delta_j)$ for a wrong expert $j \neq i$.

The quality gap from routing is:

$$\Delta\mathcal{L} = (1 - \text{acc}) \cdot (\mathbb{E}[\mathcal{L}_{wrong}] - \mathbb{E}[\mathcal{L}^*])$$

When experts are near-orthogonal (cos $\approx 0$) and weakly specialized
($\mathcal{L}_{wrong} \approx \mathcal{L}^*$), routing quality is bounded:

$$\Delta\mathcal{L} \leq (1 - \text{acc}) \cdot \epsilon$$

where $\epsilon \to 0$ as expert specialization vanishes.

### 3.3 Cluster vs Domain Discrimination

The Markov chain data generation creates structure at two levels:
- **Between clusters:** Different prototype transition matrices with biased character groups
- **Within clusters:** Small perturbations ($\sigma = 0.15$) around shared prototype

The embedding signal-to-noise ratio for cluster discrimination:

$$\text{SNR}_{cluster} = \frac{\|\mu_{C_k} - \mu_{C_l}\|}{\sigma_{within}} \gg 1$$

For within-cluster domain discrimination:

$$\text{SNR}_{domain} = \frac{\|\mu_i - \mu_j\|}{\sigma_{within}} \approx O(1)$$

This predicts: cluster-level routing should be easy, domain-level routing hard.

## 4. Worked Example

At micro scale ($d=64$, $N=15$, $V=32$):

| Component | Shape | FLOPs per query |
|-----------|-------|-----------------|
| Embedding | $(T, d) \to (d,)$ | $Td = 1024$ (sum) |
| Cosine router | $(d,) \times (N, d)$ | $2Nd = 1920$ |
| MLP router | $(d,) \times (N, d) + (N,)$ | $2Nd = 1920$ |
| Keyword router | $(V,) \times (N, V)$ | $2NV = 960$ |
| Hash ring | MD5 + binary search | ~$100$ + $O(\log 2250)$ |

All routing strategies are sub-microsecond at this scale.
At macro scale ($d=4096$, $N=1000$), cosine/MLP would be $O(Nd) = 4M$ FLOPs,
still sub-millisecond on GPU.

## 5. Assumptions

1. Bag-of-words embedding preserves enough domain signal for routing
   (valid for Markov chain data with distinct character frequency profiles)
2. Expert centroids are representative of the domain distribution
   (computed from training data, not necessarily representative of test distribution)
3. LoRA experts have learned domain-specific features
   (VIOLATED at micro scale: loss ~3.466 throughout, experts barely train)
4. Linear classifier (MLP) has sufficient capacity for 15-class discrimination
   (should be sufficient given $d=64 > N=15$)

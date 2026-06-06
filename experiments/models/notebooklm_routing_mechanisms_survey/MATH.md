# Routing Mechanisms for Composable Ternary Experts: Mathematical Analysis

## 0. Failure Mode & Impossibility Structure

### The Three Routing Failure Modes

**F1: Expert Collapse (All tokens route to same expert)**
- Degenerate state: router output g(x) = e_j for all x, where e_j is a fixed one-hot vector
- This wastes N-1 experts and reduces to a single-adapter model

**F2: Load Imbalance (Skewed utilization)**
- Degenerate state: expert utilization u_i = (1/T) sum_t g_i(x_t) concentrates on k << N experts
- Wastes memory for inactive experts, degrades quality for overloaded ones

**F3: Routing Overhead Exceeds Composition Benefit**
- Degenerate state: t_route > t_expert, making routing more expensive than just running all experts
- At our scale: t_route = 0.166ms, t_expert = 36ms, so overhead is 0.46% (ALREADY SOLVED)

### What Mathematical Structures Prevent Each Failure

**F1 Prevention — Expert Collapse:**

Three proven approaches exist with mathematical guarantees:

1. **Auxiliary Load-Balancing Loss** (Switch Transformer, arxiv 2101.03961):
   L_balance = alpha * N * sum_i(f_i * P_i)
   where f_i = fraction of tokens routed to expert i, P_i = average router probability for expert i.
   At collapse (all tokens to expert j): L_balance = alpha * N * 1.0 * P_j = alpha * N * P_j.
   This penalizes concentration but does NOT make collapse impossible — the task loss can still
   dominate if alpha is too small.

2. **Grassmannian Concentration Bound** (arxiv 2602.17798):
   Routes via Matrix Bingham distribution on Gr(r, d):
   p(U | Lambda) = (1/Z(Lambda)) * exp(tr(Lambda * U^T S U))
   where U in Gr(r, d), Lambda = diag(lambda_1, ..., lambda_r) is the concentration parameter.
   Key theorem: P(collapse) <= exp(-c * min_gap(Lambda)) where min_gap is the minimum
   eigenvalue gap. By controlling Lambda, collapse probability is exponentially suppressed.
   THIS IS THE STRONGEST GUARANTEE — collapse has exponentially vanishing probability.

3. **Gumbel-Sigmoid Independent Gates** (our proven mechanism):
   g_i(x) = sigma((W_i x + b_i + epsilon_i) / tau), epsilon_i ~ Gumbel(0,1)
   Each gate is an independent Bernoulli — no softmax zero-sum constraint.
   P(collapse) = prod_i P(g_i = indicator(i=j)) = prod_{i!=j} (1-sigma(...)) * sigma(...)
   For N experts with balanced logits, P(all-same) ~ (1/2)^(N-1), which is 2^{-24} ~ 6e-8
   at N=25. Not mathematically impossible but probabilistically negligible.

**F2 Prevention — Load Imbalance:**

1. **Hash Routing** (arxiv 2106.04426): Deterministic balanced mapping, O(0) trainable params.
   Perfect balance by construction: u_i = 1/N +/- 1/T (random hash).
   Impossibility of imbalance: hash function has no learned parameters to concentrate.

2. **LD-MoLE Analytical Sparsity Control** (arxiv 2509.25684):
   Differentiable routing with closed-form solution for dynamic expert count.
   Sparsity control objective: min ||activated_count - k_target||^2
   where k_target is differentiably learned per layer per token.

**F3 Prevention — Overhead:**
Already solved. Router is 0.46% of inference. No further work needed on latency.
The remaining question is: does routing QUALITY degrade at N>25?

### Critical Finding: Our N>25 Scaling Bottleneck

Our softmax router matches oracle at N=24. But angular concentration in high-dimensional
routing spaces (the problem L2R addresses) means that as N grows:

cos(x, w_i) -> cos(x, w_j) for i != j when d_route is fixed and N >> d_route

Formally (L2R, arxiv 2601.21349): In R^d, for N random unit vectors w_i, the maximum
pairwise cosine satisfies:
max_{i!=j} |cos(w_i, w_j)| >= sqrt((N - d) / (d(N-1)))

At d_route = 64 (our hidden dim), N = 100:
max |cos| >= sqrt((100-64)/(64*99)) = sqrt(36/6336) = 0.075

At N = 500:
max |cos| >= sqrt((500-64)/(64*499)) = sqrt(436/31936) = 0.117

This means router embeddings become increasingly indistinguishable. The solution space
for maintaining discriminability at N>25 requires EITHER:
(a) Increasing d_route (memory cost), OR
(b) Structured routing spaces (Grassmannian MoE, L2R's SIPS), OR
(c) Hierarchical routing (cluster first, then discriminate within cluster)


## 1. Taxonomy of Routing Mechanisms

### 1.1 Routing Granularity

| Level | Description | Proven Status |
|-------|-------------|---------------|
| Per-sequence | One expert set for entire input | CORRECT granularity (Finding: per-layer routing KILLED) |
| Per-token | Different experts per token | Works but ~= per-sequence on clean domains (-0.46% diff) |
| Per-layer | Different experts at each transformer layer | KILLED: actively harmful (-0.5% to -1.1%) |

**Mathematical justification for per-sequence:**
Let h_l = h_{l-1} + A_i @ B_i(h_{l-1}) at layer l with expert i.
If i varies by layer (per-layer), h_L depends on a PATH through expert space:
h_L = h_0 + sum_{l=1}^{L} A_{i_l} @ B_{i_l}(h_{l-1})
Each adapter was trained assuming SAME adapter at all layers (residual stream consistency).
Mixing adapters across layers breaks this assumption. Empirically confirmed: -18.3% degradation.

### 1.2 Gate Function Taxonomy

**Softmax (standard MoE):**
g(x) = softmax(W_r x / tau) in R^N
Properties: sum g_i = 1 (zero-sum competition), differentiable, N params per hidden dim
Failure mode: winner-take-all collapse at low tau
Our status: matches oracle at N=24, but 44% worse than Gumbel-sigmoid

**Gumbel-Sigmoid (L2R-style, our proven mechanism):**
g_i(x) = sigma((w_i^T x + b_i + epsilon_i) / tau)
Properties: gates are INDEPENDENT Bernoulli, no zero-sum, allows multi-activation
Our status: 44% better than softmax, 0.58% overhead, diversity 2.42 experts/sequence

**SIPS — Saturated Inner-Product Scoring (L2R, arxiv 2601.21349):**
g_i(x) = min(max(w_i^T phi(x), -C), C)
where phi: R^d -> R^r is a learned low-rank projection (r << d), C > 0 is saturation bound.
Properties: Lipschitz-controlled (||g||_Lip <= C/||phi||), prevents angular concentration,
router scores bounded in [-C, C] preventing scale explosion.
Key insight: saturation at +/-C prevents any single expert from dominating via score magnitude.

**Hash (deterministic, zero-parameter):**
g(x) = e_{hash(x) mod N}
Properties: O(0) trainable params, perfect balance, but no learned specialization
Our status: proven at N=20, 5.3% displacement. Not competitive with learned routing.

**Matrix Bingham on Grassmannian (arxiv 2602.17798):**
g(x) = argmax_i tr(Lambda_i U_i^T phi(x) phi(x)^T U_i)
where U_i in Gr(r, d) are expert subspaces on the Grassmann manifold.
Properties: Concentration parameter Lambda gives exponential collapse bound.
THIS ALIGNS PERFECTLY with our Grassmannian skeleton (frozen A_i matrices).


## 2. Detailed Mechanism Analysis

### 2.1 CoMoL: Core-Space Merging (arxiv 2603.00573)

**What it computes:**
Standard MoLoRA: y = x + sum_{i in S} g_i * (A_i @ B_i)(x), cost O(|S| * d * r)
CoMoL factorization: A_i = A_shared, B_i = B_shared @ C_i where C_i in R^{r x r}
So: y = x + A_shared @ (sum_{i in S} g_i * B_shared @ C_i)(x)
     = x + A_shared @ B_shared @ (sum_{i in S} g_i * C_i) @ x_projected

The "core space" is the set {C_i in R^{r x r}}: routing and merging happen in r x r
instead of d x r. Cost: O(d * r + |S| * r^2) instead of O(|S| * d * r).

**Interaction with our architecture:**
INCOMPATIBLE. Our experts have DISTINCT orthogonal A_i matrices (Grassmannian skeleton).
CoMoL requires A_shared. If we forced A_shared, we lose the Grassmannian orthogonality
guarantee that makes composition interference-free.

However, a MODIFIED CoMoL could work: if we group experts by Grassmannian proximity
(clusters of nearby A_i), each cluster shares an A_cluster, and within-cluster routing
uses core-space merging. This is hierarchical routing + core-space hybrid.

**Memory at N=100:**
Standard: N * d * r = 100 * 2560 * 16 = 4.1M params (16.4 MB at fp32)
CoMoL: d * r + N * r^2 = 2560 * 16 + 100 * 256 = 41K + 25.6K = 66.6K params (0.27 MB)
Reduction: 61x at N=100. This is THE key advantage for N>25 scaling on memory-constrained hardware.

### 2.2 LD-MoLE: Learnable Dynamic Routing (arxiv 2509.25684)

**What it computes:**
For token x at layer l:
1. Compute routing scores: s_i = f(x, l) for each expert i
2. Differentiable thresholding: activated = {i : s_i > theta(x, l)}
   where theta(x, l) is a learned threshold (not fixed, not Otsu)
3. Apply: y = x + sum_{i in activated} s_i * Expert_i(x)

**Key innovation:** Dynamic k (number of activated experts) per token per layer.
Subsumes both top-k and entropy gating into a single differentiable framework.

**Comparison with our entropy gating:**
Our entropy gating: skip ALL experts if base entropy < Otsu threshold. Binary: 0 or all experts.
LD-MoLE: skip INDIVIDUAL experts based on learned relevance. Granular: 0 to N experts per token.

| Property | Entropy Gating (ours) | LD-MoLE |
|----------|-----------------------|---------|
| Threshold | Post-hoc Otsu (fixed after calibration) | Learned during training |
| Granularity | All-or-nothing (0 or k experts) | Per-expert (0 to N) |
| Training | No training needed | Requires end-to-end training |
| Adaptivity | Fixed after calibration | Adapts per input |
| Skip rate | 63% (proven) | Not specified; analytical sparsity control |

**For our architecture:** LD-MoLE's per-layer routing is harmful (proven finding).
But its per-TOKEN dynamic-k mechanism, applied at per-sequence granularity, could
upgrade entropy gating from binary to continuous: instead of "skip all" or "use top-k",
learn "use 0, 1, 2, ..., k experts for this sequence."

### 2.3 L2R: Low-Rank Lipschitz-Controlled Routing (arxiv 2601.21349)

**What it computes:**
1. Project input to shared latent space: z = phi(x) = W_proj @ x, W_proj in R^{r_route x d}
2. Score each expert via SIPS: s_i = clamp(w_i^T z, -C, C)
3. Apply Gumbel noise for exploration: s_i' = s_i + epsilon_i, epsilon ~ Gumbel(0, tau)
4. Select top-k or apply sigmoid independently

**Why SIPS matters at scale:**
Standard inner product: s_i = w_i^T x. As d grows, ||s_i|| ~ sqrt(d) by CLT.
Different experts' scores concentrate around the same value (angular concentration).
SIPS: clamping at C bounds the dynamic range, preventing any expert from dominating
through sheer scale. The Lipschitz bound ||g||_Lip <= C * ||W_proj|| controls sensitivity.

**Interaction with Gumbel-sigmoid:**
L2R and Gumbel-sigmoid are COMPLEMENTARY, not competing:
- Gumbel noise: exploration mechanism (solves discrete selection differentiability)
- SIPS: score normalization (solves angular concentration at high N)
- Combined: g_i(x) = sigma(clamp(w_i^T W_proj x, -C, C) + epsilon_i) / tau)

This is our R1 recommendation: add SIPS to existing Gumbel-sigmoid for N>25 scaling.

**Complexity:** O(d * r_route + N * r_route) per token. With r_route = 16, d = 2560, N = 100:
O(40,960 + 1,600) = O(42,560) FLOPs. Negligible vs expert computation O(2 * d * r) = O(81,920).


## 3. Metal/Apple Silicon Compatibility Analysis

### 3.1 What Makes an Operation Metal-Friendly

Metal Compute Shaders on Apple Silicon prefer:
1. **Dense matrix multiplies** — fully utilized ALUs, vectorized across SIMD groups
2. **Uniform control flow** — no warp divergence (all threads take same branch)
3. **Coalesced memory access** — sequential reads from unified memory
4. **Batch-parallelizable** — same operation across batch dimension

Metal-hostile operations:
1. **Sparse/irregular access** — gather/scatter kills bandwidth utilization
2. **Sequential decisions** — if-then-else chains based on per-token routing
3. **Small matrix multiplies** — cannot fill SIMD groups (32-wide on Apple GPU)
4. **Synchronization barriers** — cross-threadgroup sync is expensive

### 3.2 Mechanism Metal Compatibility

| Mechanism | Core Operation | Metal-Friendly? | Notes |
|-----------|---------------|-----------------|-------|
| Softmax router | matmul(h, W_r) + softmax | YES | Single dense matmul, softmax is elementwise |
| Gumbel-sigmoid | matmul(h, W_r) + sigmoid | YES | Same as softmax but independent gates |
| SIPS (L2R) | matmul(h, W_proj) + clamp + matmul | YES | Two dense matmuls, clamp is elementwise |
| Hash routing | hash function lookup | YES | Trivially parallel, O(1) per token |
| CoMoL core merge | matmul in r x r space | YES | Small dense matmul, highly parallel |
| Grassmannian Bingham | trace computation on Gr(r,d) | MODERATE | Requires SVD or eigendecomposition per token |
| LD-MoLE dynamic-k | differentiable threshold | MODERATE | Variable work per token (load imbalance) |
| X-LoRA per-layer | per-layer gate recomputation | NO | Sequential layer dependency, L forward passes |
| SpectR spectral | SVD of expert weights at runtime | NO | SVD is iterative, not Metal-friendly |
| PHATGOOSE per-layer | per-layer per-token gates | NO | Same as X-LoRA — per-layer is harmful anyway |
| VectorDB retrieval | k-NN search in embedding space | NO | Irregular memory access, sequential |

### 3.3 Memory Budget at Scale

On M5 Pro 48GB (40GB usable after OS):
- Base model (BitNet-2B-4T): ~1.18 GB
- Per adapter (rank-16, ternary): ~1.9 KB weights + 45.2 MB runtime buffer
- Router head: ~82 KB

| N (experts) | Adapter Storage | Runtime Buffer | Router | Total |
|-------------|-----------------|----------------|--------|-------|
| 25 | 47.5 KB | 1.13 GB | 82 KB | 2.39 GB |
| 50 | 95 KB | 2.26 GB | 164 KB | 3.52 GB |
| 100 | 190 KB | 4.52 GB | 328 KB | 5.88 GB |
| 500 | 950 KB | 22.6 GB | 1.6 MB | 23.96 GB |
| 853 (max) | 1.6 MB | 38.6 GB | 2.7 MB | 39.96 GB |

Router memory is negligible even at N=853. The bottleneck is expert runtime buffers.


## 4. Complexity Comparison (Per-Token, at Routing Time)

Let d = model hidden dim, r = LoRA rank, N = number of experts, k = selected experts,
r_route = routing projection dim.

| Mechanism | FLOPs | Params | Memory | Metal? |
|-----------|-------|--------|--------|--------|
| Softmax router | O(d * N) | d * N | O(N) | YES |
| Gumbel-sigmoid | O(d * N) | d * N | O(N) | YES |
| SIPS (L2R) | O(d * r_route + N * r_route) | d * r_route + N * r_route | O(r_route + N) | YES |
| Hash | O(1) | 0 | O(1) | YES |
| CoMoL core | O(d * r + k * r^2) | d * r + N * r^2 | O(r^2) | YES |
| Grassmannian Bingham | O(N * r * d) | N * r * d | O(N * r * d) | MODERATE |
| LD-MoLE | O(d * N + N) | d * N + N | O(N) | MODERATE |

For d=2560, r=16, N=100, k=2, r_route=16:
- Softmax: 256,000 FLOPs
- Gumbel-sigmoid: 256,000 FLOPs
- SIPS: 40,960 + 1,600 = 42,560 FLOPs (6x fewer than softmax at N=100)
- CoMoL: 40,960 + 512 = 41,472 FLOPs
- Hash: ~10 FLOPs

Key insight: L2R's SIPS reduces routing FLOPs from O(d*N) to O(d*r_route + N*r_route).
At N=100, this is 6x cheaper. At N=500, it's 30x cheaper. This is the scaling advantage.


## 5. Worked Example: SIPS + Gumbel-Sigmoid at d=64, N=8, r_route=8

Given: x in R^64 (hidden state), 8 experts, projection dim 8.

Step 1: Project to latent space
W_proj in R^{8 x 64} (learned)
z = W_proj @ x in R^8
Cost: 64 * 8 = 512 FLOPs

Step 2: Score each expert via SIPS
w_i in R^8 for i = 1..8 (expert embeddings in latent space)
s_i = clamp(w_i^T z, -C, C) where C = 3.0
Cost: 8 * 8 = 64 FLOPs

Step 3: Add Gumbel noise
s_i' = s_i + Gumbel(0, 1)
Cost: 8 random samples + 8 adds = 16 FLOPs

Step 4: Apply sigmoid independently
g_i = sigma(s_i' / tau) where tau = 0.5
Cost: 8 sigmoid evaluations = ~40 FLOPs

Total routing cost: 512 + 64 + 16 + 40 = 632 FLOPs
Expert forward pass (one adapter): 2 * 64 * 4 = 512 FLOPs (h @ A @ B)
Routing/expert ratio: 632 / (2 * 512) = 0.62 at k=2

At d=2560, r=16, N=100, r_route=16:
Routing: 2560*16 + 100*16 + 100 + 100*5 = 40,960 + 1,600 + 600 = 43,160
Expert (k=2): 2 * 2 * 2560 * 16 = 163,840
Ratio: 43,160 / 163,840 = 26.3% — still acceptable, expert computation dominates.

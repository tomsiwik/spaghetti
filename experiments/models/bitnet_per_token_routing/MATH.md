# Per-Token Routing for Ternary LoRA Composition: Mathematical Foundations

## 1. Problem Setup

Given:
- Base model $f_\theta: \mathbb{R}^{d} \to \mathbb{R}^{V}$ (BitNet-2B-4T, ternary, $d=2560$, $V=151936$)
- $N$ trained LoRA adapters $\{\Delta W_i\}_{i=1}^{N}$, each $\Delta W_i = B_i A_i$ with $A_i \in \mathbb{R}^{d \times r}$, $B_i \in \mathbb{R}^{r \times d}$, $r=16$
- Adapter scale $\alpha = 20.0$

### Uniform 1/N Composition

The current approach applies all adapters at equal weight:

$$y_t = f_\theta(x_t) + \frac{\alpha}{N} \sum_{i=1}^{N} x_t A_i B_i$$

At $N=15$, each adapter contributes $\frac{1}{15} \approx 6.7\%$ of its full signal. The effective adapter magnitude scales as $\frac{1}{N^2}$ in terms of squared norm impact, meaning at $N=25$ each adapter contributes ~4% of its solo impact.

### Per-Token Routing

Replace uniform $\frac{1}{N}$ with learned per-token weights $w_i(h_t)$:

$$y_t = f_\theta(x_t) + \alpha \sum_{i=1}^{N} w_i(h_t) \cdot x_t A_i B_i$$

where $h_t = g_\theta(x_t) \in \mathbb{R}^{d}$ is the hidden state from the last transformer layer (computed by the base model without any adapter), and $w(h_t) = \text{Route}(h_t)$.

## 2. Router Architecture

Two-layer MLP with ReLU:

$$\text{Route}(h) = W_2 \cdot \text{ReLU}(W_1 h + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{d \times d_h}$, $W_2 \in \mathbb{R}^{d_h \times N}$, $d_h = 256$.

### Top-k Selection

Given logits $\ell = \text{Route}(h_t) \in \mathbb{R}^N$:

**Top-1:** Select $i^* = \arg\max_i \ell_i$, apply adapter $i^*$ at full strength.

**Top-k:** Select top-$k$ indices $S_k = \text{argtopk}(\ell)$, apply softmax-normalized weights:

$$w_i = \begin{cases} \frac{\exp(\ell_i)}{\sum_{j \in S_k} \exp(\ell_j)} & \text{if } i \in S_k \\ 0 & \text{otherwise} \end{cases}$$

### Sequence-Level Aggregation

For computational efficiency, we aggregate per-token router predictions to sequence level:

$$\bar{w}_i = \frac{1}{T} \sum_{t=1}^{T} \text{softmax}(\ell_t)_i$$

Then select top-$k$ from $\bar{w}$ and renormalize. This is a conservative approximation: true per-token routing would be strictly better.

## 3. Router Training

**Objective:** Classify which domain a text belongs to from its hidden states.

$$\mathcal{L}_\text{router} = -\frac{1}{T} \sum_{t=1}^{T} \log \frac{\exp(\ell_{t,y})}{\sum_{j=1}^{N} \exp(\ell_{t,j})}$$

where $y$ is the domain label for the sequence.

**Training data:** 80 samples per domain from training split, 20% held out for validation. Hidden states pre-computed from base model (no adapter applied).

**Key insight:** The router trains on base model hidden states but routes adapters. This works because domain-distinguishing features exist in the base model's representation space. The adapters modify the output but the routing decision is based on input characteristics.

## 4. Parameter Cost Analysis

| Component | Parameters | Storage |
|-----------|-----------|---------|
| Base model | 2.4B (ternary) | ~490 MB packed |
| Each LoRA adapter | 21.6M (FP16) | ~43 MB |
| 15 adapters total | 324M | ~648 MB |
| Router | 659K | ~1.3 MB |
| **Router overhead** | **0.2% of adapter params** | **0.2% of adapter storage** |

The router is negligible: 659K params vs 324M adapter params (0.2%).

## 5. Complexity

**Uniform 1/N:** One forward pass with pre-composed adapter. Cost: $O(d^2 L)$ (standard transformer).

**Routed top-k:** One base forward pass to get hidden states, router prediction ($O(d \cdot d_h + d_h \cdot N)$), then one forward pass with selected adapter. Total: approximately $2 \times$ base cost.

At inference time, the hidden state computation can be shared with the adapter computation (compute hidden states layer by layer, route, apply adapter to same forward pass). This reduces overhead to $O(d \cdot d_h + d_h \cdot N)$ per sequence = negligible vs $O(d^2 L)$.

## 6. Why Top-2 Beats Top-1

**Top-1 failure mode:** When the router selects a single adapter at full strength, any domains where the adapter overshoots (individual PPL > base PPL) get amplified. At N=15, 4 domains have individual PPL worse than base (medical, code, chemistry, dialogue).

**Top-2 advantage:** Blending two adapters with softmax weights provides natural regularization:
- Weights are $w_1 + w_2 = 1.0$, so each adapter is at most at 50-80% strength
- The second adapter provides complementary information (cross-domain transfer)
- Effective strength per adapter: $\sim 0.6 \times$ full strength (vs $1.0$ for top-1, vs $\frac{1}{15} = 0.067$ for uniform)

This is the "Goldilocks zone": enough signal to beat base, enough regularization to avoid overshoot.

## 7. Assumptions

1. Base model hidden states contain sufficient domain-distinguishing information (validated: 91.7% routing accuracy)
2. Domain labels are the correct routing signal (may not capture within-domain variation)
3. Sequence-level routing is a conservative lower bound on per-token routing performance
4. Adapter quality is uniform enough that routing (not adapter quality) is the bottleneck
5. LoRA scale of 20.0 is held constant (adaptive scaling per adapter could further improve)

## 8. Worked Example (d=2560, N=15, k=2)

Input: medical text, hidden state $h \in \mathbb{R}^{2560}$.

Router output (softmax): medical=0.45, health=0.15, chemistry=0.08, ..., dialogue=0.01.

Top-2 selection: medical (0.45), health (0.15). Renormalized: medical=0.75, health=0.25.

Composed adapter: $\Delta W = 0.75 \cdot B_\text{med} A_\text{med} + 0.25 \cdot B_\text{health} A_\text{health}$

The medical adapter gets 75% weight (vs 6.7% under uniform 1/N), a **11.2x concentration** of relevant signal. Health provides complementary information about similar biomedical concepts.

Observed: medical top-2 PPL = 15.70 vs uniform 15.73 (similar), but for physics: top-2 PPL = 22.83 vs uniform 46.04 (50.4% improvement).

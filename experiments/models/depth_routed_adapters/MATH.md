# Depth-Routed Adapters: Mathematical Foundations

## Problem Statement

Given a base transformer $f_\theta$ with $L$ layers, $N$ domain LoRA adapters
$\{\Delta W_i = B_i A_i^T\}_{i=1}^N$, and layer indices $l \in \{1, \ldots, L\}$:

Does adding a second routing axis (per-layer via AttnRes pseudo-queries) over the
existing per-token axis (Gumbel-sigmoid) produce measurably better composition than
per-token routing alone?

The claim: token routing selects *which experts are relevant to this input*, but
leaves all selected experts contributing equally at every layer. Layer routing
selects *which layers each expert is most effective in*, allowing early/middle/late
layer specialization. The two axes are orthogonal in function and jointly necessary
for fine-grained composition.

---

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| $x$ | $(T, d)$ | Input token sequence, $T$ tokens, $d$-dimensional |
| $h_l$ | $(T, d)$ | Hidden state at layer $l$ |
| $h_0$ | $(T, d)$ | Initial hidden state (embedding output) |
| $W_l$ | $(d, d)$ | Frozen base weight matrix at layer $l$ |
| $A_i$ | $(d, r)$ | Frozen Grassmannian projection for adapter $i$; $A_i^T A_i = I_r$ |
| $B_i^l$ | $(r, d)$ | Trained B-matrix for adapter $i$ at layer $l$ |
| $\alpha_{i,l}$ | $[0, 1]$, $\sum_i = 1$ | Layer-routing weight: how much adapter $i$ contributes at layer $l$ |
| $g_i$ | $\{0, 1\}$ | Token-routing gate: is adapter $i$ active for this input? |
| $r_i$ | $(d_r,)$ | Learned expert embedding for adapter $i$; $d_r \ll d$ |
| $w_l$ | $(d_r,)$ | Learned layer query for layer $l$ |
| $q_l$ | $(d_r,)$ | AttnRes pseudo-query for layer $l$ (from depth attention over $h_0 \ldots h_{l-1}$) |
| $\phi_l$ | $(d, d_r)$ | Linear projection from hidden state to query space |
| $d$ | $128$ | Hidden dimension (micro model) |
| $d_r$ | $32$ | Expert embedding dimension |
| $r$ | $8$ | LoRA rank (micro model) |
| $L$ | $4$ | Number of transformer layers |
| $N$ | $5$ | Number of domain adapters |

---

## Mechanism: Two-Axis Routing

### Axis 1 — Token Routing (Gumbel-Sigmoid)

For each input $x$, a shared router network $R_\phi: (T, d) \to \mathbb{R}^N$
produces per-adapter logits. During training, discrete gates are approximated via
Gumbel-Sigmoid:

$$g_i = \sigma\!\left(\frac{\log u_i - \log(1-u_i) + s_i}{\tau}\right), \quad u_i \sim \text{Uniform}(0,1)$$

where $s_i = R_\phi(x)_i$ is the router logit for adapter $i$, and $\tau$ is
temperature (annealed from 1.0 to 0.1 during training).

At inference, the gate is hard-thresholded: $g_i = \mathbf{1}[s_i > 0]$.

**What this selects:** Which adapters are semantically relevant to the current
token sequence. This is the coarse gating step validated in exp_softmax_router_scaling
(oracle-matching at N=24, gamma=0.625).

### Axis 2 — Layer Routing (AttnRes Pseudo-Queries)

At each layer $l$, a depth-attention mechanism computes a pseudo-query from the
full history of prior hidden states $\{h_0, h_1, \ldots, h_{l-1}\}$:

$$q_l = \text{softmax}\!\left(\frac{(\phi_l h_{l-1})^T H_{<l}}{\sqrt{d_r}}\right) H_{<l}^T \in \mathbb{R}^{d_r}$$

where $H_{<l} = [\phi_1 h_0 \;\|\; \phi_2 h_1 \;\|\; \cdots \;\|\; \phi_{l-1} h_{l-2}] \in \mathbb{R}^{d_r \times (l-1)}$
stacks projected representations of all prior layers (mean-pooled over tokens).

The layer-routing weight for adapter $i$ at layer $l$ is then:

$$\alpha_{i,l} = \frac{\exp(q_l^T r_i)}{\sum_{j=1}^N \exp(q_l^T r_j)}$$

**What this selects:** Which adapters are most effective at this specific depth,
conditioned on how much residual information has accumulated. This is the fine-grained
axis that token routing cannot capture.

### Combined Forward Pass

Standard uniform composition (no routing):

$$h_l = h_{l-1} + W_l h_{l-1} + \frac{\alpha_{\text{scale}}}{N} \sum_{i=1}^N B_i^l A_i^T h_{l-1}$$

Depth-routed composition:

$$h_l = h_{l-1} + W_l h_{l-1} + \alpha_{\text{scale}} \sum_{i=1}^N g_i \cdot \alpha_{i,l} \cdot B_i^l A_i^T h_{l-1}$$

where $\alpha_{\text{scale}} = \alpha_{\text{lora}} / r$ is the LoRA scaling factor
and $g_i \in \{0,1\}$ is the token-level gate from Axis 1.

The full tensor mechanics at layer $l$:
1. $z_i = A_i^T h_{l-1}$ — project to adapter subspace: $(r,)$ per token
2. $\delta_i^l = B_i^l z_i$ — adapter contribution: $(d,)$ per token
3. $\tilde{\delta}_i^l = g_i \cdot \alpha_{i,l} \cdot \delta_i^l$ — gate and weight: $(d,)$
4. $h_l = h_{l-1} + W_l h_{l-1} + \alpha_{\text{scale}} \sum_i \tilde{\delta}_i^l$

---

## Why It Works: Derivation

### Theorem 1 — Layer Specialization Under Grassmannian Orthogonality

**Setup.** The Grassmannian skeleton fixes $A_i$ such that $A_i^T A_j = 0$ for
$i \neq j$ (at micro scale: $r=8$, $d=128$, so $N_{\max} = d/r = 16$ orthogonal adapters).
The projection of any hidden state $h$ onto adapter $i$'s subspace is
$z_i = A_i^T h \in \mathbb{R}^r$.

**Claim.** If adapter $i$ was trained on domain $d_i$ with training distribution
$p_{d_i}$, then the *effective rank* of $B_i^l$ at layer $l$ decreases in layers
where domain-relevant features have already been collapsed by earlier processing.

**Proof sketch.** Let $\mathcal{F}_l = \text{span}(\{h_l(x) : x \sim p_{d_i}\})$
be the feature space at layer $l$ for adapter $i$'s domain. Standard residual
accumulation causes:

$$\mathcal{F}_l \subseteq \mathcal{F}_{l-1} + W_l \mathcal{F}_{l-1}$$

As $l$ increases, features become increasingly mixed (residual accumulation adds
contributions from all domains, diluting the per-domain signal). By layer $L$,
$\mathcal{F}_L$ may have low projection onto adapter $i$'s subspace $\text{col}(A_i)$.

Therefore, $\|z_i^l\|_2 = \|A_i^T h_l\|_2$ is maximal at the layers where domain $d_i$
features are most cleanly separable — typically early layers for surface features
(syntax, vocabulary) and middle layers for semantic features (topic, register).

The layer-routing weight $\alpha_{i,l} \propto \exp(q_l^T r_i)$ will learn to be
large precisely when $q_l$ (the depth-attention summary of processing history)
encodes that layer $l$'s features are still domain-discriminative. This is why
depth weights should be non-uniform: they track the decay of per-domain signal
across depth. $\square$

### Theorem 2 — Gumbel-Sigmoid vs Softmax for Token Routing

**Why Gumbel and not softmax?** The softmax router (exp_softmax_router_scaling)
applies to static routing at inference: we know the input and select the best
single adapter. For *training* a compositional gate, we need:
1. Discrete-like behavior at inference (gate is 0 or 1, not a fractional weight)
2. Differentiable gradients during training

The Gumbel-Sigmoid achieves this via the reparameterization trick. During training:

$$\nabla_{s_i} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial g_i} \cdot g_i(1-g_i)/\tau$$

As $\tau \to 0$, the gradient becomes concentrated near the threshold ($g_i \approx 0.5$),
providing strong learning signal for the decision boundary. At $\tau \to \infty$,
the gate becomes a smooth sigmoid (no discrete structure).

**Why not softmax for layer routing?** Layer routing uses softmax over adapters
(not Gumbel) because:
- Layer routing is *always compositional*: we want multiple adapters contributing
  at each layer, just with different weights. This is not a hard selection problem.
- Gumbel would zero out adapters at some layers entirely, losing the smoothness
  that lets the depth-routing weights vary continuously across $l$.
- Softmax guarantees $\sum_i \alpha_{i,l} = 1$: the total adapter contribution
  is constant across layers, only the distribution changes.

### Why AttnRes Pseudo-Queries Track Feature Decay

The AttnRes mechanism (arXiv 2603.15031) computes depth attention over all prior
hidden states. The pseudo-query $q_l$ at layer $l$ is a weighted average of prior
layer representations, with weights determined by how relevant each prior layer's
state is to the current position.

Crucially, $q_l$ encodes *how much processing has occurred* before layer $l$,
not just the current state. If layers 1-2 have already collapsed domain-specific
features (high entropy in $A_i^T h_l$), $q_l$ will reflect this through the
attention weights over prior states. The learned expert embeddings $r_i$ can then
distinguish "domain $i$'s features are still available at this depth" vs "domain
$i$'s features have been diluted."

This is the key mechanistic claim: **depth attention encodes processing history;
expert embeddings encode domain-feature decay rates; their dot product
$q_l^T r_i$ is a learned proxy for "how much does adapter $i$ still have to
contribute at depth $l$?"**

---

## Assumptions

1. **Grassmannian orthogonality holds.** $A_i^T A_j \approx 0$ for all $i \neq j$.
   Validated: cosine $|$cos$| = 0.00125$ at convergence, 40x below 0.05 threshold.
   This is required for the interference bound in Theorem 1 — without it, $z_i^l$
   is contaminated by other domains' projections.

2. **Domain-specific features are layer-localized.** Different domains activate
   different layers preferentially. This is assumed by analogy to the depth-utilization
   finding (Csordás et al., OpenReview 2025) that LLMs underutilize deep layers,
   and to MoDA's (arXiv 2603.15619) empirical result that shallow features degrade
   gradually under residual accumulation.

3. **L=4 provides sufficient depth for specialization.** This is the weakest
   assumption. exp_attnres_depth_composition found entropy ratio 0.775 (non-uniform
   depth weights) but only 0.39% composition improvement at L=4. The assumption is
   that adding per-layer *routing* (not just AttnRes residual reweighting) can
   amplify the effect even at shallow depth by directly modulating which adapters
   contribute at each layer.

4. **Expert embeddings $r_i \in \mathbb{R}^{d_r}$ are learnable during composition.**
   $d_r = 32$ is large enough to distinguish $N=5$ domains ($32 \gg \log_2 5 = 2.32$
   bits needed) but small enough that $N \cdot d_r = 160$ parameters does not
   overfit on short training sequences.

5. **Token routing and layer routing are independent.** $g_i$ is computed from the
   full input $x$ (global); $\alpha_{i,l}$ is computed from depth history (per-layer).
   They are multiplied as independent factors. This independence holds only if the
   "is this adapter relevant?" decision and the "is this layer the right depth for
   this adapter?" decision do not interact — a reasonable assumption since token
   routing is input-semantic and layer routing is architecture-structural.

---

## What Breaks It: Failure Conditions

### Failure 1 — Depth Collapse at L=4 (Primary Risk)

exp_attnres_depth_composition confirmed that L=4 is marginally too shallow for
depth mechanisms to produce large effects. The adapter norm distribution at L=4
under standard residuals is $[1.30, 1.86, 2.29, 3.36]$ — the gradient already
increases with depth, but not by enough to create strong layer specialization.

**Condition:** If the per-layer softmax weights $\alpha_{i,l}$ converge to nearly
uniform ($\alpha_{i,l} \approx 1/N$ for all $i, l$), layer routing degenerates to
uniform composition. The entropy of $\alpha_{\cdot,l}$ must satisfy:

$$H(\alpha_{\cdot,l}) < \log N \cdot \theta_{\text{spec}}$$

where $\theta_{\text{spec}} < 1$ indicates specialization. If $H(\alpha_{\cdot,l}) \approx \log N$
(maximum entropy, uniform) for all $l$, K1 FAILS (no layer specialization).

**Mitigation:** Expert embeddings $r_i$ are initialized with small random variance
($\sigma = 0.01$) to break symmetry. If all $r_i$ start identical, $\alpha_{i,l} = 1/N$
forever — the gradient through softmax is zero when inputs are identical.

### Failure 2 — Grassmannian Gradient Contamination

The Grassmannian A matrices are frozen. Under standard residual accumulation,
the gradient through A is:

$$\frac{\partial \mathcal{L}}{\partial A_i} = h_{l-1}^T \frac{\partial \mathcal{L}}{\partial z_i^l} \cdot B_i^l$$

which is blocked (frozen). Under depth routing, the effective scaling of this
gradient path by $\alpha_{i,l}$ could starve low-attention layers of gradient
signal for $B_i^l$. Specifically, if $\alpha_{i,l} \approx 0$ for layer $l$,
then $B_i^l$ receives no gradient from that layer. The risk: B matrices at
underweighted layers converge to zero, effectively making those layers dead.

**Condition:** If $\exists i, l$ such that $\alpha_{i,l} < \epsilon$ for all training
steps, then $B_i^l \to 0$ and the rank of the adapter at that layer collapses.

**Detection:** Track $\|B_i^l\|_F$ across training. If any layer's B norm drops
below 10% of initialization, the routing has degenerated.

### Failure 3 — Expert Embedding Collapse

If learned expert embeddings $r_i$ converge to the same vector for all $i$, then
$\alpha_{i,l} = 1/N$ regardless of $q_l$. This is a degenerate fixed point: the
softmax layer-routing collapses to uniform, providing no signal.

**Condition:** $\|r_i - r_j\|_2 < \delta$ for all $i \neq j$, where $\delta$ is
below the resolution needed to distinguish domain signatures.

**Lower bound needed:** For $N=5$ adapters in $\mathbb{R}^{d_r=32}$, the expected
pairwise distance if embeddings are iid Gaussian with $\sigma=1$ is
$\sqrt{2d_r} \approx 8.0$. We need $\|r_i - r_j\|_2 > 1.0$ to have meaningful
routing diversity.

**Mitigation:** Track pairwise $r_i$ distances during training. An auxiliary
diversity regularization term can prevent collapse:

$$\mathcal{L}_{\text{div}} = -\lambda \cdot \frac{1}{\binom{N}{2}} \sum_{i < j} \|r_i - r_j\|_2^2$$

### Failure 4 — AttnRes Attention Over Insufficient History

At layer $l=1$, the depth-attention history has only $h_0$ (the embedding). The
pseudo-query $q_1 = \phi_1 h_0$, providing no useful depth information — all
layers at $l=1$ have identical depth context regardless of domain. This means
$\alpha_{i,1}$ depends only on the dot product of $\phi_1 h_0$ with expert
embeddings $r_i$, which collapses to a token-routing-like decision, not a
depth-routing decision.

**Effect:** Layer 1 routing is degenerate (not a depth function). At L=4, this
means 25% of layers produce non-informative depth weights. This is fundamental
and cannot be mitigated without increasing L.

---

## Complexity Analysis

### Parameter Count

| Component | Parameters | Formula |
|-----------|-----------|---------|
| B matrices (all adapters, all layers) | $N \cdot L \cdot r \cdot d$ | $5 \times 4 \times 8 \times 128 = 20{,}480$ |
| Expert embeddings $r_i$ | $N \cdot d_r$ | $5 \times 32 = 160$ |
| Layer queries $w_l$ | $L \cdot d_r$ | $4 \times 32 = 128$ |
| Projection matrices $\phi_l$ | $L \cdot d \cdot d_r$ | $4 \times 128 \times 32 = 16{,}384$ |
| Token router $R_\phi$ | MLP: $d \to 64 \to N$ | $128 \times 64 + 64 \times 5 = 8{,}512$ |
| **Total trainable** | **45,664** | |
| A matrices (frozen) | $N \cdot d \cdot r$ | $5 \times 128 \times 8 = 5{,}120$ (frozen) |

For comparison: token-only routing (Axis 1 alone) uses $N \cdot L \cdot r \cdot d + R_\phi = 28{,}992$ params.
Depth routing adds $45{,}664 - 28{,}992 = 16{,}672$ params for the two routing axes (~57% overhead).

### FLOPs Per Forward Pass

**Adapter contribution at each layer $l$:**

1. Projection $z_i^l = A_i^T h_{l-1}$: $(T \times d \times r)$ FLOPs per adapter
   $= T \times 128 \times 8 = 1024T$ FLOPs, times $N=5$ = $5120T$
2. Adapter output $\delta_i^l = B_i^l z_i^l$: $(T \times r \times d)$ FLOPs per adapter
   $= T \times 8 \times 128 = 1024T$ FLOPs, times $N=5$ = $5120T$
3. Layer-routing softmax: $(N \times d_r) = 160$ FLOPs per layer (negligible)
4. Weighted sum: $(N \times T \times d) = 5 \times T \times 128 = 640T$ FLOPs

Total adapter FLOPs per layer: $\approx 10{,}880T$. Across $L=4$ layers: $43{,}520T$.

**AttnRes pseudo-query computation at each layer $l$:**

1. Project current state: $\phi_l h_{l-1}$: $(T \times d \times d_r) = T \times 128 \times 32 = 4096T$
2. Attention over $l-1$ prior states: $(d_r \times (l-1) + T \times (l-1))$ FLOPs
   At $l=4$: $32 \times 3 + T \times 3 \approx 3T$ (small; prior history is mean-pooled)
3. Total per layer: $\approx 4099T$. Across $L=4$ layers: $16{,}396T$

**Total additional FLOPs** (over base model): $\sim 60{,}000T$ per forward pass.
At $T=64$ (micro char-level sequence), this is $\sim 3.84$M FLOPs — negligible on
M5 Pro relative to the base transformer's $\sim 100$M FLOPs per forward pass.

### Memory Budget

| Component | Memory |
|-----------|--------|
| Base micro model (d=128, L=4, bf16) | ~4 MB |
| $N=5$ A matrices, bf16 | $5 \times 128 \times 8 \times 2 = 10{,}240$ bytes ≈ 10 KB |
| $N=5 \times L=4$ B matrices, bf16 | $5 \times 4 \times 8 \times 128 \times 2 = 40{,}960$ bytes ≈ 40 KB |
| Expert embeddings + layer queries | $(160 + 128) \times 2 = 576$ bytes ≈ 1 KB |
| Projection matrices, bf16 | $4 \times 128 \times 32 \times 2 = 32{,}768$ bytes ≈ 32 KB |
| Depth-attention history buffer | $L \times T \times d_r \times 2 = 4 \times 64 \times 32 \times 2 = 16{,}384$ bytes ≈ 16 KB |
| **Total experiment** | **~4.1 MB** |

This is effectively unconstrained on M5 Pro 48GB. The micro model leaves ~47.9 GB
headroom for batch training.

---

## Worked Example: Layer Weight Distribution for Two Adapters

**Setup.** $N=2$ adapters: $A_1$ (math domain), $A_2$ (creative writing). $L=4$
layers, $d_r = 4$ (reduced for clarity). Expert embeddings after training:

$$r_{\text{math}} = [1.2, -0.8, 0.3, 0.1], \quad r_{\text{creative}} = [-0.9, 0.7, -0.4, 0.2]$$

Input: mathematical problem text. After depth attention, pseudo-queries at each
layer (mean-pooled, projected):

$$q_1 = [0.5, -0.3, 0.1, 0.0] \quad \text{(from } h_0 \text{ embedding only)}$$
$$q_2 = [0.8, -0.6, 0.2, -0.1] \quad \text{(from } h_0, h_1 \text{)}$$
$$q_3 = [1.0, -0.8, 0.3, -0.2] \quad \text{(from } h_0, h_1, h_2 \text{; math features accumulating)}$$
$$q_4 = [0.7, -0.5, 0.2, -0.1] \quad \text{(math features beginning to collapse into final repr)}$$

**Layer-routing dot products $q_l^T r_i$:**

Layer 1:
- $q_1^T r_{\text{math}} = 0.5 \times 1.2 + (-0.3)(-0.8) + 0.1(0.3) + 0 = 0.60 + 0.24 + 0.03 = 0.87$
- $q_1^T r_{\text{creative}} = 0.5(-0.9) + (-0.3)(0.7) + 0.1(-0.4) + 0 = -0.45 - 0.21 - 0.04 = -0.70$

$$\alpha_{\text{math},1} = \frac{e^{0.87}}{e^{0.87} + e^{-0.70}} = \frac{2.39}{2.39 + 0.50} = 0.827$$
$$\alpha_{\text{creative},1} = 0.173$$

Layer 3 (strongest math signal):
- $q_3^T r_{\text{math}} = 1.0(1.2) + (-0.8)(-0.8) + 0.3(0.3) + (-0.2)(0.1) = 1.20 + 0.64 + 0.09 - 0.02 = 1.91$
- $q_3^T r_{\text{creative}} = 1.0(-0.9) + (-0.8)(0.7) + 0.3(-0.4) + (-0.2)(0.2) = -0.90 - 0.56 - 0.12 - 0.04 = -1.62$

$$\alpha_{\text{math},3} = \frac{e^{1.91}}{e^{1.91} + e^{-1.62}} = \frac{6.75}{6.75 + 0.20} = 0.972$$
$$\alpha_{\text{creative},3} = 0.028$$

**Interpretation.** Even though the token router gates both adapters (both are
active for a math input under a liberal threshold), the layer router assigns
math adapter 97.2% of the contribution at layer 3 where math features are
strongest. Creative writing adapter is present but contributes only 2.8%, avoiding
interference at the most critical layer. Under uniform composition ($\alpha = 0.5$
each), the creative adapter would add 50% interference at layer 3 where math
features are most concentrated.

**K1 check:** Entropy of $[\alpha_{\text{math},l}, \alpha_{\text{creative},l}]$ across layers $l = 1..4$:
- Layer 1: $H = -(0.827 \log 0.827 + 0.173 \log 0.173) = 0.567$ nats
- Layer 3: $H = -(0.972 \log 0.972 + 0.028 \log 0.028) = 0.133$ nats
- Maximum entropy (uniform): $\log 2 = 0.693$ nats
- Entropy ratio: $H_{\text{avg}} / \log N = (0.567 + 0.133)/2 / 0.693 = 0.505 < 1$

This demonstrates layer specialization (entropy ratio < 1 means non-uniform weights).
For K1 to PASS, we require this entropy ratio to be measurably below 1.0 (target < 0.8,
consistent with exp_attnres_depth_composition's 0.775 baseline).

---

## Connection to SOLE Architecture

The SOLE architecture (Scaffold + Orthogonal Learned Experts) currently uses:
- Grassmannian A matrices (frozen, orthogonal)
- Softmax token router (exp_softmax_router_scaling: oracle-matching, 0% fallback)
- No per-layer routing

Depth-routed adapters extend SOLE in a specific way: **routing weights become a
function of depth, not just input**. The composition equation becomes:

**Current SOLE:**
$$y_l = W_l x + \alpha_{\text{scale}} \cdot w_{\text{domain}} \cdot B_l A^T x$$

where $w_{\text{domain}} \in \{0, 1\}$ is the token router's binary decision.

**Depth-Routed SOLE:**
$$y_l = W_l x + \alpha_{\text{scale}} \sum_i g_i \cdot \alpha_{i,l}(q_l) \cdot B_i^l A_i^T x$$

where $\alpha_{i,l}(q_l)$ is computed dynamically at each layer.

**Integration path to SOLE:** If K2 PASSES (layer-routed composition ≥ 2% better
than token-only), the depth-routing module can be integrated into the SOLE serving
stack as an optional per-layer weight modifier. The inference cost is minimal
(~16K FLOPs per forward pass; see complexity analysis). The B matrices are already
per-layer in the standard LoRA implementation — depth routing only adds the
$\alpha_{i,l}$ multiplier and the AttnRes pseudo-query computation.

**Critical open question (from exp_attnres_depth_composition):** AttnRes introduces
softmax normalization into the gradient path of B matrices. The Grassmannian
interference bound:

$$\|\Delta W_i^T \Delta W_j\| \leq \left(\frac{\alpha}{r}\right)^2 \|B_i\| \cdot \underbrace{\|A_i^T A_j\|}_{\approx 0} \cdot \|B_j\|$$

holds under standard additive residuals where gradients flow through $B_i$ uniformly
across layers. Under depth routing, the effective gradient through $B_i^l$ is scaled
by $\alpha_{i,l}$, which varies per layer. This means layers with $\alpha_{i,l} \approx 0$
receive near-zero gradient, potentially causing $B_i^l \to 0$ at those layers (Failure
Condition 2 above).

If $B_i^l \to 0$ for some $l$, the interference at that layer also goes to zero
(the bound is trivially satisfied). But this is a pyrrhic victory: a zero B matrix
contributes nothing and is equivalent to removing the adapter at that layer. The
experiment must verify that low-$\alpha$ layers retain non-trivial B norms.

---

## Prior Results This Experiment Directly Extends

| Experiment | Finding | How Used Here |
|------------|---------|---------------|
| exp_attnres_depth_composition | AttnRes learns non-uniform depth weights (entropy 0.775); negligible composition gain at L=4 (0.39%) | Establishes mechanism is real; this exp adds per-layer routing on top to amplify the gain |
| exp_softmax_router_scaling | Softmax token router achieves oracle-matching quality (gamma 0.625) at N=24 | Provides the token-routing (Axis 1) baseline; K2 measures whether adding Axis 2 improves beyond this |
| Grassmannian skeleton (VISION.md) | $\|A_i^T A_j\| \approx 0$ at convergence; 17x interference filter | Required assumption for Theorem 1; if violated, depth routing cannot improve composition via the mechanism derived |

**References:**
- AttnRes: arXiv 2603.15031 (Kimi K2 depth attention; entropy mechanism)
- MoDA: arXiv 2603.15619 (multi-axis attention, +2.11% at 1.5B scale)
- MoLoRA: arXiv 2603.15965 (per-token LoRA routing; Gumbel-Sigmoid training)
- Depth utilization: Csordás et al., OpenReview 2025 (deep layers underutilized)
- ResFormer: arXiv 2410.17897 (value residual learning; residual dilution mechanism)

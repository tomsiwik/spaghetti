# Pre-Merge vs Dynamic Routing: Mathematical Foundations

## Setup

Consider a base model with frozen weights $W_{\text{base}}$ (a 4-layer MLP with
weight matrices $W_1^{(l)} \in \mathbb{R}^{d \times d_{ff}}$, $W_2^{(l)} \in \mathbb{R}^{d_{ff} \times d}$
for layers $l = 1, \ldots, L$).

We have $N$ domain experts, each a LoRA adapter with low-rank deltas:

$$\Delta W_i^{(l)} = \frac{\alpha}{r} A_i^{(l)} B_i^{(l)}$$

where $A_i^{(l)} \in \mathbb{R}^{d_{\text{in}} \times r}$, $B_i^{(l)} \in \mathbb{R}^{r \times d_{\text{out}}}$,
$\alpha$ is the LoRA scaling factor, $r$ is the rank.

Dimensions in this experiment: $d = 64$, $d_{ff} = 256$, $L = 4$, $r = 8$, $\alpha = 8$, $V = 32$.

## Strategy 1: Pre-Merge (Averaging)

All $N$ expert deltas are averaged into a single weight modification:

$$W_{\text{merged}}^{(l)} = W_{\text{base}}^{(l)} + \frac{1}{N} \sum_{i=1}^{N} \Delta W_i^{(l)}$$

**Inference cost:** Identical to base model (one forward pass with modified weights).
No router, no expert selection, no runtime LoRA swapping. $O(1)$ in $N$.

**Dilution effect:** Each expert's contribution is scaled by $1/N$. If expert $i$
learned a domain-specific correction $\Delta W_i$, only $\frac{1}{N} \Delta W_i$
is active. For large $N$, this approaches zero.

**Interference bound:** For orthogonal experts ($\cos(\text{vec}(\Delta W_i), \text{vec}(\Delta W_j)) \approx 0$),
the merged weight is an additive combination in orthogonal subspaces. No interference.
The only cost is dilution from $1/N$ scaling.

## Strategy 2: Dynamic Top-k Routing

For each input $x$, select the $k$ most relevant experts via cosine similarity
between the input embedding $e(x) \in \mathbb{R}^d$ and expert centroids
$c_i \in \mathbb{R}^d$ (mean embedding of expert $i$'s training data):

$$\text{score}(x, i) = \frac{e(x) \cdot c_i}{\|e(x)\| \cdot \|c_i\|}$$

Select top-$k$ experts and apply at equal weight $1/k$:

$$W_{\text{dynamic}}^{(l)}(x) = W_{\text{base}}^{(l)} + \frac{1}{k} \sum_{i \in \text{top-}k(x)} \Delta W_i^{(l)}$$

**Inference cost:** $O(N)$ for routing (cosine with all centroids), plus $O(k)$
for LoRA application. With pre-computed centroids and batch routing, this is
cheap but not free.

**Advantage over pre-merge:** Expert $i$ gets full $\frac{1}{k}$ weight (not $\frac{1}{N}$)
when selected, and zero weight when not selected. The specialist signal is $\frac{N}{k}$
times stronger for the routed expert.

## Strategy 3: Oracle Routing

Perfect routing: apply each domain's expert at full strength on that domain's test data.

$$W_{\text{oracle}}^{(l)}(x \in \text{domain}_i) = W_{\text{base}}^{(l)} + \Delta W_i^{(l)}$$

This is the upper bound on quality.

## Quality Gap Analysis

The quality gap between pre-merge and dynamic top-1 is:

$$\text{gap} = \frac{L_{\text{pre-merge}} - L_{\text{top-1}}}{L_{\text{top-1}}} \times 100\%$$

For orthogonal experts with negligible deltas relative to the base weights,
the gap approaches zero regardless of $N$. This is because:

1. $\|\Delta W_i\| \ll \|W_{\text{base}}\|$ (LoRA is a small perturbation)
2. With orthogonality, pre-merge $= \frac{1}{N} \sum_i \Delta W_i$ applies
   the correct expert at $1/N$ strength, plus $(N-1)/N$ of irrelevant (but
   non-interfering) experts
3. If the specialization gap is small (oracle barely beats base), then
   the $1/N$ dilution of a tiny improvement produces a tiny loss

The experiment tests where this analysis breaks down: does specialization
grow strong enough that $1/N$ dilution becomes detectable?

## Experimental Design

- Pre-train base model on mixed data ($T_{\text{pre}} = 2000$ steps)
- Train $N$ LoRA experts on domain-specific data ($T_{\text{LoRA}} = 500$ steps)
- Both A and B matrices are updated during LoRA training
- Evaluate on held-out test data per domain
- Vary $N \in \{5, 8, 12, 16, 20\}$
- 3 random seeds for robustness

## Computational Cost

| Component | FLOPs per token |
|-----------|----------------|
| Base forward | $2 \cdot L \cdot (d \cdot d_{ff} + d_{ff} \cdot d)$ |
| Pre-merge forward | Same as base (weights pre-computed) |
| Dynamic top-1 routing | $O(N \cdot d)$ for cosine similarity |
| Dynamic top-1 LoRA apply | $2 \cdot L \cdot (d \cdot r + r \cdot d_{ff} + d_{ff} \cdot r + r \cdot d)$ |
| LoRA overhead ratio | $\frac{r}{d} + \frac{r}{d_{ff}} = 0.156$ (15.6%) |

At micro scale: $d=64$, $d_{ff}=256$, $r=8$, $L=4$:
- Base: $2 \cdot 4 \cdot (64 \cdot 256 + 256 \cdot 64) = 262,144$ FLOPs
- Dynamic routing: $20 \cdot 64 = 1,280$ FLOPs
- LoRA apply: $2 \cdot 4 \cdot (64 \cdot 8 + 8 \cdot 256 + 256 \cdot 8 + 8 \cdot 64) = 37,888$ FLOPs

## Worked Example (d=64, N=5, r=8)

Suppose 5 experts with deltas $\|\Delta W_i\| \approx 0.01 \|W_{\text{base}}\|$.

**Pre-merge:** Each expert contributes $0.2 \times 0.01 = 0.002$ of base weight norm.
Total perturbation: $\sqrt{5} \times 0.002 = 0.0045$ (by orthogonality, norms add in quadrature).

**Dynamic top-1:** Selected expert contributes $1.0 \times 0.01 = 0.01$ of base weight norm.
Signal is $5\times$ stronger than pre-merge for the relevant expert.

**When does this matter?** Only when $0.01$ (full-strength LoRA) produces measurably
different output from $0.002$ (diluted LoRA). At micro scale with 0.0% specialization,
neither produces meaningful output change.

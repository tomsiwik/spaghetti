# Multi-Seed Validation: Mathematical Foundations

## Setup

Let the base model be $\mathcal{M}$ with frozen weights $W \in \{-1,0,1\}^{d_{out} \times d_{in}}$ (BitNet-b1.58-2B-4T, $d=2560$, 30 layers).

For each domain $i \in \{1,...,5\}$ and seed $s \in \{42, 137, 314\}$, we train a ternary LoRA adapter:

$$\Delta W_i^{(s)} = B_i^{(s)} A_i^{(s)}, \quad A_i^{(s)} \in \mathbb{R}^{d_{in} \times r}, \quad B_i^{(s)} \in \mathbb{R}^{r \times d_{out}}$$

where $r = 16$ and during training, $A$ and $B$ are quantized via STE:

$$Q(W) = \alpha \cdot \text{clip}(\text{round}(W / \alpha), -1, 1), \quad \alpha = \text{mean}(|W|)$$

The STE passes gradients through the round operation unchanged.

## Composition (Averaged-Factor Method)

Given $N=5$ adapters trained with seed $s$, the `compose_adapters` function merges the factor matrices $A_i$ and $B_i$ **separately** with $1/N$ scaling:

$$A_{\text{merged}}^{(s)} = \frac{1}{N} \sum_{i=1}^{N} A_i^{(s)}, \quad B_{\text{merged}}^{(s)} = \frac{1}{N} \sum_{i=1}^{N} B_i^{(s)}$$

The effective weight delta applied at inference is therefore:

$$\Delta W_{\text{eff}}^{(s)} = B_{\text{merged}}^{(s)} A_{\text{merged}}^{(s)} \cdot \alpha_{\text{lora}}$$

Expanding the product:

$$\Delta W_{\text{eff}}^{(s)} = \frac{1}{N^2} \left[ \sum_{i=1}^{N} B_i^{(s)} A_i^{(s)} + \sum_{i \neq j} B_i^{(s)} A_j^{(s)} \right] \cdot \alpha_{\text{lora}}$$

The first sum contains the $N$ "diagonal" terms (each adapter's own contribution). The second sum contains the $N(N-1)$ cross-terms $B_i A_j$ for $i \neq j$.

**Cross-term magnitude.** Because adapters are nearly orthogonal ($|\cos| \approx 0.002$, which is 40x below the structural bound $\sqrt{r/d} = 0.079$), the cross-terms are small relative to the diagonal terms. Empirically, the composed model behaves as if each adapter contributes at $\sim 1/N^2$ effective strength, not $1/N$.

**Contrast with ideal 1/N composition.** The "textbook" formulation $\hat{W} = W + \frac{1}{N} \sum_i B_i A_i$ would apply $1/N$ scaling to each pre-computed delta. The averaged-factor method instead applies $1/N$ to each factor, yielding $1/N^2$ on diagonal terms plus cross-term noise. This is the same composition used in all prior experiments (bitnet_2b_real_composition, bitnet_ternary_convergence), so results are self-consistent across the codebase.

**Why the composition ratio is ~3.4x.** The $1/N^2$ effective scaling (with $N=5$, so $1/25$) means each adapter contributes a much smaller correction than the $1/N = 1/5$ that textbook composition would give. The composition ratio (composed PPL / best individual PPL) of ~3.4x reflects this stronger dilution.

## Metrics Across Seeds

**Composition ratio** for seed $s$:
$$\rho^{(s)} = \frac{\bar{P}_{\text{composed}}^{(s)}}{\min_i P_{\text{individual},i}^{(s)}}$$

where $P$ denotes perplexity and $\bar{P}_{\text{composed}}$ is the mean composed PPL across domains.

**Cosine similarity** between adapters $i,j$ at seed $s$:
$$\cos_{ij}^{(s)} = \frac{|\langle \text{vec}(\Delta W_i^{(s)}), \text{vec}(\Delta W_j^{(s)}) \rangle|}{\|\text{vec}(\Delta W_i^{(s)})\| \cdot \|\text{vec}(\Delta W_j^{(s)})\|}$$

## Reproducibility Statistics

Over $S=3$ seeds:

$$\bar{\rho} = \frac{1}{S}\sum_s \rho^{(s)}, \quad \sigma_\rho = \sqrt{\frac{1}{S-1}\sum_s (\rho^{(s)} - \bar{\rho})^2}$$

**Coefficient of variation:**
$$\text{CV}(\rho) = \frac{\sigma_\rho}{\bar{\rho}} \times 100\%$$

## Kill Criteria

- **K1:** $\text{CV}(\rho) > 50\%$ implies composition quality is not reproducible
- **K2:** $\exists s: \rho^{(s)} > 10$ implies catastrophe can occur for some initializations

## Expected Values

From single-seed prior ($s=42$): $\rho = 3.45$, $|\cos| = 0.0019$.

Under the hypothesis that ternary STE training produces consistent adapter geometry:
- CV should be small (< 20%) since the base model and data are identical across seeds
- |cos| should remain near $\sqrt{r/d} \approx 0.079$ or below (structural orthogonality)
- Composition ratio should cluster around 3-4x (1/N dilution effect)

## Computational Cost

Per seed: 5 domains x 400 steps x ~0.5s/step = ~17 min training + ~3 min eval.
Total: 3 seeds x ~20 min = ~60 min.

## Dimensions

| Tensor | Shape | Count per layer |
|--------|-------|-----------------|
| $A_i^{(s)}$ | $(d_{in}, 16)$ | 7 (q,k,v,o,gate,up,down) |
| $B_i^{(s)}$ | $(16, d_{out})$ | 7 |
| Base $W$ | $(d_{out}, d_{in})$ | 7 |

Total trainable per adapter: $7 \times 30 \times 2 \times d \times r \approx 17.2M$ parameters.
Total adapters: 15 (5 domains x 3 seeds).

## Assumptions

1. Base model weights are deterministic (same checkpoint, same unpacking)
2. Data order varies by seed (shuffled indices), but same data samples
3. LoRA initialization varies by seed (different random $A$, zero $B$)
4. MLX random number generator is reproducible given same seed
5. STE gradient noise does not cause divergent training trajectories

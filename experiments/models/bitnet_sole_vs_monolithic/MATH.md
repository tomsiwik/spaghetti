# BitNet-SOLE vs Monolithic: Mathematical Foundations

## Setup

### Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| $W$ | $(d_{out}, d_{in})$ | Frozen ternary base weight, $w_{ij} \in \{-1, 0, 1\}$ |
| $A_k$ | $(d_{in}, r)$ | LoRA down-projection for expert $k$ |
| $B_k$ | $(r, d_{out})$ | LoRA up-projection for expert $k$ |
| $Q(\cdot)$ | - | STE ternary quantizer: $Q(W) = \text{clip}(\text{round}(W/\alpha), -1, 1) \cdot \alpha$ |
| $N$ | scalar | Number of domain experts (= 5) |
| $r$ | scalar | LoRA rank (= 16) |
| $d$ | scalar | Model hidden dimension (= 2560 for BitNet-2B-4T) |
| $\alpha_s$ | scalar | LoRA scaling factor (= 20.0) |

### Model Configurations

**SOLE (routed):** For input $x$ from domain $k$:
$$y_k = Wx + \alpha_s \cdot Q(B_k) \cdot Q(A_k) \cdot x$$

**SOLE (composed, 1/N scaling):** For any input $x$:
$$y_{comp} = Wx + \frac{\alpha_s}{N} \sum_{k=1}^{N} Q(B_k) \cdot Q(A_k) \cdot x$$

**Monolithic (shuffled):** Single adapter trained on $\bigcup_k D_k$:
$$y_{mono} = Wx + \alpha_s \cdot Q(B_{mono}) \cdot Q(A_{mono}) \cdot x$$

**Monolithic (sequential):** Same adapter trained on $D_1, D_2, \ldots, D_N$ in sequence:
$$y_{seq} = Wx + \alpha_s \cdot Q(B_{seq}) \cdot Q(A_{seq}) \cdot x$$

## Budget Analysis

### Parameter Budget

| Condition | Rank | Total LoRA params | Training steps |
|-----------|------|-------------------|----------------|
| SOLE (N=5 experts) | $r=16$ each | $5 \times 2 \times 7 \times 30 \times r \times d = 5 \times 2 \times 7 \times 30 \times 16 \times 2560$ | $5 \times 400 = 2000$ |
| Monolithic | $r=16$ | $2 \times 7 \times 30 \times 16 \times 2560$ | $2000$ |

SOLE uses $5\times$ more total parameters but each expert has the same rank
as the monolithic adapter. This is the design point: SOLE adds capacity per
domain rather than sharing a single low-rank subspace across all domains.

### Compute Budget

Both see exactly 2000 gradient updates. The monolithic model sees samples from
all 5 domains in every 2000-step window (shuffled). Each SOLE expert sees only
its own domain's 800 samples across 400 steps.

Per step, compute is identical: one forward + backward through the same model
with the same rank-16 LoRA.

## Theoretical Analysis

### Why SOLE Should Win (Capacity Argument)

Each domain $k$ has an optimal low-rank correction $\Delta_k^* \in \mathbb{R}^{d_{out} \times d_{in}}$
with rank $\leq r$. The monolithic adapter must find:

$$\Delta_{mono}^* = \arg\min_{\text{rank}(\Delta) \leq r} \sum_k \mathbb{E}_{x \sim D_k}[\mathcal{L}(Wx + \Delta x, y)]$$

This is a single rank-$r$ subspace that must serve all $N$ domains. If the
optimal corrections $\Delta_k^*$ span non-overlapping subspaces, the monolithic
adapter is rank-starved: it needs rank $\geq Nr$ to represent all domains but
has only rank $r$.

SOLE avoids this entirely: each expert gets its own rank-$r$ subspace,
yielding effective rank up to $Nr = 80$.

### Why Monolithic Should Win (Cross-Domain Transfer)

If domains share structure (e.g., "reasoning" patterns used in both math and
code), the monolithic adapter can learn shared representations that transfer
across domains. A rank-$r$ subspace that captures shared structure may be
more efficient than $N$ independent rank-$r$ subspaces.

### Expected Outcome

The winner depends on the ratio of shared vs domain-specific structure:
- **High overlap** (e.g., Python vs JavaScript): monolithic wins
- **Low overlap** (e.g., medical vs creative writing): SOLE wins
- **Mixed** (our 5 diverse domains): SOLE should win on most domains

### Composition Analysis

With 1/N scaling and near-orthogonal adapters ($|\cos| \approx 0.002$), the
composed adapter approximates:

$$\Delta_{comp} \approx \frac{1}{N} \sum_k \Delta_k$$

On domain $k$, the interference from other domains is:

$$\text{interference}_k = \frac{1}{N} \sum_{j \neq k} \Delta_j \cdot x_k$$

For near-orthogonal adapters, $\|\text{interference}_k\| \ll \|\Delta_k \cdot x_k / N\|$,
but the 1/N scaling dilutes the correct expert's signal to $1/N$ of its
original strength. This is why routed > composed > monolithic is the expected
ordering for per-domain PPL.

## Worked Example (d=2560, r=16, N=5)

- Subspace capacity: $N_{max} = d^2/r^2 = 2560^2/16^2 = 25,600$
- Using $N=5$: capacity utilization = $5/25600 = 0.02\%$ (well within limits)
- Expected $|\cos|$ between random rank-16 subspaces in $\mathbb{R}^{2560}$: $\sqrt{r/d} = \sqrt{16/2560} = 0.079$
- Measured (from ternary_convergence): $|\cos| = 0.0019$ (41x below random bound)
- Interference bound: $\|\text{interference}\| / \|\text{signal}\| \approx (N-1) \cdot |\cos| / N \approx 4 \times 0.002 / 5 = 0.0016$

## Kill Criteria Formalization

**K1: monolithic beats SOLE routed on >80% of per-domain metrics**

Let $\text{PPL}_k^{routed}$ be the perplexity of SOLE routed on domain $k$,
and $\text{PPL}_k^{mono}$ be the monolithic perplexity on domain $k$.

Define: $\text{mono\_wins} = |\{k : \text{PPL}_k^{mono} < \text{PPL}_k^{routed}\}|$

$$\text{K1 KILLED} \iff \text{mono\_wins} \geq \lceil 0.8 \times N \rceil = 4$$

## Assumptions

1. STE ternary quantization converges at 400 steps (validated in exp_bitnet_ternary_convergence)
2. HuggingFace datasets provide sufficient domain signal (validated in prior experiments)
3. 1/N composition scaling prevents catastrophe (validated in exp_bitnet_scale_n15)
4. Base model is frozen; only LoRA parameters train
5. Validation sets are non-overlapping with training sets (proper split)

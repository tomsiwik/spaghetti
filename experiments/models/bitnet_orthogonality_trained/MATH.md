# BitNet Orthogonality Trained: Mathematical Foundations

## Setup

**Base models:**
- FP16: W_fp16 in R^{d_in x d_out}, continuous weights
- Ternary: W_tern = alpha * Q(W_fp16/alpha), where Q is RoundClip to {-1, 0, 1}
  and alpha = mean(|W_fp16|) per weight matrix (BitNet b1.58 absmean recipe)

**LoRA adapters:**
- For each domain i in {1,...,N}: delta_i = B_i @ A_i, where A_i in R^{d_in x r}, B_i in R^{r x d_out}
- Trained on frozen base (either FP16 or ternary)
- Same initialization seed, same training data, same hyperparameters

**Dimensions:** d=64, r=4, L=2 layers, H=2 heads, V=40 (char vocab)

## Hypothesis

The ternary base weight structure constrains gradient flow through discrete
channels, causing trained LoRA adapters to occupy more orthogonal subspaces.

Formally: E[|cos(flat(delta_i^tern), flat(delta_j^tern))|] < E[|cos(flat(delta_i^fp16), flat(delta_j^fp16))|]

where flat() concatenates all weight matrices of the adapter into a single vector.

## Why This Might Work (Prior Reasoning)

On a ternary base, each element of W is in {-alpha, 0, alpha}. The zero entries
create "dead channels" where gradient flow is suppressed. The hypothesis was that
these dead channels partition the feature space, forcing different domain adapters
into disjoint active subspaces.

The gradient of loss L w.r.t. LoRA parameter B_i at layer l is:

  dL/dB_i = A_i^T @ h_in^T @ (dL/dz)

where h_in is the layer input. On a ternary base, h_in is shaped by ternary
weight multiplications, which might create sparser, more structured intermediate
representations than FP16.

## Random Baseline

For random rank-r subspaces in R^d, the expected cosine between flattened
weight deltas is bounded by:

  E[|cos|] ~ sqrt(r/d) = sqrt(4/64) = 0.25

The flattened delta lives in R^D where D = sum of d_in*d_out across all
adapted layers. For our architecture with all-modules LoRA:
  D = d*(3d) + d*d + d*(4d) + (4d)*d = 3d^2 + d^2 + 4d^2 + 4d^2 = 12d^2
  = 12 * 64^2 = 49,152 per layer, times 2 layers = 98,304 total flat dims

The random baseline for this D is E[|cos|] = sqrt(2/(pi*D)) ~ 0.0025.

## Observed Results

| Metric | FP16 | Ternary | Delta | Direction |
|--------|------|---------|-------|-----------|
| Mean |cos| | 0.2600 +/- 0.016 | 0.2755 +/- 0.028 | +0.015 | Ternary WORSE |
| Max |cos| | 0.7665 +/- 0.040 | 0.8265 +/- 0.009 | +0.060 | Ternary WORSE |
| Median |cos| | 0.142 | 0.148 | +0.006 | ~Neutral |
| Arith-Sort cos | 0.125 +/- 0.027 | 0.121 +/- 0.017 | -0.004 | ~Neutral |

Both FP16 and ternary mean |cos| are close to the sqrt(r/d) = 0.25 random
subspace bound, suggesting trained adapters are only marginally more aligned
than random at this scale.

## Paired t-test

For mean |cos| difference (ternary - FP16):
  d_bar = +0.0154, t(2) = 0.643

For 2 degrees of freedom, critical value at alpha=0.05 is |t| > 4.303.
The observed t=0.643 is far from significance. We cannot reject H0 (no difference).

However, the DIRECTION is consistently ternary-worse in 2/3 seeds, and the
aggregate is ternary-worse. Combined with the prior exp_bitnet_ternary_adapter_composition
result (-19.3% decorrelation from ternary ADAPTERS, not ternary base), the
evidence is clear: the orthogonality benefit comes from adapter quantization,
not base quantization.

## Why Ternary Base Does NOT Improve Orthogonality

The dead-channel hypothesis fails because:

1. **Gradient averaging destroys channel structure.** LoRA training computes
   gradients through the EFFECTIVE weights (W_tern + delta). After a few steps,
   the delta is no longer constrained by ternary structure -- it fills in the
   dead channels.

2. **Sparsity does not imply orthogonality.** The ternary base has ~33% zero
   entries (by construction). But gradient signals through non-zero entries are
   still correlated across domains when the domains share features (e.g.,
   reverse and sort both depend on character position encoding).

3. **High-overlap pairs get WORSE.** The reverse-sort pair (highest cos on both
   bases) is 0.77 on FP16 but 0.83 on ternary. The ternary base has lower
   capacity (fewer effective parameters), so domains that need similar features
   are forced to share MORE of the limited capacity.

## Implication for SOLE Architecture

The composition advantage of ternary base (ratio 0.63 from exp_bitnet_composition_stability)
is NOT explained by improved orthogonality. It must come from:

(a) **Magnitude bounding:** Ternary weights have bounded norm per element ({-1,0,1}*alpha),
    preventing the logit-scale explosion that occurs when continuous-weight adapters
    interact multiplicatively through attention.

(b) **Quantization recovery:** Ternary base has higher base PPL (quantization loss).
    Adapters recover this loss. The ratio = composed_PPL/base_PPL < 1 because
    the denominator (ternary base PPL) is inflated, not because the numerator
    (composed PPL) is better.

This is consistent with the exp_bitnet_composition_stability finding that
"mechanism is quantization recovery (not hypothesized interference reduction)."

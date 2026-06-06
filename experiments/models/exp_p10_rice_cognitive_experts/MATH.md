# MATH.md — RICE Cognitive Layer Identification on Gemma 4 E4B

## Context

**RICE** (arXiv:2505.14681) identifies "cognitive experts" in MoE models via
normalized pointwise mutual information (nPMI) between expert activations and
thinking tokens, then amplifies those experts to boost reasoning.

Gemma 4 E4B is **not MoE** — it has 42 dense decoder layers. We adapt RICE
from expert-level to **layer-level**: identify which layers contribute most
to reasoning, then amplify via the model's native `layer_scalar` mechanism.

**Critical tension (Finding #528):** 4-bit quantization eliminates thinking-mode
benefit on GPQA Diamond (30.8% vs 31.8%, −1.0pp). Thinking tokens under 4-bit
are semantically broken — they generate plausible-looking chains with no
reasoning improvement. This means nPMI(layer, thinking_token) may measure
noise, not cognitive specialization.

## Definitions

Let $L = 42$ layers with functions $f_l$ and scalars $s_l$. The residual
stream evolves as:
$$h_l = s_l \cdot (h_{l-1} + \text{attn}_l(h_{l-1}) + \text{mlp}_l(h_{l-1}) + \text{gate}_l(h_{l-1}))$$

**Residual contribution** of layer $l$ at token position $t$:
$$\Delta_{l,t} = \|h_{l,t} - h_{l-1,t}\|_2$$

**Normalized contribution** (relative to input):
$$c_{l,t} = \Delta_{l,t} / \|h_{l-1,t}\|_2$$

## nPMI Formulation

Partition tokens into thinking ($\mathcal{T}$) and non-thinking ($\mathcal{N}$).
For each layer $l$, define "high activation" as $c_{l,t} > \text{median}_t(c_{l,\cdot})$.

$$\text{nPMI}(l) = \frac{\log \frac{p(c_l^{high} \cap \mathcal{T})}{p(c_l^{high}) \cdot p(\mathcal{T})}}{-\log p(c_l^{high} \cap \mathcal{T})}$$

Range: $[-1, 1]$. Value $> 0.3$ indicates the layer is disproportionately
active during thinking.

## Theorem 1 (Quantization Noise Dominates Thinking Signal)

**Claim:** Under $q$-bit quantization with per-layer error $\epsilon_q$, the
compounded error through $L$ reasoning steps grows as $O(L \cdot \epsilon_q)$.
When this exceeds the cognitive signal, nPMI between any layer and thinking
tokens approaches zero.

**Argument:** Each quantized layer introduces error $\epsilon_q \approx 2^{-q}$
in its weight matrices. For 4-bit, $\epsilon_q \approx 0.06$. After $L = 42$
layers, the compounded perturbation to the residual stream is:
$$\|\delta h_L\| \leq \sum_{l=1}^{L} \epsilon_q \cdot \|W_l\| \cdot \|h_{l-1}\| \approx L \cdot \epsilon_q \cdot \bar{W} \cdot \bar{h}$$

For the thinking signal to be detectable, it must produce layer-level activation
differences exceeding this noise floor. Finding #528 empirically showed the
signal does NOT exceed the noise (thinking gives −1.0pp, consistent with random).

**Prediction 1:** nPMI(layer, thinking) < 0.3 for ALL 42 layers.
**Prediction 2:** K1 FAILS — no cognitive layers identifiable via thinking tokens.

## Theorem 2 (Layer Contribution Heterogeneity)

**Claim:** Even if thinking tokens carry no signal, layer contributions
$\Delta_{l,t}$ are heterogeneous across layers, with characteristic patterns.

**Argument:** Transformer layer contributions follow documented patterns
(Ethayarajh 2019, arxiv:1909.00512): early layers handle syntactic features,
middle layers encode semantic relationships, late layers specialize for
prediction. The `layer_scalar` $s_l$ values learned during training encode
the model's own assessment of layer importance.

**Prediction 3:** Layer contribution variance $\sigma^2(\bar{\Delta}_l) > 0$,
likely with U-shaped profile (high at boundaries, lower in middle).

**Prediction 4:** Layer scalars $s_l$ are NOT uniform — some layers are
genuinely more important than others.

## Amplification Analysis

If cognitive layers ARE identified (K1 passes against prediction), RICE proposes
scaling by $\beta = 64$. For a dense model with residual connections:

$$h_l^{amp} = \beta \cdot s_l \cdot f_l(h_{l-1}) + h_{l-1}$$

The residual stream norm after amplification:
$$\|h_l^{amp}\| \approx \|h_{l-1}\| + \beta \cdot s_l \cdot \|f_l(h_{l-1})\|$$

For $\beta = 64$ and typical $\|f_l\|/\|h_{l-1}\| \approx 0.1$, the amplified
layer contributes $\sim 6.4\times$ the input norm. This catastrophically
distorts the residual stream for downstream layers.

**Prediction 5:** $\beta = 64$ will crash model quality. Maximum safe
amplification $\beta_{max} \approx 2.0$ (doubling layer contribution).

## Kill Criteria Predictions

| Kill Criterion | Prediction | Reasoning |
|---|---|---|
| K1: >= 2 layers with nPMI > 0.3 | **FAIL** | Thinking tokens are noise under 4-bit (Finding #528) |
| K2: AIME +5pp from amplification | **FAIL** | No cognitive layers to amplify (K1 prerequisite) |
| K3: MMLU < 2pp degradation | **N/A** | Only tested if K2 is attempted |

**Expected outcome:** KILLED. This would be the 5th independent confirmation
that 4-bit quantization fundamentally limits reasoning capability — not just
in generation (Finding #528) but in the layer-level activation patterns that
support reasoning.

## Alternative: Correct-vs-Incorrect nPMI

As a secondary analysis (not part of kill criteria), we compute nPMI between
layer activations and CORRECT vs INCORRECT answer classification. This signal
does not depend on thinking tokens and may reveal which layers are most
responsible for correct reasoning under 4-bit quantization. This is exploratory
and not expected to pass K1 thresholds, but provides diagnostic value.

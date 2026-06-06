# Self-Contrast Decoding for LoRA Adapter Composition

## Type: Frontier Extension

**Proven result:** SCMoE (2405.14507) demonstrates that unchosen MoE experts contain
useful signal extractable via contrastive decoding. Mixtral GSM8K: 61.79 -> 66.94
(+8.3%) training-free. Contrastive Decoding (Li et al., 2210.15097) provides the
theoretical foundation: expert vs amateur logit subtraction amplifies the expert's
unique capabilities.

**Gap:** SCMoE operates on native MoE experts (FFN blocks within one model). We extend
to LoRA adapters (low-rank additive perturbations to a shared base model). The adapters
were trained independently on different domains. The question is whether non-primary
adapters carry useful contrastive signal or just noise.

## A. Failure Mode Identification

**Primary failure:** Non-primary adapters contribute noise rather than signal. When
the amateur logits (base + non-primary) are subtracted from expert logits
(base + primary), the contrastive term amplifies spurious correlations in
non-primary adapters rather than suppressing irrelevant knowledge.

**When this happens:** If non-primary adapter outputs are random w.r.t. the query,
contrastive decoding becomes `logits_expert - alpha * noise`, which degrades
generation quality by introducing high-variance perturbations.

**Why this is a real risk:** Finding #242 shows H^1 = 3 pairwise topological
obstructions between adapter pairs. This means some adapter pairs are fundamentally
incompatible in function space. Using an incompatible adapter as "amateur" could
amplify exactly the wrong features.

## B. The Right Question

Wrong: "How do we extract value from unchosen adapters?"
Right: "Under what conditions does logit subtraction with a non-primary LoRA adapter
provably sharpen the primary adapter's output distribution?"

## C. Prior Mathematical Foundations

**Contrastive Decoding (Li et al., 2210.15097):**
Given expert distribution p_E and amateur distribution p_A, the contrastive
distribution is:

    p_CD(x_t | x_{<t}) proportional to p_E(x_t | x_{<t}) * [p_E(x_t | x_{<t}) / p_A(x_t | x_{<t})]^alpha

In log-space:

    logit_CD = (1 + alpha) * logit_E - alpha * logit_A

This amplifies tokens where the expert is relatively more confident than the amateur.

**SCMoE (2405.14507) Theorem (informal):**
For MoE with expert e_i, if unchosen experts {e_j}_{j != i} have lower affinity
for input x, then the contrastive signal p_{e_i}(x) / p_{avg(e_j)}(x) upweights
tokens in e_i's specialization and downweights shared/generic tokens.

**Key condition:** The amateur must be "reasonably competent but less specialized."
If the amateur is pure noise, contrastive decoding amplifies noise.

## D. Proof Sketch (Frontier Extension -- Full Proof Requires New Math)

**Setting:** Base model M with weights W. Primary adapter P with low-rank
perturbation Delta_P = B_P^T A_P^T (scale s_P). Non-primary adapters
{Q_1, ..., Q_{K-1}} with perturbations Delta_{Q_j}.

Expert logits: z_E = (W + s_P * Delta_P)(x)
Amateur logits: z_A = (W + (1/K-1) * sum_j s_{Q_j} * Delta_{Q_j})(x)

Contrastive logits:
    z_CD = (1+alpha) * z_E - alpha * z_A
         = W(x) + (1+alpha) * s_P * Delta_P(x) - alpha * (1/K-1) * sum_j s_{Q_j} * Delta_{Q_j}(x)

**Observation 1:** The base model contribution W(x) remains with coefficient 1
(the (1+alpha) and -alpha cancel on the W(x) terms). This means contrastive
decoding does NOT distort the base model's knowledge -- it only modifies the
adapter contributions.

**Observation 2:** The primary adapter is amplified by (1+alpha). For alpha=0.5,
the primary effect is 1.5x. This is equivalent to using a higher adapter scale.

**Observation 3:** The non-primary adapters contribute a negative signal scaled
by alpha/(K-1). For K=5 and alpha=0.5, each non-primary adapter's negative
contribution is 0.125x -- small relative to the 1.5x primary.

**Proposition 1 (Signal condition):** Self-contrast improves over single-adapter
when the non-primary adapters' average logit shift anticorrelates with the
"irrelevant" tokens for the primary domain. Formally, let:
- z_domain = tokens where primary adapter has learned specialization
- z_generic = shared/generic tokens

If avg(Delta_{Q_j}(x)) has positive correlation with z_generic and low
correlation with z_domain, then subtracting it suppresses generic tokens
and relatively boosts domain-specific tokens.

**Proposition 2 (Noise condition):** Self-contrast degrades when
avg(Delta_{Q_j}(x)) is uncorrelated with both z_domain and z_generic
(pure noise). In this case, the subtraction adds variance proportional
to alpha^2 * Var(avg(Delta_Q)) to the logits.

**Prediction:** Domains where adapters have clear specialization (math, code)
should benefit from self-contrast. Domains where adapters are weakly trained
or overlap significantly (legal, finance at scale=1-4) may degrade.

## E. Quantitative Predictions

From the behavioral eval baseline (Finding #238):
- Base scores: medical=0.263, code=0.419, math=0.10, legal=0.098, finance=0.176
- Single-adapter (routed): medical=0.291, code=0.624, math=0.80, legal=0.096, finance=0.156

**P1:** For math (strongest adapter, scale=20), self-contrast with alpha in [0.1, 0.5]
should maintain or improve score >= 0.80 (primary signal is strong, non-primary
adapters are clearly non-math).

**P2:** For code (strong adapter, scale=20), self-contrast should maintain score >= 0.62.

**P3:** For finance (weakest adapter, scale=1), self-contrast is likely to DEGRADE
because the primary signal is weak and non-primary noise is relatively large.

**P4:** Overall, self-contrast should beat single-adapter on >= 2/5 domains.
K1 requires >= 3/5 domains worse to KILL. We predict 1-2 domains worse (finance, possibly legal).

**P5:** Latency should be approximately 2x single-adapter (two forward passes:
expert + amateur). Well under K2's 3x threshold.

## F. Worked Example (Conceptual)

Consider a math query "What is 24 * 7?"

Expert (base + math adapter at scale 20):
- Token "168": logit = 5.0 (strong math signal)
- Token "The": logit = 3.0 (generic)

Amateur (base + avg(medical, code, legal, finance)):
- Token "168": logit = 2.0 (no math specialization)
- Token "The": logit = 3.5 (generic, slightly boosted by general adapters)

Contrastive (alpha=0.5):
- Token "168": (1.5)(5.0) - (0.5)(2.0) = 7.5 - 1.0 = 6.5 (amplified)
- Token "The": (1.5)(3.0) - (0.5)(3.5) = 4.5 - 1.75 = 2.75 (suppressed)

The math-specific token is relatively boosted vs the generic token.

## G. Complexity & Architecture Connection

**Per-token cost:**
- Single adapter: 1 forward pass (base + merged primary adapter)
- Self-contrast: 2 forward passes (expert + amateur) + logit arithmetic
- Overhead: approximately 2x compute, but pre-merge is free so both are just
  standard forward passes through different weight configurations

**Memory:** Peak = 2x model weights (if computing both simultaneously) OR
1x model weights + logit buffer (if sequential with weight swap).
We use sequential: restore base -> merge expert -> forward -> save logits ->
restore base -> merge amateur -> forward -> contrastive arithmetic.

**Implementation:** Pre-merge is free on MLX (0.80% overhead proven).
The bottleneck is generating tokens sequentially with two different weight
configurations per token. For greedy decoding (temperature=0), we can
pre-compute the contrastive logits and sample once.

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   It does NOT make failure impossible -- this is a frontier extension. The property
   that makes it WORK (when it works) is that non-primary adapters' shared generic
   signal is suppressed by subtraction, relatively amplifying domain-specific tokens.

2. Which existing theorem(s) does the proof build on?
   Contrastive Decoding (Li et al., 2210.15097), SCMoE (Chen et al., 2405.14507).

3. What specific numbers does the proof predict?
   P1: math score >= 0.80; P2: code score >= 0.62; P3: finance likely degrades;
   P4: <= 2/5 domains worse; P5: latency approximately 2x.

4. What would FALSIFY the proof (not just the experiment)?
   If self-contrast degrades ALL domains including math and code (the strongest
   adapters), then the assumption that non-primary adapters carry suppressive
   signal for generic tokens is wrong -- they carry noise.

5. How many hyperparameters does this approach add?
   Count: 1 (alpha). Why can't it be derived? The optimal alpha depends on the
   SNR of the non-primary adapter signal, which varies per domain. We sweep
   alpha in {0.1, 0.3, 0.5, 1.0} to discover the empirical optimum.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is a standalone inference-time technique that requires zero training.
   It operates on existing trained adapters with no modifications.

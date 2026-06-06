# N=25 v_proj+o_proj Composition Scaling — Mathematical Framework

## Context
Finding #505 proved N=5 parameter-merged composition preserves behavioral quality
(4/5 domains >= 100% retention, ensemble effect). This experiment tests scaling to N=25.

Finding #506 killed HF task-completion training data (distribution mismatch). P8-style
explanatory data is the validated approach.

## Theorem: Interference Scaling Under Equal-Weight Parameter Merging

**Setup.** Given N rank-r LoRA adapters {(A_i, B_i)}_{i=1}^N trained on v_proj+o_proj
layers, the composed weight update is:

$$\Delta W_{\text{composed}} = \frac{\alpha}{N} \sum_{i=1}^{N} B_i A_i$$

where alpha/N is the per-adapter weight under equal weighting (1/N normalization).

**Theorem 1 (Interference Bound).** For input x, the composed activation is:

$$\Delta W_{\text{composed}} \cdot x = \frac{\alpha}{N} \sum_{i=1}^{N} B_i (A_i x)$$

The contribution of adapter i to domain j's evaluation query is:

$$\text{signal}_j = \frac{\alpha}{N} B_j (A_j x)$$

$$\text{interference}_j = \frac{\alpha}{N} \sum_{i \neq j} B_i (A_i x)$$

The signal-to-interference ratio for domain j is:

$$\text{SIR}_j = \frac{||B_j(A_j x)||}{||\sum_{i \neq j} B_i(A_i x)||}$$

**Lemma 1 (Random B-matrix Interference).** If B-matrices are approximately random
in their output directions (validated by Finding #353: max activation cos = 0.29 at N=5),
then the interference from (N-1) cross-terms has expected norm:

$$E[||\sum_{i \neq j} B_i(A_i x)||^2] \approx (N-1) \cdot E[||B_i(A_i x)||^2]$$

(sum of approximately independent random vectors: variance grows linearly)

Therefore: $E[||\text{interference}||] \approx \sqrt{N-1} \cdot E[||B_i(A_i x)||]$

**Corollary.** SIR degrades as:

$$\text{SIR}(N) \propto \frac{1}{\sqrt{N-1}}$$

At N=5: SIR(5) ∝ 1/2 = 0.5
At N=25: SIR(25) ∝ 1/√24 ≈ 0.204

Ratio: SIR(25)/SIR(5) ≈ 0.204/0.5 = 0.408 (2.45x degradation)

## Predictions

**P1: Retention under scaling.** If N=5 retention = R_5, then:

N=25 retention ≈ R_5 × SIR(25)/SIR(5) = R_5 × 0.408

But this assumes interference dominates signal, which contradicts the N=5 ensemble effect.
The ensemble effect (retention > 100%) suggests cross-adapter contributions are POSITIVE
for many domains (shared vocabulary between math/code/medical creates constructive interference).

**Two competing effects:**
- Destructive: SIR degradation → retention ∝ 1/√N
- Constructive: ensemble effect → retention > 100% from vocabulary overlap

**Revised prediction (accounting for ensemble):**

At N=5, mean retention = 113% (from Finding #505, excluding legal).
The ensemble bonus is +13%. Under random B-matrix assumption, interference
grows as √N but ensemble benefit grows as N (more domains contributing vocabulary).

The net effect depends on domain vocabulary overlap:
- High-overlap domains (math-physics, medical-biology): ensemble persists
- Low-overlap domains (legal-astronomy): SIR degradation dominates

**Predicted outcomes:**
- Mean retention across 25 domains: 70-90% (ensemble partially compensates SIR loss)
- High-overlap cluster (STEM): retention 85-110%
- Low-overlap domains: retention 40-70%
- No domain < 30% (catastrophic threshold — would indicate rank collapse)
- PPL degradation: 2-4% (parameter space interference is bounded by Welch)

## Kill Criteria (from experiment DB)
- K1324: Mean retention >= 70% across 25 domains → PASS if ensemble partially compensates
- K1325: No single domain < 30% → PASS unless rank collapse occurs
- K1326: PPL <= 5% degradation → PASS (parameter-space interference bounded)
- K1327: Latency <= 110% of base → PASS (pre-merged weights, no runtime overhead)

## What Would Kill This

1. **Rank collapse**: If 25 adapters' A-matrices span a low-rank subspace,
   effective adapter dimensionality collapses. Detectable by: mean retention < 30%.
2. **Catastrophic interference**: If B-matrix output directions cluster (not random),
   interference grows faster than √N. Detectable by: multiple domains < 30%.
3. **Scale-dependent phase transition**: If there's a critical N beyond which
   composition quality cliff-drops. Detectable by: N=25 retention << N=5 × 0.408.

## References
- Finding #505: N=5 composition, ensemble effect, 113% mean retention (4/5 domains)
- Finding #506: Distribution mismatch killed HF data; P8-style explanatory data validated
- Finding #504: v_proj+o_proj correct projection target for behavioral quality
- Finding #353: Activation-space cosine similarity max 0.29 at N=5
- Finding #502: TF-IDF routing 84.2% at N=25 (routing works, composition untested)
- LoRA (arXiv:2106.09685): Low-rank adaptation, weight-space modification
- DoRA (arXiv:2402.09353): Output-path modifications improve generation quality

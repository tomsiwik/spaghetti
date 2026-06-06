# Mathematical Foundations: Retrain-from-Scratch + Quality Gate

## Notation

| Symbol | Definition | Shape/Domain |
|--------|-----------|--------------|
| W_base | Ternary base model weights | {-1,0,1}^{d_out x d_in} |
| A_i, B_i | LoRA adapter matrices for expert i | A: R^{d_in x r}, B: R^{r x d_out} |
| Delta_i = B_i A_i^T | Effective weight delta for expert i | R^{d_out x d_in} |
| s | LoRA scale factor | R (= 20.0) |
| KR(M, D) | KR-Test score of model M on contrastive set D | [0, 1] |
| delta_KR(i) | KR improvement: KR(base + adapter_i) - KR(base) | [-1, 1] |
| cos(i, j) | Mean absolute cosine between adapters i, j | [0, 1] |
| PPL(M, D) | Perplexity of model M on validation set D | R+ |

## Evolve Quality Gate

The quality gate accepts a retrained adapter if it passes both criteria:

1. **Knowledge criterion**: delta_KR(i) > tau_KR where tau_KR = 0.03
   - Threshold calibrated from KR-Test evaluation (1 SE at n=50)
   - Measures factual knowledge improvement over base

2. **Composition criterion**: max_j |cos(i, j)| < tau_cos where tau_cos = 0.01
   - Ensures new adapter is orthogonal to existing experts
   - Preserves composition stability per Grassmannian skeleton guarantee

Combined gate: G(i) = I[delta_KR(i) > tau_KR] * I[max_j |cos(i, j)| < tau_cos]

## Monotonic Improvement Hypothesis

Given rounds r = 1, 2, 3 with progressively more training data:
- D_1 subset D_2 subset D_3 (nested data, 800/1200/1600 samples)
- A_r trained from fresh init on D_r

**Claim**: KR(base + A_r) is non-decreasing in r.

**Justification**: More diverse training data should improve domain coverage,
which cross-item contrastive pairs are designed to measure. The fresh init
ensures no warm-start confound (killed in clone-compete).

## PPL vs KR-Test: Different Signals

PPL measures token-level prediction quality across the entire vocabulary.
KR-Test measures binary discrimination between correct and wrong continuations.

At BitNet-2B scale with 300 training steps:
- PPL is sensitive to adapter learning (base 26.62 -> retrained 13.08 = 2.03x)
- KR-Test is NOT sensitive at n=50 pairs (base 0.54 -> retrained 0.56 = +0.02)

The divergence occurs because:
1. PPL averages over all tokens; KR-Test only evaluates at the response boundary
2. Legal text has high surface-level predictability (formal language, citations)
3. At 300 steps, the adapter learns stylistic patterns (lowering PPL) before
   factual associations (which KR-Test measures)

## Statistical Power Analysis

With n=50 contrastive pairs:
- Standard error: SE = sqrt(p(1-p)/n) = sqrt(0.56*0.44/50) = 0.070
- To detect delta=0.03: z = 0.03/0.070 = 0.43 (power ~ 20%)
- To detect delta=0.10: z = 0.10/0.070 = 1.43 (power ~ 58%)
- Need n=200 for delta=0.03 at 80% power

At n=50, the KR-Test cannot distinguish a +0.02 effect from noise. The
experiment is underpowered for the quality gate threshold.

## Cosine Analysis

Observed |cos| between retrained legal and other adapters: 0.013-0.016
Quality gate threshold: 0.01

These cosine values are higher than the intra-domain |cos|=0.00125 from
prior experiments because the prior experiments used adapters trained on
the same base model checkpoint. Here, the comparison is between adapters
trained on different data domains, which naturally have slightly different
gradient directions.

The 0.01 threshold may need calibration. The Grassmannian skeleton
guarantee (||Delta_i^T Delta_j|| -> 0 when A_i perp A_j) operates at the
delta level, not the raw parameter level. Cosine at 0.015 still gives
17x decorrelation per the proven Grassmannian filter.

## Worked Example

Base KR-Test on 50 legal pairs: 27/50 = 0.540
Retrained R1 KR-Test: 28/50 = 0.560
Delta: +1/50 = +0.020
SE: 0.070
z-score: 0.020 / 0.070 = 0.286
p-value: 0.39 (not significant)

Base PPL: 26.62, Original (degenerate) PPL: 57.59, Retrained PPL: 13.08
PPL improvement over degenerate: 57.59 / 13.08 = 4.40x (massive)
PPL improvement over base: 26.62 / 13.08 = 2.03x

The PPL improvement is unambiguous. The KR-Test improvement is noise.

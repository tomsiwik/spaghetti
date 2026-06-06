# Data Scaling Per Expert: Mathematical Foundations

## Variables and Notation

| Symbol | Shape / Type | Definition |
|--------|-------------|------------|
| N | scalar (int) | Number of training examples for expert |
| r | scalar (int) | LoRA rank |
| d | scalar (int) | Model dimension |
| L | scalar (float) | NTP loss (cross-entropy) |
| PPL | scalar (float) | Perplexity = exp(L) |
| alpha | scalar (float) | Power law decay exponent |
| a | scalar (float) | Power law coefficient |
| P_expert | scalar (int) | Number of trainable LoRA parameters |
| T | scalar (int) | Fixed training steps (compute budget) |

## Core Scaling Law

We model expert quality as a power law in training data:

$$PPL(N) = a \cdot N^{-\alpha} + PPL_\infty$$

where PPL_infty is the asymptotic perplexity (capacity-limited floor).

In log-log space, the simplified model (ignoring PPL_infty):

$$\log PPL = \log a - \alpha \log N$$

This is a standard data scaling law analogous to Chinchilla (Hoffmann et al., 2022)
but applied to LoRA fine-tuning rather than pretraining.

## Capacity Bound

A rank-r LoRA adapter has:

$$P_{expert} = 2 \cdot r \cdot (\text{number of adapted weight matrices}) \cdot d$$

For our FFN-only LoRA on a 4-layer GPT with d=64, r=8:

$$P_{expert} = 2 \cdot 8 \cdot (2 \cdot 4) \cdot 64 = 8192 \text{ parameters}$$

Each training example is a name (~5-10 characters = ~5-10 tokens).
The effective "tokens seen" at step t with batch size B:

$$\text{tokens}_t = t \cdot B \cdot \text{seq\_length}$$

For T=300, B=32: tokens_total ~ 300 * 32 * 10 ~ 96K tokens.

With N=50 examples, each example is seen ~1920 times (extreme overfitting).
With N=5000, each example is seen ~19 times (moderate repetition).

## Overfitting-Underfitting Tradeoff

The validation loss has two regimes:

**Regime 1 (N < N_crit)**: Expert overfits to limited data, val loss > base.

$$L_{val}(N) > L_{base} \quad \text{for } N < N_{crit}$$

**Regime 2 (N >= N_crit)**: Expert specializes correctly, val loss < base.

$$L_{val}(N) < L_{base} \quad \text{for } N \geq N_{crit}$$

The crossover N_crit depends on adapter capacity and training steps.
Empirically: N_crit ~ 200 for r=8, T=300, d=64.

## Saturation Analysis

Define the marginal efficiency:

$$\eta(N_1, N_2) = \frac{PPL(N_1) - PPL(N_2)}{N_2 - N_1}$$

Saturation occurs when eta drops below a threshold for all subsequent intervals.

We define the saturation point N_sat as:

$$N_{sat} = \min\{N_i : \forall j > i, \frac{PPL(N_j) - PPL(N_{j+1})}{PPL(N_j)} < \tau\}$$

With tau = 5%, empirically N_sat = 200.

## Empirical Results (d=64, r=8, T=300)

| N | PPL (mean +/- std) | Relative to base |
|---|-----|------|
| 50 | 2.32 +/- 0.13 | -43.3% (worse) |
| 100 | 1.87 +/- 0.03 | -15.6% (worse) |
| 200 | 1.63 +/- 0.01 | -0.9% (break-even) |
| 500 | 1.56 +/- 0.00 | +3.4% (better) |
| 1000 | 1.55 +/- 0.00 | +4.3% |
| 2000 | 1.54 +/- 0.00 | +4.7% |
| 5000 | 1.54 +/- 0.00 | +4.9% |

Power law fit: PPL = 2.72 * N^(-0.077), R^2 = 0.68.

Low R^2 is expected: the power law does not capture the two-regime structure.
A piecewise model (overfitting regime + saturation regime) would fit better.

## Worked Example: Cost-Optimal Budget

Given:
- Expert cost at N=1000 via Groq batch: $0.02 (1000 examples * $0.02/1K)
- PPL at N=1000: 1.55
- PPL at N=200: 1.63
- PPL improvement N=200 to N=1000: 0.9%

At N=200:
- Cost: $0.004
- PPL: 1.63
- Marginal cost to go from 200 to 1000: $0.016 for 0.9% improvement

The diminishing returns are extreme: doubling from 200 to 500 gives 4.3%
improvement; doubling from 500 to 1000 gives 0.9%; doubling from 1000 to
2000 gives 0.4%.

**Cost-optimal recommendation**: N=500 balances quality and cost.
N=200 is the minimum viable dataset. N>1000 is wasteful.

## Scaling to Macro

At macro scale (d=4096, r=16, real data), several factors change:
1. **Adapter capacity**: P_expert = 2 * 16 * (4 * 7 layers * 7 matrices) * 4096 ~ 26M params
   Much higher capacity -> may need more data before saturating
2. **Data diversity**: Real text has higher entropy than character-level names
3. **Training regime**: More steps, better optimization -> may shift saturation point

Prediction: saturation point will scale with adapter capacity.
Rough estimate: N_sat ~ P_expert / (10 * seq_length) ~ 26M / (10 * 512) ~ 5000.
This suggests 1000-5000 examples per expert at macro scale, consistent with
the pilot-50 distillation using 1000 examples.

## Assumptions

1. Single domain (a_e names). Other domains may have different saturation points
   depending on domain complexity.
2. Fixed training steps (T=300). Longer training may shift the saturation point
   slightly but is unlikely to change the qualitative finding.
3. Character-level tokenization. BPE/sentencepiece would change the tokens/example
   ratio but not the fundamental diminishing-returns structure.
4. FFN-only LoRA (rank-8). All-modules LoRA at higher rank would have more capacity
   and potentially later saturation.

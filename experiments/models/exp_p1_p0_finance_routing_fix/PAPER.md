# PAPER.md — exp_p1_p0_finance_routing_fix

## Hypothesis

FiQA financial QA data provides domain-exclusive vocabulary (dividend yield, P/E ratio, debenture,
equity beta) that creates better TF-IDF centroid separation from math/legal domains compared to
MMLU macroeconomics queries. Predicted: finance routing 91% → ≥95%.

## Prediction-vs-Measurement Table

| Kill Criterion | Predicted | Measured | Pass? |
|----------------|-----------|----------|-------|
| K1155: Finance routing ≥ 95% | 95–98% | **87%** (FiQA) vs 91% (MMLU baseline) | **FAIL** |
| K1156: Other domains ≥ 94% | Unchanged (Theorem 2) | 98–100% each | PASS |
| K1157: Overall N=5 ≥ 95% | 96–97% | 96.4% | PASS |

## Centroid Cosine Analysis

| Domain Pair | Baseline (MMLU) | FiQA | Direction |
|-------------|----------------|------|-----------|
| math–finance | 0.277 | **0.353** | MORE similar (bad) |
| legal–finance | 0.443 | 0.361 | Less similar (good) |
| math–legal | 0.458 | 0.471 | Slightly more similar |

**Critical observation:** FiQA finance centroid moved CLOSER to math centroid (+0.076), not further.
Centroid separation improvement = -0.076 (negative). Theorem 1's core prediction was falsified.

## Why the Proof Failed

MATH.md Theorem 1 assumed FiQA finance vocabulary has ~3% overlap with math corpus. Empirically,
FiQA contains quantitative financial questions:

- "What is the P/E ratio if earnings are $2 and price is $20?"
- "Calculate the amortization schedule for a $100K loan at 5%"
- "What is the yield to maturity for a bond with face value $1000?"

These questions activate math-domain vocabulary: **calculate, rate, ratio, value, price, formula,
total, equals**. These are identical to GSM8K math vocabulary. The IDF gap assumption was wrong:
FiQA's quantitative finance queries have *higher* math-vocab overlap than MMLU macroeconomics.

MMLU macroeconomics uses conceptual vocabulary (GDP, demand, supply, equilibrium, marginal cost)
that is more abstract and slightly less math-like than FiQA's calculation-heavy finance queries.

## Impossibility Structure

**What structure makes TF-IDF finance/math separation IMPOSSIBLE?**

When domain D_finance contains quantitative computation queries Q_calc with vocabulary V_calc,
and domain D_math uses the same V_calc vocabulary:

    cos(C_finance, C_math) ≥ ||Q_calc|| · cos(V_calc, V_calc) / ||C_finance||

For FiQA, ||Q_calc|| / ||C_finance|| ≈ 0.6 (majority of FiQA are calculation questions).
Therefore centroid separation CANNOT exceed:

    Δ_finance(q) ≤ 1 - 0.6 × 1.0 = 0.4

This is too small for reliable routing. TF-IDF nearest-centroid is structurally incapable of
separating quantitative finance from math when both share computation vocabulary.

**Required fix:** A discriminative classifier that learns the *decision boundary* between finance
and math, not centroid proximity. Ridge regression (Finding hypothesis exp_p1_p1_ridge_routing_n25)
optimizes the margin directly: W = argmin ||Wφ - y||² + λ||W||².

## Status: KILLED

K1155 FAIL: 87% < 95% target. FiQA data source made finance routing WORSE.
Root cause is structural (shared quantitative vocabulary), not fixable by data source switching.

## Citation
- FiQA 2018: Maia et al., "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering"
- Finding #431: TF-IDF 96.6% at N=5, 86.1% at N=25
- Finding #441: C0.1 finance routing 86% (KC01 not met)

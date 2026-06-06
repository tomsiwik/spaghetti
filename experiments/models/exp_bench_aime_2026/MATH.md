# MATH.md: AIME 2026 Baseline + Pierre Adapted

## Type: Guided Exploration

## Context

AIME 2026 (30 competition mathematics problems, integer answers 0-999) is
the hardest publicly-available math benchmark tracked by MathArena. Problems
require olympiad-level reasoning across combinatorics, number theory, and
algebra — far beyond the K-12 arithmetic of GSM8K.

Google reports **42.5%** pass@4 for Gemma 4 E4B (presumably thinking-enabled,
full precision or 8-bit). We run 4-bit quantized (mlx-community/gemma-4-e4b-it-4bit)
with thinking enabled (mlx_lm.server default streaming with thinking tokens).

Prior calibration points:
- MMLU-Pro: Google 69.4% → E4B-4bit thinking 62.1% (Finding #530, exp_bench_mmlu_pro)
  Quantization gap: 7.3pp (10.5%)
- GPQA Diamond: Google 58.6% → E4B-4bit non-thinking ~35-42% (exp_bench_gpqa_diamond)

GSM8K math adapter: trained on grade-school arithmetic (arXiv:2405.04301, Finding #179).
Domain distance from GSM8K to AIME ≈ maximal — different proof techniques, much higher
ceiling requirements, no calculation shortcuts.

## Theorem 1: Quantization-Adjusted AIME Performance Bound

**Claim:** For Gemma 4 E4B at 4-bit quantization with thinking enabled, the expected
AIME 2026 pass@4 accuracy satisfies:

$$\hat{a}_{4bit} = a_{Google} \cdot (1 - \delta_q)$$

where $\delta_q \in [0.10, 0.15]$ is the relative degradation from 4-bit quantization
on hard reasoning tasks.

**Prior (quantization):** MMLU-Pro calibration gives $\delta_q = 0.105$ (relative).
On harder tasks (AIME requires multi-step symbolic manipulation), quantization noise
compounds across more reasoning steps: expected $\delta_q \in [0.10, 0.18]$.

**Prediction:**
$$\hat{a}_{4bit} \in [42.5\% \times 0.82, 42.5\% \times 0.90] = [34.9\%, 38.3\%]$$

Central estimate: ~37% pass@4 (≈11/30 problems).

**K1417 (within 10pp of 42.5%):** EXPECTED PASS. ~37% is 5.5pp below 42.5%, which is
within the 10pp threshold (range [32.5%, 52.5%]). K1417 FAILS only if quantization
degrades >15pp relative (base < 27.5%) or if the 4-bit model dramatically underperforms
on multi-step symbolic reasoning. High variance: N=30 problems × 2 seeds.

## Theorem 2: Math Adapter Domain Transfer Bound

**Claim:** A math adapter trained on GSM8K provides negligible uplift on AIME 2026.

**Formal:** Let $\mathcal{D}_{GSM}$ and $\mathcal{D}_{AIME}$ be the problem distributions.
The expected adapter uplift $\Delta(x) = P_{adp}(x) - P_{base}(x)$ satisfies:

$$\mathbb{E}_{x \sim \mathcal{D}_{AIME}}[\Delta(x)] \approx 0$$

**Reasoning:** The adapter updates attention/FFN weights to amplify patterns in GSM8K
(2-digit arithmetic, equation solving). AIME requires combinatorial construction,
modular arithmetic, polynomial identities — zero overlap with GSM8K training signal.

Finding #179 measured: math adapter → 48% GSM8K (vs 2% base), +46pp. Same adapter on
MMLU-Pro: ~0pp delta (adapters degrade MCQ recall, per learning from P9.G0).

**Prediction:** Math adapter AIME uplift: $\Delta \in [-3\%, +3\%]$ (within noise).
**K1418 (adapter ≥ base + 10pp):** EXPECTED FAIL.

## Theorem 3: Eval Time Budget

With n=4 seeds × 30 problems × ~90s per problem (4096 max tokens, MLX server):
$T_{eval} \approx 4 \times 30 \times 90 = 10,800\text{s} = 180\text{min} = 3\text{h}$

This EXCEEDS the 2h budget. To meet K1419, we must use n=2 seeds or limit max_tokens.
With n=2 seeds × 30 problems × 60s per problem: $T_{eval} \approx 3,600\text{s} = 60\text{min}$.

**K1419 (< 2h):** CONDITIONAL PASS. Use n=2 seeds to ensure budget.

## Predictions Table (to be filled by PAPER.md)

| Prediction | Value | Source |
|---|---|---|
| Base E4B-4bit pass@2 AIME 2026 | 30–40% | Theorem 1 |
| Math adapter AIME uplift | < 5pp | Theorem 2 |
| K1417 (base within 10pp of 42.5%) | FAIL if <32.5% | Theorem 1 |
| K1418 (adapter ≥ base + 10pp) | FAIL | Theorem 2 |
| K1419 (<2h, n=2) | PASS | Theorem 3 |

## Kill Criteria Reference

- K1417: Base E4B within 10pp of Google's 42.5% (32.5–52.5% range)
- K1418: Math adapter ≥ base + 10pp
- K1419: Eval completes in < 2h on M5 Pro

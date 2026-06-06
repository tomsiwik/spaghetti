# MATH.md: GPQA Diamond Baseline + Pierre Adapted

## Type: Verification

## Context

GPQA Diamond (Rein et al., 2023; arXiv:2311.12022) is a 198-question graduate-level
science MCQ benchmark (physics, chemistry, biology) with 4 options. Questions are
designed to be "Google-proof" — domain experts score ~65%, non-experts ~34%.
Google reports 58.6% for Gemma 4 E4B (full precision, thinking enabled).

We run on 4-bit quantized (mlx-community/gemma-4-e4b-it-4bit) with thinking disabled
for tractable runtime. Prior result from exp_bench_mmlu_pro establishes the baseline
calibration: non-thinking 4-bit Gemma 4 E4B scored 42.3% on MMLU-Pro vs 69.4%
reported (thinking enabled) — a 27.1pp gap.

## Theorem 1: Non-Thinking Accuracy Bound on GPQA Diamond

**Claim:** For a model with thinking-enabled accuracy $a_T$ on a reasoning benchmark,
the non-thinking accuracy $a_{NT}$ satisfies:
$$a_{NT} = a_T - \epsilon_t - \epsilon_q$$
where $\epsilon_t$ is the thinking penalty and $\epsilon_q$ is the quantization penalty.

**Prior (quantization):** Dettmers et al. (arXiv:2208.07339), Frantar et al.
(arXiv:2210.17323): $\epsilon_q \in [1, 3]\text{pp}$ for 4-bit on >1B models.

**Prior (thinking penalty):** Our exp_bench_mmlu_pro measured $\epsilon_t + \epsilon_q = 27.1\text{pp}$
on MMLU-Pro (10-option, reasoning-heavy). GPQA Diamond is harder (graduate-level) but
has only 4 options (less elimination overhead). Two competing effects:

1. *Harder questions require more chain-of-thought* — thinking penalty larger
2. *4 options vs 10 options* — less multi-step elimination, thinking penalty smaller
3. *Random baseline higher* — 25% vs 10%, compresses the range

**Calibration from MMLU-Pro:**
- MMLU-Pro ratio: $a_{NT}/a_T = 42.3/69.4 = 0.610$
- Applying same ratio to GPQA: $58.6 \times 0.610 = 35.7\%$
- But 4-option random baseline is 25% (vs 10% for MMLU-Pro), which compresses degradation
- Adjusted: the thinking gap should be relatively smaller on 4-option MCQ

**Prediction:** $a_{NT} \in [32, 42]\%$ (central: 37%)

Key insight: GPQA Diamond is close to expert ceiling (~65%), so the model is already
operating near the noise floor. Non-thinking may push accuracy toward the non-expert
baseline (~34%), which is near random (25%).

## Theorem 2: NTP Adapter Effect on MCQ

**Prior (Finding #517, exp_bench_mmlu_pro):** NTP adapters uniformly degrade MCQ
by $-6.2\text{pp}$ across all domains, even in-domain (math adapter -13pp on math).
The mechanism is format conflict: NTP loss shifts attention toward language modeling,
away from instruction-following.

**For GPQA Diamond:** The same NTP format conflict applies. However, since GPQA
accuracy is lower (closer to random baseline), the degradation may be smaller
in absolute terms due to floor effects.

$$\Delta_{\text{adapted}} = \max(\delta_{NTP}, a_{random} - a_{base})$$

where $\delta_{NTP} \approx -6\text{pp}$ and $a_{random} = 25\%$.

**Prediction:** Adapted accuracy = $\max(a_{base} - 6, 25) \approx a_{base} - 4\text{pp}$
(floor effect clips degradation).

## Predictions

| Criterion | Prediction (no thinking) | Range |
|-----------|--------------------------|-------|
| Base accuracy | 37% | [32, 42] |
| Google gap | -21.6pp | [-26.6, -16.6] |
| Adapted - base | -4pp | [-6, 0] |
| Runtime | ~20 min | [10, 40] min |

## Kill Criteria Mapping

From experiment definition (IDs from DB):

- **K1: Base within 10pp of Google's 58.6% (i.e. >= 48.6%):** FAIL predicted.
  37% is 21.6pp below 58.6%. This matches MMLU-Pro pattern (non-thinking gap).
  K1 was set assuming thinking; the measurement establishes the non-thinking baseline.

- **K2: Pierre adapter >= base + improvement:** FAIL predicted.
  NTP adapters degrade MCQ (Finding #517). No SFT adapter available.

- **K3: < 2h runtime:** PASS predicted. 198 questions at ~5 q/s = ~40s compute +
  generation time. Even with CoT, < 20 min expected.

## Failure Mode

If base < 25% (random chance on 4-option MCQ): eval pipeline is broken.
Check prompt format, answer extraction, task configuration.

If base > 48%: non-thinking Gemma 4 is stronger on GPQA than predicted.
Would mean thinking penalty is smaller on 4-option format — useful calibration.

## Purpose

This experiment completes the benchmark calibration suite. Combined with MMLU-Pro,
it establishes:
1. The non-thinking penalty on 4-bit Gemma 4 E4B across difficulty levels
2. Whether NTP adapter degradation is consistent across benchmark formats
3. Baseline for future SFT adapter experiments on science domains

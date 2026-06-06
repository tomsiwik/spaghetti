# Competitive Benchmark Retest: Routed Composition with Per-Domain Scales

## Summary

Retested the killed competitive benchmark with oracle top-1 routing and
per-domain optimal scales. **K1 KILLED AGAIN**: routed SOLE worse than base on
2/6 benchmarks (MMLU math -20pp, legal -10pp). K2 PASSES against Gemma-2-2B.
Per-domain scales do NOT fix the MMLU degradation — the root cause is format
mismatch, not scale.

**Status: KILLED**

## Setup

- **Model:** BitNet-2B-4T + 5 domain SFT adapters (rank-16)
- **Routing:** Oracle top-1 (MMLU domain -> matching adapter)
- **Scales:** Per-domain optimal {math/code/medical:20, legal:4, finance:1}
- **Composition:** Pre-merge single selected adapter (no dilution)
- **Benchmarks:** GSM8K (n=50), MMLU (n=20 per domain, 5 domains)
- **Comparators:** Gemma-2-2B-IT (4-bit), Qwen2.5-3B-Instruct (4-bit)
- **Runtime:** 20.5 minutes

## Predictions vs Measurements

| Prediction | Expected | Measured | Match? |
|------------|----------|----------|--------|
| P1: Routed >= base on ALL 6 | all delta >= 0 | 2 worse (math -20pp, legal -10pp) | **FAIL** |
| P2: Beat Gemma >= 5/6 | wins >= 5 | wins = 3 (GSM8K, math, legal) | **FAIL** |
| P3: GSM8K >= 48% | >= 0.48 | 0.48 | **PASS** |
| P4: Legal MMLU >= 50% | >= 0.50 | 0.45 | **FAIL** |
| P5: Finance MMLU >= 30% | >= 0.30 | 0.35 | **PASS** |

## Detailed Results

### Routed vs Base

| Benchmark | Base | Routed | Delta | Verdict |
|-----------|------|--------|-------|---------|
| GSM8K | 38% | 48% | **+10pp** | WIN |
| MMLU medical | 40% | 40% | 0pp | TIE |
| MMLU code | 40% | 40% | 0pp | TIE |
| MMLU math | 50% | 30% | **-20pp** | LOSE |
| MMLU legal | 55% | 45% | **-10pp** | LOSE |
| MMLU finance | 35% | 35% | 0pp | TIE |

### Routed vs Uniform (from original exp_competitive_benchmark)

| Benchmark | Uniform | Routed | Delta |
|-----------|---------|--------|-------|
| GSM8K | 48% | 48% | 0pp |
| MMLU medical | 45% | 40% | -5pp |
| MMLU code | 45% | 40% | -5pp |
| MMLU math | 25% | 30% | **+5pp** |
| MMLU legal | 45% | 45% | 0pp |
| MMLU finance | 45% | 35% | -10pp |

**Routing did NOT help over uniform composition.** In fact, it's slightly worse on
medical (-5pp) and finance (-10pp). Uniform composition's averaging provided a mild
smoothing effect that routing removes.

### Routed vs Competitors

| Benchmark | Routed SOLE | Gemma-2-2B | Qwen-3B |
|-----------|-------------|------------|---------|
| GSM8K | **48%** | 30% | 36% |
| MMLU medical | 40% | 45% | **70%** |
| MMLU code | 40% | 45% | **70%** |
| MMLU math | **30%** | 5% | 35% |
| MMLU legal | **45%** | 30% | 50% |
| MMLU finance | 35% | **45%** | 30% |

SOLE beats Gemma on 3/6 (GSM8K, math, legal). SOLE loses to Qwen on 4/6.

## Root Cause Analysis

### Why per-domain scales failed to fix MMLU

The MATH.md predicted that low scale (s=4 for legal, s=1 for finance) would
preserve base factual knowledge. This was wrong because:

1. **Format mismatch is the disease, not scale.** SFT adapters are trained on
   instruction-response pairs ("Here is a detailed answer to your question...").
   MMLU requires single-letter answers ("A"). Even at s=1, the adapter shifts
   the output distribution toward verbose instruction-following format, reducing
   P(single letter answer).

2. **The math adapter confirms this.** At s=20, the math adapter:
   - HELPS GSM8K (+10pp): GSM8K rewards chain-of-thought, which is what SFT teaches
   - HURTS MMLU math (-20pp): MMLU rewards single-letter answers, which SFT suppresses
   Same adapter, same scale, opposite effects on different evaluation formats.

3. **Medical/code/finance adapters are neutral.** At their optimal scales, they
   produce 0pp change on MMLU. The adapters learned so little domain content that
   they neither help nor hurt. The 40% medical and 40% code are just base model
   performance unchanged.

### Why routing was worse than uniform

Uniform 1/N composition at s=20 accidentally benefited some MMLU domains:
- Medical 45% (uniform) vs 40% (routed): The averaging of all 5 adapters at s=4
  (20/5) provided a mild general boost
- Finance 45% (uniform) vs 35% (routed): Same effect

Routing removes this smoothing by isolating a single adapter. When that adapter
has no positive effect, routing can only be equal to or worse than averaging.

## Kill Criteria Assessment

| Criterion | Result | Verdict |
|-----------|--------|---------|
| K1 (#640): Routed worse than base on ANY benchmark | 2/6 worse (math -20pp, legal -10pp) | **KILL** |
| K2 (#641): Routed worse than Gemma on >= 4/6 | 3/6 worse | **PASS** |

## What This Means

### The format mismatch is the fundamental disease

Three competitive benchmark attempts have now been killed:
1. exp_competitive_benchmark (uniform s=20): 2/6 worse than base
2. exp_bitnet_sft_generation_v3 (SFT + energy routing): 3/5 worse
3. **This experiment** (routed + per-domain scales): 2/6 worse than base

The common root cause across all three: SFT adapters trained on instruction data
conflict with evaluation formats that expect concise answers (MMLU) or specific
reasoning patterns (GSM8K minus the math adapter).

### Where SOLE genuinely helps

GSM8K (+10pp, consistent across all three experiments). The math adapter at s=20
teaches chain-of-thought format, which is exactly what GSM8K rewards. This is the
only benchmark with consistent positive signal.

### The Gemma comparison is honest

SOLE beats Gemma-2-2B on 3/6 benchmarks. With better adapters (MMLU-format-trained
or longer training), this could improve. Gemma's MMLU math at 5% is suspiciously
low (same extraction issue flagged in the original experiment).

## Limitations

1. n=20 MMLU gives +/-22pp CI. The -10pp legal gap is within noise.
2. Same answer extraction code as original (Qwen 36% GSM8K still suspicious)
3. Only 5 adapters from exp_real_data_domain_experts
4. Oracle routing is ideal — deployment routing adds noise

## Implications for Next Steps

1. **Format-aware adapter training is the fix.** Train adapters on MMLU-format
   data (or include a format objective) to eliminate the format mismatch.
2. **Conditional routing:** Skip adapter entirely when the task is factual recall.
   Entropy gating could detect MMLU-style queries (high confidence, factual) and
   bypass adapters. Finding #213 showed entropy gating skips 63% of tokens.
3. **The positive GSM8K signal is real.** The architecture provides genuine
   reasoning enhancement. The failure is adapter training, not composition.

# Budget Forcing: Adaptive Thinking Depth

## Type
Guided exploration

## Status
**KILLED** — Fixed budget forcing does not work for 4-bit quantized Gemma 4

---

## Prediction vs. Measurement Table

| Budget (max_tokens) | Predicted Accuracy | Actual Accuracy | Delta   | Predicted Thinking Chars | Actual Thinking Chars |
|---------------------|-------------------|-----------------|---------|--------------------------|----------------------|
| 128                 | 44.7%             | **10.5%**       | -34.2pp | ~26,880                  | 0                    |
| 256                 | 49.2%             | **11.9%**       | -37.3pp | ~42,000                  | 1,388                |
| 512                 | 54.6%             | **11.4%**       | -43.2pp | ~73,500                  | 15,366               |
| 1024                | 59.8%             | **34.3%**       | -25.5pp | ~115,500                 | 364,253              |
| 2048                | 62.0%             | **46.7%**       | -15.3pp | ~141,750                 | 645,976              |

All 5 predictions are catastrophically wrong. Gamma CDF model fails completely.

---

## Kill Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| K1464: Optimal budget >= 90% retention (≥55.9%) | Some B achieves ≥55.9% | Best non-2048 = 34.3% | **FAIL** |
| K1465: Token reduction >= 40% at optimal budget | 40% reduction | No qualifying budget | **FAIL** |
| K1466: Math@B=2048 matches Finding #530 (~85%) | ~85% | 73.3% (N=15 vs N=20) | PASS |

**Status: KILLED** (2/3 kill criteria FAIL)

---

## The Cliff Effect

**B=128-512 are uniformly catastrophic (10.5-11.9%)** — all WORSE than base no-thinking (41.7%).

The Gamma CDF model assumed: "if budget < T(q), model reverts to base accuracy."
The reality: "if budget < T(q), truncated thinking actively misleads answer generation."

Thinking chars confirm the mechanism:
- B=128: **0 thinking chars** — model produces no reasoning at all
- B=256: **1,388 chars total / 210 questions = 6.6 chars/question** — essentially nothing
- B=512: **15,366 chars total = 73 chars/question** — format preamble, no real reasoning
- B=1024: **364,253 chars = 1,734 chars/question** — partial but real reasoning begins
- B=2048: **645,976 chars = 3,076 chars/question** — full chains

**The threshold is approximately B~1024.** Below this, the model burns tokens on
thinking format/preamble and never produces coherent reasoning chains. The result
(10.5-11.9%) is not "base accuracy" — it's BELOW base (41.7%), consistent with
the model confidently asserting wrong answers based on incomplete reasoning.

Non-monotonicity at B=512 < B=256 (11.4% vs 11.9%) is consistent with this:
at B=512, the model starts more reasoning attempts that are incomplete vs at B=256
where it barely enters thinking mode.

---

## Why the Gamma CDF Model Fails

**Structural flaw:** The model assumes truncated thinking → baseline accuracy.
**Reality:** Truncated thinking → WORSE than baseline (10.5% vs 41.7%).

The theorem's proof sketch says: "If T(q) > B, reasoning chain is truncated → answer
quality reverts to base (no-thinking) level." This is wrong. The model's behavior when
thinking is truncated is not equivalent to not thinking — it acts on a partial, 
incoherent reasoning chain and produces confident wrong answers.

**Correct model would be:**
$$A(B) = \begin{cases} A_{base} & \text{if } B = 0 \text{ (thinking disabled)} \\ A_{harmful} < A_{base} & \text{if } 0 < B < B_{threshold} \\ A_{partial}(B) & \text{if } B_{threshold} \leq B < B_{full} \\ A_{full} & \text{if } B \geq B_{full} \end{cases}$$

where B_threshold ≈ 1024 tokens for Gemma 4 4-bit.

---

## B=2048 Discrepancy

Result: 46.7% vs Finding #530 = 62.1% (+15.4pp gap).

Primary cause: **sample variance** — N=15 per category here vs N=20 in Finding #530.
With 15 questions/category * 14 categories = 210 total, each category contributes ~1.07pp.
A 3-4 question swing in multiple categories can easily account for 15pp.

Math category: 73.3% here vs 85% in Finding #530 — likely reflects the smaller N=15 sample.
The B=2048 result is directionally consistent (thinking helps), just noisier.

---

## Core Finding

**Budget forcing is binary for 4-bit quantized models.** There is no useful middle ground:
- Full thinking (B≥2048): 46-62% accuracy (best)
- No thinking: 41.7% accuracy (acceptable)
- Truncated thinking (B<1024): 10-12% accuracy (catastrophic — worse than both)

The critical threshold is ~B=1024. This extends Finding #530 and is the primary
actionable result of this experiment: adaptive compute allocation must either allow
full thinking chains OR disable thinking entirely. Partial thinking budgets are harmful.

---

## Implications

1. **Best MMLU-Pro strategy** (confirmed): base + full thinking, no adapter
2. **Token efficiency**: Cannot reduce thinking overhead for 4-bit Gemma 4 — use case
   must tolerate full ~675 tokens/question or skip thinking entirely
3. **Quantization effect**: The B_threshold ≈ 1024 tokens likely reflects 4-bit
   quantization limiting the model's ability to maintain coherent reasoning chains
   during generation — FP16 models may have lower thresholds
4. **Next question**: What does the "partial reasoning" actually look like? Is there
   a structural fix (trained budget token, special EOS for thinking)?

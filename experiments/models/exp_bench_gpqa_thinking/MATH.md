# MATH: GPQA Diamond with Thinking Mode

## Type
Verification — quantitative prediction from two prior findings.

## Prior Results
- **Finding #518**: Gemma 4 E4B 4-bit scores 31.8% on GPQA Diamond without thinking (198q, 4-option MCQ). Gap to Google's 58.6% is 26.8pp.
- **Finding #517**: On MMLU-Pro, non-thinking scores 42.3%, thinking scores 69.4% — a 27.1pp boost.
- Google reports 58.6% on GPQA Diamond with thinking enabled.

## Theorem (Empirical Prediction)

**Claim:** The thinking-mode capability boost Δ_think is approximately constant across reasoning-intensive MCQ benchmarks, independent of answer-option count.

**Evidence:**
- MMLU-Pro (10-option): Δ_think ≈ 27.1pp (42.3% → 69.4%)
- GPQA Diamond (4-option): Δ_think predicted ≈ 26.8pp (31.8% → 58.6%)

**Mechanism:** Thinking mode externalizes intermediate reasoning steps that cannot be compressed into a single forward pass. GPQA Diamond requires 3-7 step graduate-level reasoning chains. Without thinking, the model must perform implicit chain-of-thought within residual stream capacity, losing ~27pp.

## Predictions

| Metric | Predicted | Tolerance | Kill if |
|--------|-----------|-----------|---------|
| Base + thinking accuracy | 58.6% | ±9pp | < 50% (K1458) |
| Thinking boost over 31.8% | ≥ 26pp | -11pp | < 15pp (K1459) |
| Total eval time | < 2h | — | ≥ 4h (K1460) |

## Kill Criteria Derivation
- **K1458** (≥ 50%): Google's 58.6% minus 9pp margin for quantization + prompt differences.
- **K1459** (≥ 15pp boost): Conservative lower bound — even half the observed MMLU-Pro boost would be significant. Below 15pp suggests thinking mode doesn't help GPQA-style reasoning.
- **K1460** (< 4h): 198 questions × ~2048 max thinking tokens. At ~50 tok/s thinking generation, ~20s/question × 198 ≈ 1.1h. Factor 3.5x safety margin.

## What Would Kill This
- If thinking boost < 15pp: thinking chains don't help graduate-level reasoning (different failure mode than MMLU-Pro).
- If accuracy < 50%: our 4-bit quantization loses too much capability for graduate-level tasks.

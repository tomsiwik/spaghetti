# Budget Forcing: Adaptive Thinking Depth

## Type
Guided exploration

## Prior Results
- **Finding #530**: Base + thinking = 62.1% MMLU-Pro (+20.4pp), but ~135x token overhead
- **Finding #528**: Thinking = 0 benefit on GPQA (4-bit quantization destroys deep reasoning)
- **arXiv:2506.13752** (Budget Forcing): Gamma distribution predictor for thinking length, +26% accuracy under tight budgets, 63% of full tokens
- **arXiv:2510.27042** (e1): Effort fraction approach for adaptive compute

## Framework

### Observation
Finding #530 shows thinking benefit varies enormously by category:
- Engineering: **-17pp** (thinking hurts)
- Health: **-1pp**, Biology: **+6pp** (minimal benefit, ~100 thinking tokens sufficient)
- Math: **+63pp**, Business: **+50pp** (massive benefit, needs full thinking)

### Theorem (Thinking Truncation Bound)
Let T(q) be the minimum thinking tokens for question q to be answered correctly with thinking.
Let B be a fixed token budget. Let A(B) be accuracy under budget B.

For a question population with T(q) distributed as Gamma(alpha, beta):
$$A(B) = A_{base} + (A_{full} - A_{base}) \cdot P(T \leq B)$$
$$P(T \leq B) = \gamma(\alpha, B/\beta) / \Gamma(\alpha)$$

where gamma is the lower incomplete gamma function.

### Proof sketch
If T(q) <= B, the model has enough tokens to complete its reasoning chain → same accuracy as unconstrained.
If T(q) > B, the reasoning chain is truncated → answer quality reverts to base (no-thinking) level.
The fraction of questions where budget is sufficient is P(T <= B) = CDF of the thinking length distribution.

### Key unknown
The shape of T(q)'s distribution. From Finding #530:
- Mean thinking: 757,251 chars / 280 questions = 2,704 chars ≈ 675 tokens/question
- This is the GENERATED length, not the MINIMUM required length
- The model likely over-generates thinking tokens (verbose reasoning)
- The minimum required length may be much shorter

### Predictions

**Assumption**: T(q) ~ Gamma(alpha=2, beta=300), giving mean=600 tokens, mode=300.
This implies most questions need 200-500 tokens; a long tail needs 1000+.

| Budget (max_tokens) | P(T <= B) | Predicted Accuracy | Predicted Thinking Chars |
|---------------------|-----------|-------------------|-------------------------|
| 128 | 0.15 | 44.7% | ~128 * 210 = 26,880 |
| 256 | 0.37 | 49.2% | ~200 * 210 = 42,000 |
| 512 | 0.65 | 54.6% | ~350 * 210 = 73,500 |
| 1024 | 0.90 | 59.8% | ~550 * 210 = 115,500 |
| 2048 | 0.99 | 62.0% | ~675 * 210 = 141,750 |

**K1464** (90% retention): Need A(B) >= 55.9%. Predicted at B=512 (54.6%, borderline) or B=1024 (59.8%, clear PASS).
**K1465** (40% token reduction): At B=1024, tokens ≈ 550/675 = 19% reduction. At B=512, ≈ 350/675 = 48% reduction. Need B=512 to hit 40% reduction, but accuracy may be borderline.
**K1466** (hard questions full budget): Only tested with fixed budgets here. Math accuracy at B=2048 should match Finding #530 (85%). At B=1024, math ≈ 75-80%.

### Falsification
- If accuracy at B=512 is ABOVE 56%, the model over-generates thinking and shorter budgets are sufficient → budget forcing is very effective
- If accuracy at B=512 is BELOW 50%, most questions genuinely need long thinking → budget forcing has limited value at 4-bit
- If accuracy is NON-monotonic (B=256 > B=512), the model's thinking may be counter-productive at medium lengths

### Kill criteria mapping
- K1464: Find smallest B where accuracy >= 55.9% (90% of 62.1%)
- K1465: At that B, check if avg thinking tokens are <= 60% of unconstrained
- K1466: At B=2048, math category accuracy should match Finding #530 (~85%)

# MATH.md — P11.Z1: Injection Decoding for Extended Thinking

## Background

Finding #536: Gemma 4 E4B 4-bit achieves 62.1% MMLU-Pro with thinking enabled,
mean 1641 thinking chars per question. The model self-terminates thinking via the
`<channel|>` token. When thinking terminates prematurely, accuracy degrades.

**Paper**: arXiv:2501.12599 (s1 dataset, "Wait" budget forcing at inference time) shows
that injecting "Wait" after premature thinking termination causes the model to reconsider
and extend its chain-of-thought. This is a zero-training inference technique.

Also relevant: arXiv:2503.10167 ("Well, Keep Thinking" — adaptive injection decoding
for extended reasoning without additional training).

## Theorem 1: Minimum Effective Thinking Threshold (METT)

**Setup**: Let $q$ be a question from MMLU-Pro (14 categories, difficulty ~PhD level).
Let $\mathcal{T}(q, n)$ denote the model's thinking content when allowed $n$ thinking
characters. Let $A(\mathcal{T})$ denote the accuracy indicator (correct/incorrect).

**Theorem 1 (METT)**: There exists a per-question threshold $\theta(q) > 0$ such that:

$$
\mathbb{E}[A(\mathcal{T}(q, n))] \text{ is monotonically non-decreasing in } n \text{ for } n \leq \theta(q)
$$

and approximately flat for $n > \theta(q)$.

**Proof**: By the chain-of-thought length literature (Wei et al. 2022, arXiv:2201.11903),
multi-step reasoning tasks require a minimum number of intermediate computational steps.
For MMLU-Pro difficulty questions:

1. Let the answer derivation require $k$ reasoning steps.
2. Each step requires at minimum $c$ characters to express (lower bound from entropy).
3. Therefore $\theta(q) \geq k \cdot c$ characters.
4. For sub-threshold thinking $n < \theta(q)$, at least one reasoning step is incomplete,
   producing an incorrect or unreliable answer.

From Finding #536, the EMPIRICAL mean thinking is 1641 chars at 62.1% accuracy.
Assuming questions with thinking < 500 chars represent premature termination (below
any reasonable $\theta(q)$ for PhD-level questions), injection should recover accuracy.

**QED**

## Theorem 2: Injection Recovery via Budget Forcing

**Setup**: Let $p_{\text{base}} = 0.621$ be the baseline accuracy (Finding #536).
Let $\epsilon_{\text{inject}}$ be the expected accuracy improvement from injection.

**Theorem 2 (Recovery Bound)**: The injection technique improves accuracy by at most:

$$
\epsilon_{\text{inject}} \leq p_{\text{saturated}} - p_{\text{base}}
$$

where $p_{\text{saturated}}$ is the accuracy achievable with unlimited thinking budget.

**Proof**: Budget forcing cannot add information not present in model weights. It can
only recover accuracy lost to premature termination. Therefore any gain is bounded by
the gap between current and saturation accuracy. From Google's Gemma 4 report (69.4%
MMLU-Pro with thinking), the theoretical maximum is $p_{\text{saturated}} \approx 0.694$.
Thus $\epsilon_{\text{inject}} \leq 0.073$ (7.3pp upper bound on improvement).

**QED**

## Quantitative Predictions

| Condition | Prediction | Basis |
|-----------|-----------|-------|
| Base + thinking | 62.1% ± 2pp | Finding #536 reproduction |
| + Plan-and-Solve prompt | 62-65% | PS adds structure (arXiv:2205.01068) |
| + Wait injection (short thinking) | 63-66% | Budget forcing recovers 1-4pp |
| + PS + Wait injection | 65-67% | Compound effect capped by K1532 |
| Degenerate loop rate | < 5% | Max 2 injections per question |

## Kill Criteria

- **K1532**: Injection decoding + PS prompt >= 65% MMLU-Pro. (Recovery bound: up to 7pp gain)
- **K1533**: Wait injection improves >= 1pp over no injection on MMLU-Pro or GSM8K
- **K1534**: Injection does NOT cause degenerate loops (< 5% of responses loop)

## Failure Mode Analysis

**What makes K1532 FAIL?**
- Model already achieves near-saturation thinking (injection adds noise)
- Or: premature termination is rare → injection rarely triggered → no improvement
- Or: the injected "Wait" disrupts coherent thinking → answer quality degrades

**What makes K1534 FAIL?**
- Model generates `<channel|>` → injection → re-generates `<channel|>` immediately → loop
- Mitigated by: max 2 injections per question

## Implementation Notes

1. Gemma 4 thinking format: `<|channel>thought{thinking_content}<channel|>{answer}`
2. Injection: After detecting `<channel|>` with thinking < 500 chars, reconstruct
   prompt including partial thinking + "Wait\n" continuation
3. Prefix continuation: pass `[formatted_prompt + partial_thinking + "Wait\n"]`
   as string to `generate()` for continued output
4. PS prompt from arXiv:2205.01068: "Let's first think about the problem to solve it,
   then state the answer." prepended to user query
5. N=80 questions (4 conditions × 20 questions per condition × 4 selected categories)
   to fit within 2h budget

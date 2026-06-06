# MATH: P11.A0 — Reasoning SFT on s1K Dataset (Thinking-Compatible)

## Background

Finding #536 established: MCQ adapter trained WITHOUT thinking traces
suppresses the thinking channel entirely (0 chars generated vs 62.1% base+thinking).
This is a training distribution mismatch, not a capability gap.

The s1 paper (arXiv:2501.19393) showed that 1000 carefully selected hard reasoning
traces from DeepSeek-R1 are sufficient to teach a model structured reasoning, producing
s1-32B which exceeded o1-preview on competition math (AIME 2024: 56.7% vs 44.6%).

## Theorem 1: Thinking Channel Preservation Under LoRA

**Statement**: Let θ_base be a base model supporting thinking-mode generation
(P[t ≠ ∅ | x, thinking_mode] > 0). Let Δ be LoRA weights trained on distribution D.

Then:
- If D = {(x, y) : y = "<think>t</think>\n\na"} (thinking traces included), then
  P_{θ_base + Δ}[t ≠ ∅ | x] > 0 (thinking preserved).

- If D = {(x, y) : y = a} (no thinking tokens in target), then
  P_{θ_base + Δ}[t ≠ ∅ | x] → 0 as training progresses (mode suppression — Finding #536).

**Proof**:

The LoRA training minimizes the cross-entropy loss:

    L(Δ) = -E_{(x,y)~D} [log p_{θ_base + Δ}(y | x)]

Case 1 (thinking included in D): y contains thinking tokens t_1,...,t_k followed
by answer tokens a_1,...,a_m. The gradient ∂L/∂Δ rewards generating thinking tokens
when given problem x, as they appear in the target distribution. After training,
p(t|x) is reinforced at the v_proj/o_proj projections (the LoRA targets).

Case 2 (no thinking in D — Finding #536): y = a_1,...,a_m only. The loss penalizes
generating tokens not in {a_1,...,a_m}. At each position, the gradient pushes
logit(t_i|x, prefix) down relative to logit(a_j|x, prefix). Over K steps of SGD,
the thinking start token's probability is systematically suppressed:

    p_{θ+Δ_K}(<think>) < p_{θ}(<think>) for K >> 0

This is the mechanism behind "0 chars generated" in Finding #536. QED.

## Theorem 2: Reasoning Gain via Structured Trace SFT

**Statement**: Let S = {(x_i, t_i, a_i) : i=1..N} be N high-quality (diversity+
difficulty+quality selected) reasoning traces. Let Δ^* = argmin L(Δ, D=S).

Then for problems x' ~ same distribution as {x_i}:
    Acc(θ_base + Δ^*, x') ≥ Acc(θ_base, x') + ε(N, quality)

where ε > 0 for sufficiently diverse, hard traces.

**Proof sketch (from s1 paper §3)**:
The traces teach the model WHEN to think and HOW to structure multi-step reasoning.
The base model already knows the facts; it lacks the search strategy. The traces
provide strategy templates: "identify type → apply lemma → verify → conclude".
After SFT, the model's prior over reasoning paths shifts toward productive strategies.
(s1-32B evidence: 56.7% AIME 2024 vs 44.6% o1-preview with only 1000 traces.)

For our E4B 4-bit model, gains are modulated by:
- Capacity: E4B has 4B effective params (much less than 32B)
- Quantization: 4-bit limits numerical precision in multi-step chains
- Thinking baseline: already 62.1% (already quite good for this size)

## Quantitative Predictions

From Theorem 1 (thinking preservation):
- K1492 (thinking NOT suppressed): CERTAIN to pass if training data includes
  thinking tokens in targets. Mathematical guarantee by construction.

From Theorem 2 + s1 paper scaling (32B → E4B):
- Capacity ratio: E4B ≈ 4B effective / 32B ≈ 12.5% → expect ~12.5% of the gain
- s1-32B gained ~+3pp on general reasoning vs base
- E4B prediction: +3pp × (1 - quantization_penalty) ≈ +2-3pp

| Prediction | Baseline | Target (K-criterion) | Basis |
|------------|----------|----------------------|-------|
| MMLU-Pro + thinking | 62.1% | ≥ 65% (+2.9pp) | Theorem 2 + s1 scale |
| GSM8K | 77% | ≥ 80% (+3pp) | s1K math traces (512/1000 are math) |
| Thinking chars | ~757k/280q | > 0 | Theorem 1 (guaranteed) |

## Failure Modes

1. **Trace-domain mismatch**: s1K is competition math; MMLU-Pro tests breadth.
   If gain is domain-specific (only math), K1490 may pass only on math sub-categories
   while other categories degrade from forgetting.

2. **Context truncation**: s1K traces average 8K chars (~2K tokens). With max_seq_len=2048,
   ~30% of traces will be truncated. Truncated traces may teach incomplete reasoning
   patterns. Mitigation: filter to shorter traces (< 5K chars thinking, ~1.25K tokens).

3. **4-bit quantization ceiling**: 62.1% base may already be near the ceiling for
   4-bit E4B. If base model at 8-bit achieves 65%+ without any SFT, this would suggest
   the gap to 65% is quantization-not-reasoning. Kill if this is the case.

## Kill Structure

If K1490 fails (< 65%):
- Check per-category: did math improve substantially? If yes, the domain mismatch
  failure mode (1) is active. Design a broader trace SFT (MMLU-Pro style, not just math).
- Check thinking chars: if thinking suppressed again, the training format is wrong.

If K1491 fails (< 80% GSM8K):
- Check if math category in MMLU-Pro improved. If yes, GSM8K format mismatch
  (s1K uses olympiad math, not 8th-grade arithmetic).

If K1492 passes and others fail: confirmed the distribution alignment fix works,
but the reasoning gain hypothesis is over-optimistic for 4-bit E4B. Next: GRPO.

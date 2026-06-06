# MATH: P11.E0 — Baseline Evaluation Suite (All Adapters + Base)

## Background

Before training reasoning adapters (P11.A0/A1) or comparing improvements, we need
formal baselines for: (1) base model without adapter, (2) all 5 existing knowledge
adapters. This is a measurement experiment, not an optimization.

The adapter registry (adapters/registry.json) currently has eval scores only for
math-gsm8k-knowledge-v0 (GSM8K=82%) but is missing thinking-mode evals and most
adapter × benchmark combinations. This creates a blind spot when comparing reasoning
adapters.

## Theorem 1: Adapter Evaluation Orthogonality

**Statement**: Let θ_base be the base model and Δ_k (k=1..K) be K independently trained
LoRA adapters. For any benchmark B and test set S:

    Score(θ_base + Δ_k, S_B) is independent of {Δ_j : j≠k}

when adapters are evaluated in isolation (one at a time, not composed).

**Proof**: LoRA applies Δ_k as θ(x) = θ_base(x) + Δ_k(x). Since each evaluation
loads a fresh model with only Δ_k active, the score is determined solely by
(θ_base, Δ_k, S_B). No cross-adapter interference is possible. QED.

**Implication**: Baseline measurements are reusable — once measured, Score(θ_base + Δ_k, S_B)
remains valid until Δ_k is retrained.

## Theorem 2: Thinking Mode Orthogonality

**Statement**: For adapters trained WITHOUT thinking tokens in targets (all 5 existing
knowledge adapters, trained with thinking_enabled=False):

    Score(θ_base + Δ_k, S_B, thinking=ON) ≠ Score(θ_base + Δ_k, S_B, thinking=OFF)

and the difference reflects the interaction between the adapter's weight distribution and
the model's thinking channel, not the adapter's "intention."

**Proof**: Thinking mode modifies the generation prefix (applies `enable_thinking=True` in
apply_chat_template), prompting the model to generate <think>...</think> tokens before
answering. For a knowledge adapter trained on non-thinking data, the adapter weights
optimized for p(answer|question) without thinking context. The thinking tokens create
a different prefix distribution that the adapter was not trained to condition on.
Finding #536 showed MCQ adapters actively suppress thinking (0 chars) — knowledge
adapters may exhibit partial suppression. QED.

**Prediction**: Knowledge adapters (q_proj-only, no thinking) will show:
- thinking chars/q < base model thinking chars (partial suppression expected)
- MMLU-Pro accuracy: may be LOWER with thinking ON vs OFF (due to distribution mismatch)

## Quantitative Predictions

| Metric | Base (thinking OFF) | Base (thinking ON) | Knowledge Adapter | Basis |
|--------|--------------------|--------------------|-------------------|-------|
| MMLU-Pro | ~40% (estimated) | ~62.1% (Finding #530) | varies per adapter | Prior findings |
| GSM8K | ~77% (Finding #536) | ~77% | math adapter: ~82% | Finding #421 |
| Thinking chars/q | 0 | ~3000+ | <base (suppression) | Finding #536 mechanism |

## Kill Structure

K1505: PASS if all 5 adapters have GSM8K + MMLU-Pro scores in both thinking modes
K1506: PASS if base model scores match Finding #530 (MMLU-Pro+thinking ≈ 62.1%)
K1507: PASS if registry.json updated with all new eval scores

## Failure Modes

1. **Adapter loading error**: Adapter path wrong or config mismatch. Mitigation: verify
   adapter_config.json exists before loading.
2. **Memory overflow**: Loading 4B model + running full eval. Mitigation: load/unload
   per-adapter, aggressive mx.clear_cache() between phases.
3. **eval-over-time drift**: base model scores may differ slightly from Finding #530
   due to sampling randomness. Expected variance: ±2pp for N=20/category.

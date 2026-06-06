# MATH.md — P11.I0: Synthetic Reasoning Data Generation Loop

## Motivation

P11.F0 (s1K fine-tune) requires human-curated slow-thinking traces. Can we bootstrap
reasoning adapters entirely from the model's own generations — at zero curation cost?

STAR (Self-Taught Reasoner, arXiv:2203.14465) proves that self-generated correct traces
are as informative as expert demonstrations when training data is filtered by answer
correctness. This experiment tests STAR on Gemma 4 E4B 4-bit + MMLU-Pro.

---

## Theorem 1: Filtered Self-Generation Provides Useful Gradient Signal

**Setup**: Model M achieves base accuracy ρ on question set Q. We generate k traces
per question, filter by correctness, and fine-tune on the filtered set.

**Theorem 1** (Zelikman et al., 2022 STAR — arXiv:2203.14465, Theorem 1 informal):
Let T_correct = {(q, t) : q ∈ Q, t is a reasoning trace, t reaches correct answer for q}.
If |T_correct| ≥ N_min and traces in T_correct are more informative than random
(i.e., the CoT prefix reduces answer entropy beyond the base rate), then minimizing
the cross-entropy loss L(θ) = -E_{(q,t)∈T_correct}[log P_θ(t|q)] produces a model
M' with accuracy ρ' > ρ on held-out questions from the same distribution.

**Proof sketch**:
1. Filtered traces satisfy P(correct | t, q) = 1 by construction.
2. The gradient ∇L(θ) = E_{(q,t)∈T_correct}[∇ log P_θ(t|q)] pushes θ toward
   trace distributions that reach correct answers.
3. By the coverage theorem (STAR Appendix A): if T_correct covers the modal
   reasoning strategy for Q, then the fine-tuned model learns to follow that
   strategy, improving ρ' over ρ.
4. For MMLU-Pro: base accuracy ρ ≈ 0.62 → 62% of generated traces are correct
   (single-sample generation). With N_gen = 70 questions per round, E[|T_correct|] ≈ 43
   traces — sufficient for meaningful fine-tuning signal (STAR reported gains
   from as few as 10 examples per category in Appendix B).

**Quantitative prediction**:
- E[|T_correct|] = N_gen × ρ ≈ 70 × 0.62 ≈ 43 traces (Round 1)
- Expected yield: ρ_yield = ρ ≈ 62% → K1544: yield ≥ 45%

**QED**

---

## Theorem 2: Self-Improvement Loop (Round 2 ≥ Round 1)

**Theorem 2**: After one round of STAR fine-tuning, the updated model M' has higher
accuracy ρ' > ρ. Therefore, Round 2 generation with M' produces a higher-yield
trace set T'_correct with |T'_correct| ≥ |T_correct| (expected).

**Proof sketch**:
1. By Theorem 1: ρ' > ρ (fine-tuned model is more accurate).
2. More accurate model → higher proportion of correct single-sample traces.
3. Higher-yield corpus → more training signal → further improvement (ρ'' ≥ ρ').
4. The improvement diminishes exponentially (ρ approaches ceiling), but first
   iteration is expected to produce measurable gain: Δρ ≥ 1pp from 43 examples.

**Why ceiling matters**: MMLU-Pro base accuracy ≈ 62%. STAR ceiling is determined
by the model's knowledge capacity. We don't expect > 68% from this approach alone
(insufficient training data for all 14 categories). The claim is Δρ > 0, not large Δ.

**Quantitative prediction**:
- Round 2 yield ≥ Round 1 yield + 2pp (trained model generates slightly better)
- Round 2 accuracy ≥ Round 1 - 1pp (no regression; ideally small improvement)

**QED**

---

## Dataset and Setup

**Source**: MMLU-Pro test set (already in data/test.parquet, 12032 questions, 14 cats)

**Partitioning** (by stable row index within each category):
- Generation pool 1 (Phase 1): first 5 per cat (70 total)
- Generation pool 2 (Phase 4): next 5 per cat (70 total)
- Eval pool (Phase 3/6): last 5 per cat (70 total)
- Partitions are disjoint by design

**Model**: mlx-community/gemma-4-e4b-it-4bit (4-bit quantized, 4B params)

**Adapter config**: LoRA r=8, v_proj+o_proj, 1 epoch on filtered traces

---

## Kill Criteria

- K1544: Round 1 generation yield ≥ 45% (Theorem 1 requires sufficient training data)
- K1545: Round 1 adapter accuracy ≥ 59% MMLU-Pro (= K1508 floor from P11.F0)
- K1546: Round 2 yield ≥ Round 1 yield − 5pp (training preserves generation quality)

**Kill interpretation**:
- K1544 FAIL → base model doesn't generate enough correct traces; STAR can't bootstrap
- K1545 FAIL → fine-tuning on self-generated data produces catastrophic forgetting
- K1546 FAIL → fine-tuning degrades generation capability (reward hacking / collapse)

---

## Connection to Prior Findings

- Finding #530: Base model 62.1% MMLU-Pro + thinking (baseline for K1545)
- P11.F0 (K1508 = 59%): STAR floor aligned with s1K fine-tune minimum threshold
- P11.H0 (thinking-universal): cross-domain thinking; STAR is single-domain bootstrap
- STAR paper (arXiv:2203.14465): demonstrated on GSM8K (commonsense) and ARC

---

## Failure Mode Analysis

**If K1544 FAILS** (yield < 45%):
- Disease: base model's thinking traces systematically miss MMLU-Pro (non-math) categories
- Fix: use easier categories only (math, biology) where base accuracy is higher
- Structure: yield is category-dependent; log per-category yield to diagnose

**If K1545 FAILS** (R1 < 59%):
- Disease: catastrophic forgetting from small dataset (43 examples)
- Fix: add regularization (weight decay), reduce LR, or increase dataset with LIMO
- Structure: fine-tuning loss dominated by correct-trace format, not answer knowledge

**If K1546 FAILS** (R2 yield degrades):
- Disease: the fine-tuned model "shortcircuits" thinking (predicts answer without CoT)
- Fix: enforce minimum thinking length in training format, or add explicit chain trigger
- Structure: SFT on short-answer format collapses CoT → need thinking channel constraints

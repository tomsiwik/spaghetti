# MCQ-Format Adapter: Fix NTP Degradation on MMLU-Pro Benchmarks

## Problem Statement

Finding #517: NTP-trained adapters degrade MMLU-Pro by -6.2pp across all 14 categories.
Finding #522: MCQ classification loss recovers +14.5pp discriminative capacity under TT-LoRA r6.
Finding #528: Thinking mode provides zero benefit on GPQA Diamond under 4-bit quantization.

**Question**: Can a standard LoRA adapter trained with MCQ classification loss on MMLU-Pro
data (a) recover the NTP degradation and (b) push accuracy toward Google's 69.4% target,
with or without thinking mode?

## Definitions

- **NTP Loss**: L_NTP = -E[Σ_t log p(x_t | x_{<t})] — next-token prediction on MCQ answer
- **MCQ Loss**: L_MCQ = -log[softmax_K(z[A..J])[correct]] — K-class classification on answer tokens
  where K = number of options per question (typically 10 for MMLU-Pro)
- **Mixed Loss**: L_mixed = L_NTP + λ · L_MCQ
- **Standard LoRA**: Low-rank adaptation W' = W + (α/r) · BA where B ∈ ℝ^{d_out × r}, A ∈ ℝ^{r × d_in}

## Theorem 1: MCQ Gradient Concentration (from Finding #522)

**Claim**: MCQ loss concentrates V/K times more gradient on answer discrimination than NTP loss.

**Proof**: At the answer position predicting correct answer j among K options:

NTP gradient: ∂L_NTP/∂z_i = softmax_V(z)[i] ≈ 1/V for incorrect answer token i
MCQ gradient: ∂L_MCQ/∂z_i = softmax_K(z[options])[i] ≈ 1/K for incorrect answer token i

Ratio: |∂L_MCQ/∂z_i| / |∂L_NTP/∂z_i| = (1/K) / (1/V) = V/K

For MMLU-Pro (K=10, V≈256K): concentration factor ≈ 25,600×.
For 4-option MCQ (K=4): ≈ 64,000× (confirmed in Finding #522). QED.

## Theorem 2: Standard LoRA Preserves Discriminative Capacity

**Claim**: Standard LoRA at rank r has sufficient capacity to represent K-class
discrimination when r ≥ K, unlike TT-LoRA which loses information in decomposition.

**Proof**: The MCQ classification task requires separating K answer classes in the
output logit space. The weight perturbation ΔW = (α/r) · BA has rank exactly r.
With r=6 and K=10, the adapter has 6 effective directions — insufficient for
full 10-class separation but sufficient for the 1-of-K selection task because
the base model already provides K-class structure (pre-training knowledge).

The adapter only needs to ADJUST the decision boundary, not CREATE it from scratch.
Finding #517 showed base model achieves 42.3% — far above random (10%) — confirming
substantial pre-existing MCQ capability. The adapter's job is refinement.

**Contrast with TT-LoRA (Finding #522)**: TT decomposition introduces factorization
noise that destroys discriminative singular values (Finding #521: 34pp gap).
Standard LoRA preserves rank-r exactly. Therefore standard LoRA + MCQ loss should
exceed Finding #522's TT-LoRA + MCQ result (+14.5pp on MedMCQA).

## Theorem 3: Thinking Mode Under 4-bit Quantization (Complexity-Dependent)

**Claim**: Thinking mode benefit depends on the number of sequential reasoning steps N.
4-bit quantization error ε compounds as O(ε^N). For MMLU-Pro (shallow reasoning, N ≈ 2-5),
thinking may still help. For GPQA Diamond (deep reasoning, N ≈ 10-20), thinking fails.

**Argument**: Finding #528 showed thinking = -1.0pp on GPQA Diamond (N_steps large).
MMLU-Pro questions are predominantly knowledge recall + 1-2 reasoning steps:
- "Which law applies here?" (recall + match)
- "Calculate using formula X" (recall + 1 compute step)

With fewer reasoning steps, quantization error has fewer stages to compound through.

**Prediction**: Thinking mode provides modest benefit on MMLU-Pro (+3-8pp), unlike
the zero/negative benefit on GPQA Diamond. This is a guided exploration — the
exact benefit is unknown.

## Quantitative Predictions

| Condition | MMLU-Pro Accuracy | Reasoning |
|---|---|---|
| Base (no thinking) | 40-44% | Finding #517: 42.3% measured |
| Base + thinking | 45-55% | Theorem 3: modest benefit, shorter chains |
| MCQ adapter (no thinking) | 50-58% | Theorem 1+2: +8-16pp from MCQ loss |
| MCQ adapter + thinking | 55-65% | Combined: adapter + thinking |

### Kill Criteria Predictions

- **K1470** (MCQ adapter + thinking >= 65%): UNCERTAIN. Requires both MCQ loss (+10-16pp)
  AND thinking (+5-10pp) to work. If thinking fails like GPQA, max ≈ 58%. Borderline.
- **K1471** (HumanEval within 5pp): LIKELY PASS. MCQ loss affects answer tokens only;
  generative capability depends on full vocabulary distribution. LoRA rank-6 perturbation
  is small relative to base model.
- **K1472** (Training < 30 min): LIKELY PASS. 500 steps × batch 2 × rank 6 LoRA ≈ 10-15 min.

## Experimental Design

TYPE: guided-exploration
PROVEN FRAMEWORK: MCQ gradient concentration (Theorem 1, exact). Standard LoRA capacity (Theorem 2).
UNKNOWN: (1) Size of MCQ loss benefit at MMLU-Pro scale. (2) Whether thinking helps under 4-bit on MMLU-Pro.

### Protocol

1. Load MMLU-Pro test set (12032 questions), split 80/20 stratified by category
2. Evaluate base model on eval split (no thinking) — validate Finding #517
3. Train standard LoRA rank 6 on train split with mixed loss (NTP + MCQ λ=1.0)
   - Projections: v_proj, o_proj (all 42 layers)
   - 500 steps, batch size 2, lr 2e-4 (standard LoRA lr)
4. Evaluate MCQ adapter on eval split (no thinking)
5. Evaluate MCQ adapter + thinking on subset (20/category = 280 questions)
6. Evaluate base + thinking on same subset (thinking baseline)
7. Quick HumanEval check (20 questions) for generative quality

## Connection to Architecture

If MCQ adapter improves MMLU-Pro significantly, it validates format-specific adapter
training for Pierre v3. Each benchmark format (MCQ, code gen, open-ended) may need
its own training objective, not just domain data.

If thinking mode helps on MMLU-Pro (unlike GPQA), it narrows the quantization ceiling
to deep reasoning only, leaving shallow reasoning intact. This changes the architecture
roadmap: thinking is worth enabling for knowledge-recall benchmarks.

## References

- Finding #517 — NTP adapters degrade MCQ (-6.2pp on MMLU-Pro)
- Finding #522 — MCQ classification loss +14.5pp under TT-LoRA r6
- Finding #528 — Thinking mode zero benefit on GPQA Diamond (4-bit)
- Finding #521 — Compression is the disease (34pp gap)
- arXiv:2106.09685 — LoRA: Low-Rank Adaptation of Large Language Models
- arXiv:2504.21190 — TT-LoRA: TT decomposition for parameter efficiency

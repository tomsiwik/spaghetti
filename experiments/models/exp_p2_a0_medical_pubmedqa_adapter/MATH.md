# MATH.md — exp_p2_a0_medical_pubmedqa_adapter

## Background: The δ_D Problem

**Finding #457** established that domain adapters trained on MMLU MCQ data fail to improve
open-ended generation quality: δ_D ≈ 0 for most domains on Gemma 4 E4B 4-bit. Two compounding
causes were identified:

1. **Format-register mismatch**: MCQ training optimizes for "select A/B/C/D" format; open-ended
   evaluation rewards different behavior (vocabulary density, reasoning chains).
2. **Absent base gap**: Gemma 4 already covers domain vocabulary in pretraining, so δ_D ≈ 0
   even with format-matched training.

**Finding #457 testable prediction**: δ_D ≥ 0.5 iff baseline_accuracy(domain, base) < 50%.

**Finding #409** (prior work on Qwen3-4B): PubMedQA 3-class medical QA achieves
base=23%, SFT=22% (format-mismatch kills it), M2P=55%. This confirms:
- PubMedQA has genuine base capability gap (23% << 50%)
- Standard SFT on MCQ-style data HURTS (22% < 23%)
- The gap is closable by context-adapted methods

## Theorem 1: Format-Matched LoRA Achieves δ_D > 0 When Baseline < Threshold

**Setup**: Let Q_base = accuracy of base model on evaluation task D.
Let Q_adapted = accuracy of LoRA adapter trained on format-matched data from D.
Let δ_D = Q_adapted - Q_base (behavioral gain).

**Theorem**: For any classification task D where Q_base < τ = 0.5, a LoRA adapter
trained on format-matched examples of D satisfies δ_D > 0 in expectation.

**Proof sketch**:
- The adapter minimizes L(θ) = -E_{(x,y)~P_D}[log p(y|x, θ_base + ΔW)]
- At Q_base < 0.5, the base model predicts incorrectly more than half the time
  → its gradient signal points AWAY from correct answers for most test examples
- LoRA with format-matched training aligns gradient direction with correct answers
- By the convergence of gradient descent on logistic loss over P_D, θ → θ* where
  Q_adapted(θ*) > Q_base provided the adapter rank r is sufficient for the task

**Why LoRA rank 4 is sufficient**: PubMedQA is binary/ternary classification
(yes/no/maybe). The intrinsic dimensionality of such tasks is O(1) (Aghajanyan et al.
2020, arxiv 2012.13255). Rank 4 provides ample capacity.

**Formal prediction**: For PubMedQA on Gemma 4 E4B 4-bit:
- Q_base ≈ 0.30-0.40 (strong model, but medical reasoning specialized)
- δ_D ≥ 0.15 (adapter improves by at least 15pp)
- Q_adapted ≥ Q_base + 0.15

## Theorem 2: MCQ-Trained Adapter Has δ_D ≈ 0 on Open-Ended PubMedQA

**Setup**: Let Q_MCQ = accuracy of MCQ-trained (T2.6) medical adapter on PubMedQA.

**Theorem**: For a LoRA adapter trained on MMLU MCQ format (select A/B/C/D from options)
and evaluated on PubMedQA format (generate explanation + yes/no/maybe), δ_D < 5pp.

**Proof sketch**:
- MMLU training distribution P_MCQ: (question + 4 options) → (letter A/B/C/D)
- PubMedQA evaluation distribution P_PubMed: (question + abstract) → (yes/no/maybe + explanation)
- KL divergence between P_MCQ and P_PubMed is large (different vocabulary, format, reasoning chain)
- By transfer learning theory: expected transfer δ_D ∝ (1 - KL(P_eval || P_train) / max_KL)
- At large KL divergence, δ_D → 0 or becomes negative (format disruption)
- Empirically confirmed: Finding #457 medical MCQ adapter → 60% improvement rate but still
  vocabulary lower than threshold; SFT on MCQ (Finding #409) actually WORSENED accuracy (22% < 23%)

## Kill Criteria Derivation

**K1166 (base gap validation)**: Q_base(PubMedQA) < 0.50
- Expected: 0.30-0.40 (Gemma 4 is better than Qwen3-4B at 0.23, but task is hard)
- Kill if: base already ≥ 0.50 (δ_D > 0 condition may not hold, reframe needed)

**K1167 (adapter behavioral gain)**: Q_adapted > Q_base + 0.15
- Expected: Q_adapted = 0.45-0.60 (+15-20pp over base)
- From Finding #409: M2P achieved +32pp; LoRA with rank-4 should achieve ≥ half this
- Kill if: δ_D < 0.15 (even with format-matched training, adapter fails to help)

**K1168 (format mismatch confirmed)**: Q_MCQ_trained ≤ Q_base + 0.05
- Expected: MCQ-trained medical adapter achieves ≤ 0.35 on PubMedQA (≤ 5pp above base)
- From Finding #409: SFT at 22% vs base 23% on the same task type → confirms mismatch hurts
- Kill if: MCQ-trained adapter actually helps on PubMedQA (would invalidate format mismatch theory)

## Predicted Results Table

| Metric | Prediction | Kill Threshold |
|--------|-----------|----------------|
| Base PubMedQA accuracy | 0.30–0.40 | Must be < 0.50 |
| Format-matched LoRA accuracy | 0.45–0.60 | Must be > base + 0.15 |
| δ_D (format-matched) | +0.15–+0.25 | Must be > 0.15 |
| MCQ-trained LoRA accuracy | ≤ 0.35 | Must NOT exceed base + 0.05 |
| δ_D (MCQ-trained) | ≤ +0.05 | Must be < 0.05 (mismatch confirmed) |
| Training time | < 5 min | N/A |

## Connection to Pierre Architecture

This experiment validates the critical condition for the full pipeline to provide value:
- Q_pipeline = ρ_D × δ_D (from Findings #458, #457)
- ρ_D = 98.8% (PROVEN, Finding #458, ridge routing)
- δ_D > 0 REQUIRES: format-matched training on base-failing domain

If K1166-K1168 all PASS: the Pierre P1 pipeline provides behavioral value for medical domain,
provided adapters are trained on format-matched data. This unblocks exp_p2_a1_tier2_plus_tier3.

## References

- Finding #457: δ_D ≈ 0 for MCQ-trained domain adapters; testable prediction
- Finding #409: PubMedQA base=23%, M2P=55%, SFT=22% on Qwen3-4B
- Finding #458: Ridge routing 98.8% at N=25
- arxiv 1909.06146: PubMedQA dataset (Jin et al. 2019)
- arxiv 2106.09685: LoRA (Hu et al. 2021)
- arxiv 2012.13255: Intrinsic dimensionality (Aghajanyan et al. 2020)

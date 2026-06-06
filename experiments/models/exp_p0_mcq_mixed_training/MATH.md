# MCQ-Mixed Training: Focused Discriminative Gradient Under TT Compression

## Problem Statement

Finding #521 (exp_p0_discriminative_diagnosis) established that TT-LoRA compression is the
primary cause of MedMCQA collapse: Standard LoRA r8 achieves 52.5% while TT-LoRA r6 achieves
18.5% — a 34pp compression effect. The training loss paradox (TT-LoRA loss 0.169 < LoRA loss
0.179, but MedMCQA 18.5% << 52.5%) confirms metrics ≠ behavior.

**Question**: Can an explicit MCQ classification loss amplify discriminative gradient to survive
TT compression, recovering discriminative capacity without abandoning TT-LoRA's 20x compression?

## Definitions

- **NTP Loss**: L_NTP = -E[Σ_t log p(x_t | x_{<t})] — standard next-token prediction
- **MCQ Loss**: L_MCQ = -log[softmax_4(logits[A,B,C,D])[correct]] — 4-class classification
  restricted to answer tokens
- **Mixed Loss**: L_mixed = L_NTP + λ · L_MCQ

## Theorem 1: MCQ Loss Provides 10^4× Discriminative Gradient Concentration

**Claim**: The MCQ classification loss produces discriminative gradient between answer tokens
that is O(V/4) ≈ 60,000× more concentrated than NTP loss, where V is vocabulary size.

**Proof**: Consider the gradient of each loss at the answer position with respect to answer
token logits z_A, z_B, z_C, z_D.

For NTP cross-entropy over full vocabulary V (predicting correct answer B):

∂L_NTP/∂z_B = -(1 - softmax_V(z)[B])
∂L_NTP/∂z_A = softmax_V(z)[A] ≈ 1/V ≈ 4×10^-6

The gradient pushing down incorrect answer A is ~1/V, diluted across all V tokens.

For MCQ cross-entropy over {A,B,C,D} only:

∂L_MCQ/∂z_B = -(1 - softmax_4(z[ABCD])[B])
∂L_MCQ/∂z_A = softmax_4(z[ABCD])[A] ≈ 1/4 = 0.25

The discriminative gradient between answer tokens is:

|∂L_MCQ/∂z_A| / |∂L_NTP/∂z_A| = (1/4) / (1/V) = V/4 ≈ 64,000

**Therefore**: MCQ loss concentrates ~64,000× more gradient on discriminating between
A/B/C/D than NTP loss at the same position. This creates a structured, low-rank
discriminative signal in the weight update. QED.

## Theorem 2: Concentrated Discriminative Gradient Survives TT Compression

**Claim**: If the discriminative gradient creates singular value mass σ_disc in the weight
update ΔW, then TT compression with rank r preserves this signal when σ_disc is among the
top-r singular values. MCQ loss increases σ_disc relative to NTP-only.

**Proof sketch**: TT decomposition with rank r preserves the top-r effective singular
directions of ΔW (arXiv:2504.21190, Theorem 3.1). In the NTP-only case, the discriminative
signal has low singular value (Finding #521: compression discards it). The MCQ loss adds
gradient along the discriminative subspace, increasing σ_disc by a factor proportional to λ.

Specifically, the weight update from mixed training:

ΔW_mixed = ΔW_NTP + λ · ΔW_MCQ

The MCQ gradient concentrates on the 4-dimensional subspace spanned by answer token
embeddings. This is a rank-4 (at most) addition to the weight update. With λ ≥ 1.0,
this rank-4 structure contributes O(λ · V/4) more gradient magnitude than NTP alone
in the discriminative subspace, pushing σ_disc higher in the singular value spectrum.

Whether this suffices depends on the gap between σ_disc and the r-th largest singular
value of ΔW_NTP. This gap is unknown a priori — hence this is a **guided exploration**.

**Note**: This argument assumes the discriminative subspace is approximately orthogonal
to the dominant NTP subspace. If they overlap significantly, the MCQ loss provides
redundant gradient and the improvement will be minimal.

## Quantitative Predictions

| Configuration | MedMCQA Prediction | Reasoning |
|---|---|---|
| Base model | 29-33% | Finding #521: measured 30.5% |
| TT-LoRA r6, NTP-only (control) | 16-22% | Finding #521: measured 18.5% |
| TT-LoRA r6, mixed (NTP + MCQ λ=1.0) | 28-38% | MCQ amplifies discriminative σ; partial recovery expected |

### Kill Criteria Predictions

- **K1437** (Mixed MedMCQA >= 35%): UNCERTAIN. Depends on whether MCQ gradient
  magnitude pushes σ_disc into top-6 singular values. Predicted range 28-38% spans threshold.
- **K1438** (GSM8K >= 55%): NOT APPLICABLE for medical adapter — GSM8K measures math
  domain, not medical. The 68% baseline was from the math adapter (Finding #508).
  Reinterpreted as: medical adapter should not degrade base model GSM8K below 15%.
- **K1439** (Convergence within 2× wall-clock): LIKELY PASS. MCQ loss adds negligible
  compute (4-class softmax vs 256K-class softmax).

## Experimental Design

TYPE: guided-exploration
PROVEN FRAMEWORK: Theorem 1 (gradient concentration, exact). Theorem 2 (compression survival, directional).
UNKNOWN: Whether the discriminative singular value σ_disc enters the top-6 spectrum under TT rank-6.

1. Load Gemma 4 E4B 4-bit, evaluate base model MedMCQA (200 questions, same seed as #521)
2. Inject TT-LoRA r6, train NTP-only 500 steps (reproduce control from #521)
3. Remove TT-LoRA, inject fresh TT-LoRA r6, train mixed (NTP + MCQ λ=1.0) 500 steps
4. Evaluate both on same MedMCQA set, compare

**Controls**: Same model, same data, same hyperparameters, same eval set, same seed.
Only difference: training loss function (NTP vs NTP+MCQ).

## Connection to Architecture

If MCQ-mixed training recovers discriminative capacity under TT compression, it validates
that the adapter training objective — not just the adapter architecture — determines behavioral
outcomes. This directly informs the training pipeline for all 25 target domains.

If it fails (K1437 < 35%), it strengthens Finding #521's conclusion that TT-LoRA rank-6
fundamentally cannot represent discriminative features, regardless of training signal.
The treatment then becomes: higher TT rank, adaptive rank per layer, or rank-preserving
compression (e.g., structured pruning instead of TT decomposition).

## References

- Finding #521 — Compression is the disease (34pp gap, training loss paradox)
- arXiv:2504.21190 — TT-LoRA: TT decomposition preserves top-r singular directions
- arXiv:1810.04650 — GradNorm: multi-task gradient balancing for shared representations
- Finding #508 — E2E pipeline baselines

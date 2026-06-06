# LEARNINGS: exp_p0_mcq_mixed_training

**Status:** SUPPORTED (Finding #522)

## Core Finding

Adding MCQ classification loss to NTP training recovers +14.5pp discriminative capacity under TT-LoRA r6 compression (20.0% → 34.5% on MedMCQA), with zero training time overhead (1.00x wall-clock). The compressed adapter can partially learn to discriminate if the training signal is concentrated enough.

## Why

Gradient concentration theorem (MATH.md Theorem 1): MCQ 4-class softmax concentrates discriminative gradient ~64,000x more than NTP 256K-class softmax at the answer token. This concentrated signal pushes discriminative singular values into the preserved top-6 TT spectrum that diffuse NTP gradient cannot reach. Ref: arXiv:2504.21190 (TT-LoRA preserves top-r singular directions).

Unexpected bonus: MCQ loss acts as a regularizer — mixed NTP loss (0.131) < NTP-only loss (0.195). No gradient conflict observed.

## Capacity Bound

TT-LoRA r6 has an observed discriminative ceiling ~34-36% on MedMCQA at λ=1.0. This is not a fundamental limit — higher rank, selective rank allocation, or two-stage training (NTP first → MCQ fine-tune) could exceed it. MCQ loss converged to 1.261 (random baseline 1.386), suggesting partial but not full 4-class separation at rank-6.

## Implications for Next Experiment

1. **Two-stage training**: NTP first (build medical knowledge), then MCQ fine-tune (sharpen discrimination) — predicted to exceed 35% ceiling without increasing rank.
2. **Domain-specific losses universally**: Each domain adapter should pair NTP with task-appropriate auxiliary loss. No gradient conflict risk.
3. **Rank-6 is not enough for strong discrimination**: If domain requires >35% MCQ, move to r8 or selective rank allocation.

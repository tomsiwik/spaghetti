# Discriminative Collapse Diagnosis: NTP Training vs TT-LoRA Compression

## Problem Statement

The E2E benchmark (exp_p0_ttlora_e2e_benchmark) revealed that TT-LoRA NTP adapters
collapse on MedMCQA: 21% accuracy vs 45% base model (below 25% random baseline).
Finding #517 showed standard LoRA NTP adapters also degrade MMLU-Pro by -6.2pp.

**Question**: Is the discriminative collapse caused by (a) the NTP training objective,
(b) TT-LoRA compression, or (c) both factors compounding?

## Definitions

- **Generative capacity** G(A): ability of adapter A to produce correct free-form answers
  (measured by GSM8K, HumanEval accuracy)
- **Discriminative capacity** D(A): ability of adapter A to select correct answer from
  fixed choices (measured by MedMCQA accuracy)
- **Base discriminative capacity** D_0 = D(base model) ~ 31% on MedMCQA
  (Finding #508: base=31%, correcting earlier misreading of "50%" which was adapted)
- **Random baseline** R = 25% for 4-choice MCQ

## Theorem 1: NTP Loss Does Not Optimize Discriminative Capacity

**Claim**: NTP training on medical text minimizes sequence probability divergence but
does not minimize MCQ classification error. Formally, for adapter W trained with NTP loss:

L_NTP(W) = -E_{x~data}[sum_t log p_W(x_t | x_{<t})]

The gradient dL_NTP/dW pushes p_W toward the data distribution of next tokens.
For MCQ, the relevant quantity is:

D(W) = P(argmax_{a in {A,B,C,D}} p_W(a | question) = correct_answer)

**Proof**: NTP loss operates over ALL token positions equally within the response.
For medical text training data (explanations, diagnoses, treatment descriptions),
the vast majority of tokens are medical vocabulary, not answer letters.

The gradient contribution from answer-discriminative tokens (the actual A/B/C/D
choice points) is diluted by O(L) non-discriminative tokens where L is average
response length. For L ~ 200 tokens, the discriminative signal is ~0.5% of the
total gradient.

Moreover, NTP training on explanatory medical text actively reshapes the output
distribution toward medical vocabulary (drugs, conditions, procedures), which may
SHIFT probability mass away from the concise answer-letter tokens that MCQ requires.

**Therefore**: dL_NTP/dW and dD/dW are approximately orthogonal. Minimizing L_NTP
provides negligible signal for improving D, and may actively degrade it. QED.

## Theorem 2: TT-LoRA Compression Amplifies Discriminative Loss

**Claim**: TT decomposition with rank r preserves the top-r singular directions of
the weight update DW. If discriminative features have smaller singular values than
generative features, compression amplifies discriminative loss.

**Proof**: Let DW = U S V^T be the SVD of the adapter weight update.
TT decomposition with rank r approximates: DW_TT ~ U_r S_r V_r^T

The approximation error ||DW - DW_TT||_F = sqrt(sum_{i>r} sigma_i^2).

For NTP training, the dominant singular directions capture:
1. Frequent medical vocabulary (high sigma, many training tokens)
2. Syntactic patterns (medium sigma)
3. Answer-discriminative features (low sigma, sparse training signal per Theorem 1)

TT-LoRA rank-6 preserves the top ~6 effective directions. By Theorem 1, the
discriminative signal has low singular value magnitude, so it falls in the truncated
tail. Standard LoRA rank-8 has higher effective rank and retains more of this tail.

**Prediction**: Standard LoRA retains more discriminative capacity than TT-LoRA,
but both degrade relative to base. The ordering is:

D(base) > D(LoRA_NTP) > D(TT-LoRA_NTP) > R

## Quantitative Predictions

**CORRECTION (Phase 1 result)**: Base model measured 30.5%, consistent with Finding #508
(base=31%). The prior MATH.md incorrectly cited "base ~45%" — that 50% figure was the
*adapted* model, not the base. Predictions revised accordingly.

| Configuration | MedMCQA Prediction | Reasoning |
|---|---|---|
| Base model (no adapter) | 29-33% | Finding #508 base=31%, Phase 1 measured 30.5%. Consistent |
| Standard LoRA rank-8, NTP 500 steps | 35-45% | Finding #508: 31%→50% at 1000 steps. 500 steps = partial improvement. K1430 likely PASS |
| TT-LoRA rank-6, NTP 500 steps | 18-28% | E2E benchmark measured 21%. Compression discards discriminative signal per Theorem 2 |

### Kill Criteria Predictions

- **K1430** (Standard LoRA >= 35%): LIKELY PASS. Finding #508 got 50% at 1000 steps.
  At 500 steps, still well above 35%.
- **K1431** (TT-LoRA >= 35%): LIKELY FAIL. E2E benchmark got 21%. Predicted: 18-28%.
- **K1432** (Both below base ~31%): REVISED. Original threshold was "below 45% base"
  which was based on incorrect base estimate. Now: both below 31% → NTP is the disease.

### Diagnostic Decision Tree

1. If K1430 PASS and K1431 FAIL: **Compression is the disease** (standard LoRA preserves
   discriminative features, TT-LoRA discards them)
2. If K1430 FAIL and K1431 FAIL: **NTP is the disease** (both compression levels fail,
   training objective is the root cause)
3. If K1430 PASS and K1431 PASS: **Neither is the disease at these thresholds**
   (unexpected — would require revisiting E2E benchmark methodology)

## Experimental Design

1. Load Gemma 4 E4B 4-bit base model
2. Evaluate base model on 200 MedMCQA validation questions (control)
3. Train Standard LoRA rank-8 on medical NTP data (500 steps, same as E2E)
4. Evaluate Standard LoRA on same 200 MedMCQA questions
5. Reset model, inject TT-LoRA rank-6, train on same data
6. Evaluate TT-LoRA on same 200 MedMCQA questions
7. Compare: base vs LoRA vs TT-LoRA

**Controls**: Same training data, same hyperparameters (lr, steps, batch size),
same evaluation questions (fixed seed), same prompt format.

## Connection to Architecture

This experiment determines whether the discriminative collapse is structural
(training objective) or incidental (compression artifact). The answer dictates
the fix:

- If NTP is the disease: Need mixed-objective training (exp_p0_mcq_mixed_training)
- If compression is the disease: Need higher TT rank or discriminative-aware truncation
- If both: Need both fixes, applied in order of effect size

## References

- arXiv:2504.21190 — TT-LoRA: TT decomposition preserves dominant singular directions
- Finding #517 — Standard LoRA NTP adapter degrades MMLU-Pro by -6.2pp
- Finding #508 — E2E pipeline baselines (GSM8K 17→73%, HumanEval 18→63%, MedMCQA 31→50%)
- E2E benchmark results — TT-LoRA MedMCQA 21% (below random)

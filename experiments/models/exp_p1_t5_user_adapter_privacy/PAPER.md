# PAPER.md — T5.4: User Adapter Privacy (Isolation Proof)

## Abstract

We tested three privacy guarantees for personal style adapters in a multi-user serving
system: behavioral isolation (K1110), membership inference resistance (K1111), and
geometric isolation (K1112). Only behavioral isolation held. MIA delta was 40pp (vs
predicted <20pp) and Grassmannian cosine was 0.6219 (vs predicted <0.50). The experiment
is KILLED on two criteria, with structural fixes identified.

## Experimental Setup

- User A adapter: rank-4 LoRA, trained on 40 general science QA pairs + "Hope that helps, friend!" sign-off (T5.1, Finding #436)
- User B adapter: rank-4 LoRA, trained on 20 science QA pairs + "Best regards, colleague." sign-off (100 iters)
- Model: mlx-community/gemma-4-e4b-it-4bit (Gemma 4, 4-bit)
- K1112: CPU-only safetensors analysis, lora_a matrices (q_proj), 16 layers
- K1110: User A adapter applied to 5 User B test queries, measure B-style compliance
- K1111: User A adapter applied to 10 member (T5.1 train) + 10 non-member queries

## Prediction vs Measurement Table

| Criterion | Theorem | Prediction | Measurement | Pass? |
|-----------|---------|------------|-------------|-------|
| K1110: Behavioral isolation (0 B sign-offs) | Theorem 1 (routing exclusivity) | 0/5 produce "Best regards, colleague." | 0/5 (0.0%) | **PASS** |
| K1111: MIA delta < 20pp | Theorem 2 (uniform style injection) | \|member - non_member\| < 20pp | \|100% - 60%\| = 40pp | **FAIL** |
| K1112: max\|cos(Y_A_a, Y_B_a)\| < 0.50 | Theorem 3 (JL + training drift) | ~0.10-0.30 (JL=0.040, drift=0.15) | 0.6219 (worst: layer 28, mean=0.411) | **FAIL** |

## Per-Kill-Criterion Analysis

### K1110 PASS — Behavioral Isolation Holds

User A's adapter (trained on "Hope that helps, friend!") produced 0/5 occurrences of
"Best regards, colleague." when applied to User B's queries. User A compliance was 60%
(3/5 produced A's sign-off). Base compliance for both markers ≈ 0%.

**Conclusion:** Routing exclusivity (T3.7 Finding #430) guarantees zero cross-user
behavioral contamination. This result is structurally guaranteed — not probabilistic.

### K1111 FAIL — MIA Delta = 40pp (Kill Condition Triggered)

| Split | Compliance | Questions |
|-------|-----------|-----------|
| Member (10 training questions) | 100% | photosynthesis, vaccines, DNA, etc. |
| Non-member (10 held-out questions) | 60% | boiling point, rainbow, hurricanes, etc. |
| **Delta** | **40pp** | Predicted: <20pp |

Theorem 2 predicted uniform style injection: "the adapter learns P(style_marker | ANY_q)
not P(style_marker | q_i ∈ training)." This was REFUTED.

Root cause: member and non-member questions are similar-distribution (both general science)
but the adapter learned to associate the sign-off with the specific token sequences in the
training set. With only 40 training examples, rank-4 LoRA can fit per-sample patterns.

### K1112 FAIL — max|cos|=0.6219 (Kill Condition Triggered)

| Metric | Measured | Predicted |
|--------|----------|-----------|
| max\|cos(Y_A_a, Y_B_a)\| | 0.6219 | <0.50 (expected 0.10-0.30) |
| mean\|cos\| | 0.4112 | ~0.19 |
| JL bound (random) | 0.0395 | 0.040 |
| Worst layer | Layer 28 | — |

Theorem 3 predicted that different style gradients would keep lora_a subspaces separated.
This was REFUTED. Root cause: lora_a matrices extract features from input hidden states h.
When both users are asked general science questions, BOTH adapters learn similar input
representations in lora_a (input side). Style differentiation happens in lora_b (output
side). The experiment measured the wrong matrix for isolation.

## Impossibility Structures Identified

**K1111 — Why per-example MIA signal is structurally unavoidable:**
A rank-r adapter trained on n examples with similar-distribution train/test splits will
ALWAYS show higher compliance on training examples until the model perfectly generalizes.
Perfect generalization requires n → ∞ or true OOD non-member questions. With n=40
and same-domain questions, partial memorization is guaranteed by the learning dynamics.

**K1112 — Why lora_a isolation fails for same-input-distribution users:**
lora_a(h) = A^T h is a linear map acting on the input representation h. When two users
receive the same distribution of questions (both general science), their A matrices
learn to extract similar features from h (e.g., "what kind of answer structure is
appropriate for a science question"). Style specificity lives in lora_b (the output
direction), not lora_a (the input direction). Measuring cos(Y_A_a, Y_B_a) tests the
wrong object for personal style isolation.

## Structural Fixes (for Follow-Up)

1. **K1111 fix**: Use genuinely OOD non-member questions (e.g., legal/medical for a
   general-science user) OR use differential privacy (DP-SGD) during user training.
   Alternatively, measure compliance on semantically distant topics where the style
   adapter cannot generalize its specific sign-off.

2. **K1112 fix**: Measure cos(Y_A_b, Y_B_b) for lora_b matrices (output directions)
   instead of lora_a. Alternatively, use the full product AB as the subspace representation
   (measures effective weight update direction in d×d space, not intermediate d×r space).

3. **Orthogonalization**: Apply Gram-Schmidt to User B's lora_a init relative to User A's
   lora_a (T3.6 technique, Finding #429). This guarantees geometric isolation by construction.

## Runtime

- Total: 34.2s (Phase 0: 16.8s training, Phase 1: 0.005s, Phase 2: 2.6s, Phase 3: 12.2s)
- Status: Script exited with code 1 (all_pass=False written to results.json)

# MATH.md — T2.5: SFT-Residual M2P on Gemma 4

## V2 Audit Rerun (2026-04-18, audit-2026-04-17-rerun + code-bug tags)

V1 theorems unchanged. V2 adds precondition pre-registration per the class-level
standing rule (4th instance this loop — see PAPER.md V2 section for context).

**Preconditions pre-registered BEFORE probe:**

| ID | Precondition | Source |
|---|---|---|
| P1 | T2.1 math adapter `.safetensors` on disk | Theorem 1 requires B_sft to compute B_applied = B_sft + ΔB |
| P2 | T2.1 `data/math/train.jsonl` on disk | Training loop feeds samples |
| P3 | Upstream T2.1 verdict ≠ `KILLED` | B_sft validity inherited from T2.1 SFT phase |

If any precondition fails, K1044/K1045/K1046 route FAIL as *unmeasurable*
(honest cannot-measure; distinguishes from measured-and-fell-short). This is
the pre-registered routing — no KC text changed from V1 thresholds.

**MATH.md git-diff check (PLAN.md §1 item 5):** V1 KCs unchanged. V2 adds only
precondition structure above the KC table. No threshold was relaxed. No KC
removed. V1 measurements (QR=0.707 < 0.738) are preserved in PAPER.md V1.

---

## V1 (unchanged below this line)

## Experiment Type: Verification

## Problem

Finding #403 proved that SFT-residual M2P prevents degradation below SFT quality on Qwen3-4B.
Does the same mathematical guarantee hold on Gemma 4 E4B?

Specifically: does initializing ΔB = 0 guarantee that M2P starts at SFT quality (quality_ratio = 1.0
exactly), and can M2P improve beyond SFT (quality_ratio ≥ 0.90) after training?

---

## Prior Math

**Theorem A (Residual Learning, He et al. 2016, arXiv:1512.03385):**
For F(x) = H(x) + x (residual connection), initializing H = 0 gives F = identity at step 0.
The identity is easy to learn; deviation from identity is parameterized by H.

**Finding #403 (SFT-Residual M2P on Qwen3-4B):**
B_applied = B_sft + output_scale × head(z) with zero-init head achieves quality_ratio = 1.175.
Init quality_ratio = 1.00 exactly (SFT floor). M2P improves beyond SFT after 1000 steps.

**LoRA Forward (mlx_lm LoRALinear):**
For input x ∈ ℝ^(... × d_in):
  output = W_base(x) + scale × (x @ A) @ B
where A ∈ ℝ^(d_in × rank), B ∈ ℝ^(rank × d_out).

**Gemma 4 E4B Architecture (T2.1 confirmed):**
- 42 layers total: 35 local (d_out=2048), 7 global (d_out=4096)
- Global layers: indices [5, 11, 17, 23, 29, 35, 41]
- Local layers: all others  
- LoRA rank=6, scale=6.0 (from T2.1 math adapter)
- T2.1 baseline: SFT_acc=82% GSM8K, base_acc=0%

---

## Theorem 1: Zero-Init Guarantee

**Setup:**
Let B_sft ∈ ℝ^(rank × d_out_l) be the SFT-trained B-matrix for layer l (from T2.1 adapter).
Let ΔB_l ∈ ℝ^(rank × d_out_l) be a learnable correction initialized to 0_matrix.
Define the SFT-residual connection: B_applied_l = B_sft_l + ΔB_l.

**Theorem:**
At initialization (ΔB_l = 0 for all l):
  ||B_applied_l - B_sft_l||_F = 0 for all l ∈ {0, ..., 41}

**Proof:**
||B_applied_l - B_sft_l||_F = ||B_sft_l + ΔB_l - B_sft_l||_F = ||ΔB_l||_F = ||0||_F = 0.
QED.

**Corollary:** At step 0, the LoRA output for every layer equals the T2.1 SFT adapter output:
  scale × (x @ A) @ B_applied_l = scale × (x @ A) @ B_sft_l + scale × (x @ A) @ 0 = SFT output.

Therefore, model quality at step 0 = T2.1 SFT quality (82% GSM8K). ■

---

## Theorem 2: Gradient Flow After Zero Init

**Claim:** Although ΔB_l = 0 at step 0, gradient ∂L/∂ΔB_l is non-zero, enabling learning.

**Proof:**
Let L be the CE loss. The LoRA output z_l = scale × (x @ A_l) @ B_applied_l.
  ∂L/∂ΔB_l = ∂L/∂B_applied_l (chain rule, since ∂B_applied_l/∂ΔB_l = I_r×d_out)
  ∂L/∂B_applied_l = (scale × x @ A_l)^T @ (∂L/∂z_l)

For non-trivial input x: (x @ A_l) ≠ 0, so ∂L/∂ΔB_l ≠ 0 generically.

Contrast with output_scale × head(z) formulation: if output_scale = 0, then
  ∂L/∂head_params = output_scale × (...) = 0 → NO gradient at step 0 (cold start).

The ΔB_l = 0 formulation avoids cold-start: gradient flows from step 1 without warmup.
QED.

---

## Predictions

| Quantity | Prediction | Justification |
|----------|-----------|---------------|
| ||ΔB_l||_F at step 0 | 0 exactly for all 42 l | Theorem 1 (deterministic) |
| Quality at step 0 | 82% GSM8K (= SFT quality) | Theorem 1 Corollary |
| Quality after 500 steps | ≥ 73.8% (= 90% × 82%) | Finding #403 shows M2P improves over SFT |
| B_applied time | < 10ms | 42 rank×d_out matrix additions on M5 Pro |

---

## Kill Criteria

| ID | Criterion | Threshold | Predicted |
|----|-----------|-----------|---------|
| K1044 | M2P GSM8K accuracy after training | ≥ 73.8% | ≥ 82% (improve over SFT) |
| K1045 | B_applied computation time on M5 Pro | < 10ms | < 1ms (trivial additions) |
| K1046 | ||B_applied - B_sft||_F at step 0 | = 0 (all 42 layers) | 0 exactly (Theorem 1) |

---

## Failure Modes

**K1044 FAIL scenario:** ΔB_l grows large and disrupts the SFT-learned B_sft structure.
- Protection: Initialize ΔB_l = 0 with small LR (1e-5). Catastrophic forgetting requires
  ΔB_l to grow by O(||B_sft_l||_F) ≈ O(0.35) before quality degrades below 0.90 × SFT.
- Since B_sft already provides task structure, ΔB_l merely fine-tunes — less risk than
  training from scratch.

**K1046 FAIL scenario:** Float precision issues make ||ΔB_l||_F > 0.
- Impossible by construction: mx.zeros() returns exactly 0. No floating point error.
  (float error arises from operations, not initialization.)

---

## Connection to P1 Architecture

SFT-residual M2P is the foundation of the M2P-in-adapter design:
1. SFT phase: train B_sft for each domain (done in T2.1)
2. M2P phase: learn ΔB = f(context) for within-domain personalization
3. At serving: B_applied = B_sft + M2P(context) applied per-request

The zero-init guarantee makes M2P SAFE: wrong context → ΔB → 0 → falls back to SFT quality.
This is impossible to achieve with random-init ΔB (T2.4 showed QR = -5.89 with random init).

---

## References

- He et al. (2016), arXiv:1512.03385 — Residual learning: zero-init ensures identity at start
- Hu et al. (2022), arXiv:2106.09685 — LoRA: low-rank adaptation of frozen LLMs
- Finding #403 — SFT-residual M2P on Qwen3-4B achieves quality_ratio=1.175
- T2.1 math adapter — Gemma 4 E4B GSM8K training baseline (82% accuracy)

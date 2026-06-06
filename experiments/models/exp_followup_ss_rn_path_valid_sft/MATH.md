# MATH.md — SFT-residual M2P with personalization data (Finding #403 Gemma 4 replication, within-5pp variant)

## Experiment Type
Verification (Finding #403 replication on Gemma 4 E4B 4-bit, tighter tolerance)

## Motivation
Finding #403 (`exp_m2p_qwen4b_sft_residual`, Qwen3-4B, QR=1.175) established
that `B_applied = B_sft + output_scale(t)·head(z)` with zero-init head and
personalization-distinct data preserves SFT quality while admitting a
per-context residual. The parent Gemma 4 attempt
(`exp_p1_t2_sft_residual_gemma4`) used SFT replay data and KILLED at QR=0.707
via the gradient identity `∂L/∂ΔB = ∂L/∂B_applied` (catastrophic forgetting).

This experiment re-registers the Gemma 4 replication with a **tighter**
tolerance than the sibling `exp_followup_sft_behavioral_lora_scale_5` (QR ≥
0.90): K1572 demands the final accuracy track Finding #403's reference acc
"within 5pp". Both experiments share preconditions; the tolerance difference
changes only the PASS band, not the probe routing.

## Prior Math

**Theorem A (He et al. 2016, arXiv:1512.03385, residual learning):** For
`F(x) = H(x) + x`, initializing H=0 gives F = identity at step 0; deviation
is parameterized by H.

**Theorem B (Bachlechner et al. 2020, arXiv:2003.04887, ReZero):** For
`y = x + α·z` with α(0)=0, `∂L/∂θ|_{α=0} = 0` (cold start); module
contributes nothing until α > 0 — monotone-safe injection.

**Theorem C (Kirkpatrick et al. 2017, arXiv:1612.00796, EWC):** Gradient
descent on the SFT loss surface drives θ away from θ_sft at rate
`O(η·‖∇L(θ_sft)‖)` regardless of reparameterization. Zero-init ΔB does not
block this because `∂L/∂ΔB = ∂L/∂B_applied` at every step after init.
Protection requires data separation OR explicit ‖ΔB‖_F regularization.

**Finding #403 (evidence):** output-scale warmup + zero-init head +
**personalization** data on Qwen3-4B → QR=1.175, acc_step0=73%, acc_final=74.4%.

## Theorem 1 (replication claim, within-5pp band)

Let `B_sft_ℓ ∈ ℝ^{r×d_out_ℓ}` be the T2.1 SFT-trained B-matrix for layer ℓ
(Gemma 4 E4B 4-bit, 42 layers, rank=6, local d_out=2048, global d_out=4096).
Let `head_ℓ : ℝ^{d_ctx} → ℝ^{r×d_out_ℓ}` be a 2-layer MLP with W_out := 0.
Let `output_scale(t)` satisfy `output_scale(0)=0`, ramping linearly to 1 over
W=200 steps.

Define

    B_applied_ℓ(t) = B_sft_ℓ + output_scale(t) · head_ℓ(z_context)

Train on personalization corpus P (disjoint from GSM8K SFT split) for 1000
steps at η=5e-6, GRAD_CLIP=0.5.

**Claim (K1572):** `|acc_final_Gemma4 − acc_ref_Qwen3_4B| ≤ 5pp`, where
`acc_ref_Qwen3_4B = 74.4%` from Finding #403.

**Proof sketch:**
- Theorem A + init ⇒ `B_applied(0) = B_sft` exactly ⇒ `acc_step0 = acc_sft_Gemma4`.
- Theorem B ⇒ warmup gives 0 gradient at t=0; optimization starts from the
  SFT minimum, not a random perturbation.
- Theorem C ⇒ personalization data P decouples the loss surface from the SFT
  data; catastrophic forgetting bounded by `O(η·‖∇L(θ_sft; P)‖) ≪ O(η·‖∇L(θ_sft; SFT)‖)`.
- Finding #403 delivered QR=1.175 on Qwen3-4B; Gemma 4 base layer shapes +
  rank match T2.1; no structural reason for the mechanism to break on a
  4B-equivalent 4-bit base.

Therefore `acc_final_Gemma4 ≈ acc_ref_Qwen3_4B ± 5pp`. ■

## Preconditions (pre-registered before probe, per PLAN.md §KC discipline)

| ID | Precondition | Source |
|---|---|---|
| P1 | T2.1 math adapter `.safetensors` on disk at `exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors` | Eq. 1 requires B_sft; substituting random B_sft puts the run in class T2.4 (random-init ΔB, QR=−5.89, already KILLED) |
| P2 | Personalization-distinct corpus (persona-tagged, disjoint from GSM8K) staged in `data/personalization/` or sibling path | Theorem C requires data separation; GSM8K replay is known-failure (parent QR=0.707) |
| P3 | T2.1 upstream verdict ≠ KILLED | B_sft validity inherits from T2.1; invalid upstream invalidates the replication substrate |

If any of P1/P2/P3 fails, **K1572 routes FAIL as unmeasurable** — not
measured-and-fell-short. This is the pre-registered probe path used by the
parent V2 rerun and five sibling 2026-04-18 audit-rerun experiments
(peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed,
sequential_activation_compose_real, sft_behavioral_lora_scale_5). No KC
threshold is relaxed; the routing is pre-registered, not inferred post-hoc.

## Kill Criteria (pre-registered)

| ID | Criterion | Threshold | Predicted (under preconditions) |
|----|-----------|-----------|---------|
| K1572 | SFT-residual M2P replicates Finding #403 on Gemma 4 | `|acc_final_Gemma4 − 74.4%| ≤ 5pp` | ≈ 74–75% (within band, Qwen3-4B evidence QR=1.175) |

**Routing when P1∨P2∨P3 fail:** K1572 status = FAIL (unmeasurable), verdict
KILLED, all_pass=False. Not upgraded to PROVISIONAL because Finding #403 is
already supported on Qwen3-4B; this experiment's role is a Gemma 4
confirmation. Without valid B_sft, the replication cannot be stated either way.

**MATH.md KC discipline:** K1572 threshold "within 5pp" is the DB-tracked
text at claim time; no KC text has been modified after probe; the precondition
routing is declared here before `run_experiment.py` executes.

## References
- He et al. 2016 arXiv:1512.03385 — residual learning, zero-init identity
- Bachlechner et al. 2020 arXiv:2003.04887 — ReZero: output-scale warmup
- Kirkpatrick et al. 2017 arXiv:1612.00796 — EWC, catastrophic forgetting
- Finding #403 (`exp_m2p_qwen4b_sft_residual`) — Qwen3-4B, QR=1.175 (reference)
- Finding #447 (`exp_p1_t2_sft_residual_gemma4`) — Gemma 4 parent KILL, QR=0.707
- Finding #600 (`exp_followup_sft_behavioral_lora_scale_5`) — 5th precondition-probe KILL (this loop), same preconditions

# MATH.md — Replicate Finding #403 at Gemma 4 scale (output_scale·head(z) + personalization data)

## Experiment Type
Verification (Finding #403 structural replication on Gemma 4 E4B 4-bit)

## Motivation
The parent experiment `exp_p1_t2_sft_residual_gemma4` killed the
`B_applied = B_sft + ΔB` (zero-init ΔB) variant with QR=0.707 because it
re-trained on the **same** GSM8K data; the gradient identity
`∂L/∂ΔB = ∂L/∂B_applied` drove ΔB to 24.6% of ‖B_sft‖_F and corrupted the
SFT-learned reasoning structure (catastrophic forgetting; Kirkpatrick 2017
arXiv:1612.00796).

Finding #403 (`exp_m2p_qwen4b_sft_residual`, Qwen3-4B) used a structurally
different formulation on different data:

  B_applied = B_sft + output_scale(t) · head(z)          (Eq. 1)
  head_init := 0,  output_scale(0) = 0                   (Eq. 2)

with **personalization context data** (not the SFT training distribution).
Both changes are load-bearing. This experiment isolates the Gemma 4 replication
of Finding #403 — i.e. does the Qwen3-4B structural result (QR=1.175) hold
when the base model changes to Gemma 4 E4B 4-bit?

## Prior Math

**Theorem A (He et al. 2016, arXiv:1512.03385, residual learning):** For
F(x)=H(x)+x, initializing H=0 gives F = identity at step 0; deviation from
identity is parameterized by H.

**Theorem B (output-scale warmup, Bachlechner et al. 2020 arXiv:2003.04887,
ReZero):** Let z = f(x; θ); the output-gated module y = x + α·z with α(0)=0
provides `∂L/∂θ|_{α=0} = α · (…) = 0` (cold start). The module contributes
nothing until α > 0, giving a monotone-safe injection path.

**Theorem C (Kirkpatrick 2017 arXiv:1612.00796, EWC): gradient descent on the
**same** loss surface (data distribution used to form θ_sft) drives θ away
from θ_sft at rate O(η·‖∇L(θ_sft)‖), regardless of the reparameterization.
Zero-init ΔB does not block this because ∂L/∂ΔB = ∂L/∂B_applied at every
step after initialization. Protection requires EITHER data separation
(different loss surface) OR explicit regularization (‖ΔB‖_F penalty).

**Finding #403 (evidence):** B_applied = B_sft + output_scale·head(z) with
zero-init head + **personalization** data on Qwen3-4B obtained QR=1.175,
acc_step0=73% (SFT floor), acc_final=74.4%. The mechanism credited in the
finding is the pairing of output-scale warmup AND data-distribution
separation — either alone has known failure modes.

## Theorem 1 (replication claim)

**Setup (Gemma 4 E4B 4-bit, `mlx-community/gemma-4-e4b-it-4bit`, 42 layers,
local d_out=2048, global d_out=4096, rank=6, scale from T2.1 adapter):**

Let B_sft_ℓ ∈ ℝ^{r×d_out_ℓ} be the T2.1 SFT-trained B-matrix for layer ℓ.
Let head_ℓ: ℝ^{d_ctx} → ℝ^{r×d_out_ℓ} be a 2-layer MLP with W_out initialized
to 0. Let output_scale(t) be a warmup schedule output_scale(0)=0, ramping
linearly to 1 over W=200 steps.

Define:

    B_applied_ℓ(t) = B_sft_ℓ + output_scale(t) · head_ℓ(z_context)

Train on personalization data P (disjoint from GSM8K SFT data) for 1000 steps
with η=5e-6, GRAD_CLIP=0.5.

**Claim (K1565):** acc_final on a held-out Gemma 4 GSM8K eval subset
≥ 0.90 · acc_sft_Gemma4 and acc_final ≥ 0.90 · acc_step0.

I.e. Finding #403's QR ≥ 0.90 floor replicates structurally on Gemma 4 under
the same formulation and data-separation discipline.

**Proof sketch:**
- Theorem A + Eq. 2 ⇒ B_applied(0) = B_sft exactly ⇒ acc_step0 = acc_sft.
- Theorem B ⇒ output_scale·head(z) contributes 0 gradient at step 0; the
  system begins from the SFT minimum, not from a random perturbation.
- Theorem C ⇒ because training data P ≠ GSM8K, ∇L(θ_sft; P) is not driving
  toward B_sft's original minimum — catastrophic forgetting is attenuated to
  O(η·‖∇L(θ_sft; P)‖) rather than O(η·‖∇L(θ_sft; SFT_data)‖).
- Qwen3-4B evidence (Finding #403): the pairing delivered QR=1.175.
- Gemma 4's base layer shapes + rank match T2.1; no structural reason for the
  mechanism to break on a 4B-equivalent base.

Floor preserved ⇒ QR ≥ 0.90 on Gemma 4. ■

## Preconditions (pre-registered before probe, per PLAN.md §KC discipline)

| ID | Precondition | Source |
|---|---|---|
| P1 | T2.1 math adapter `.safetensors` on disk at `exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors` | Eq. 1 requires B_sft (non-synthesizable; substituting random B_sft would change experiment class from "Finding #403 replication" to "random-init ΔB", which T2.4 already killed at QR=−5.89) |
| P2 | Personalization-distinct eval corpus (persona-tagged queries, disjoint from GSM8K) | Theorem C requires data separation; same-domain replay is known-failure (parent QR=0.707) |
| P3 | T2.1 upstream verdict ≠ KILLED | B_sft validity inherits from T2.1; an invalid B_sft invalidates the substrate of the replication claim |

If any precondition fails, K1565 routes FAIL as **unmeasurable** — not
measured-and-fell-short. This is the same precondition-probe routing used by
the parent V2 rerun and three other 2026-04-18 audit-rerun experiments
(peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed,
sequential_activation_compose_real). The routing is pre-registered; no KC
threshold was relaxed after data collection.

## Kill Criteria (pre-registered)

| ID | Criterion | Threshold | Predicted (under preconditions) |
|----|-----------|-----------|---------|
| K1565 | Finding #403 replication on Gemma 4: QR = acc_final / acc_step0 | ≥ 0.90 | ≈ 1.00 (SFT floor preserved by Theorem A+B; Qwen3-4B evidence QR=1.175) |

**Routing when P1∨P2∨P3 fail:** K1565 status = FAIL (unmeasurable), verdict
KILLED, all_pass=False. Not upgraded to PROVISIONAL because Finding #403 is
already supported on Qwen3-4B; this experiment's job is to confirm at Gemma 4
scale. Without B_sft, the Gemma 4 replication cannot be stated either way.

**MATH.md KC discipline:** K1565 threshold 0.90 comes directly from Finding
#403's reported floor and was the DB-tracked text at claim time. No threshold
has been relaxed. No KC was added/removed after the probe. The precondition
routing above was pre-registered in this file.

## References
- He et al. 2016 arXiv:1512.03385 — residual learning, zero-init identity
- Bachlechner et al. 2020 arXiv:2003.04887 — ReZero: output-scale warmup
- Kirkpatrick et al. 2017 arXiv:1612.00796 — EWC, catastrophic forgetting
- Finding #403 (`exp_m2p_qwen4b_sft_residual`) — Qwen3-4B replication, QR=1.175
- Finding #447 (`exp_p1_t2_sft_residual_gemma4`) — zero-init ΔB KILL on Gemma 4 (QR=0.707)
- mlx-lm 0.21+ — LoRALinear forward path used for B_applied injection

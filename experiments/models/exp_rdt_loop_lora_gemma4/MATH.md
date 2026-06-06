# MATH — exp_rdt_loop_lora_gemma4

## Setup

Architecture: frozen `mlx-community/gemma-4-e4b-it-4bit` (hidden=2560, 42 layers,
num_heads=8, num_kv_heads=2, head_dim=256). Recurrent block = layers 12–20
(9 consecutive DecoderLayers). Per-loop LoRA rank r=16 on `v_proj` and `o_proj`
per Pierre Finding #562 default. N_loops = 6. LTI-injection at block entry.

References:
- Bae 2024, Relaxed Recursive Transformers (arxiv:2410.20672) — loop-indexed LoRA.
- Parcae (Prairie 2026, arxiv:2604.12946) — LTI stability (verified in MLX as
  Finding #667 via `exp_rdt_lti_injection_mlx`).
- Saunshi 2025 (arxiv:2502.17416) — latent reasoning scales with loop depth.
- Pierre Finding #562 — partition-QR Grassmannian A-matrix orthogonality at
  Gemma 4 native dims.

Libraries: `mlx==0.31.1`, `mlx-lm==0.31.2`, seed 42.

## Forward pass (inference, T loops)

Let `H` = hidden state entering layer 12. Let `Block(h)` = sequential composition
of layers 12–20 with per-layer LoRA selected by current loop index t:

  for t in 1..T:
    H_in_t = A_t ⊙ H + B_t ⊙ e + H        (LTI-injection, e = layer-12 input embedding)
    H      = Block_t(H_in_t)               (layers 12–20 with loop-t LoRA deltas)

Where for each layer ℓ ∈ {12..20}, projection p ∈ {v_proj, o_proj}:

  W_ℓ,p,t = W_ℓ,p,base + α · B_ℓ,p,t @ A_ℓ,p,t
  with A_ℓ,p,t ∈ ℝ^{r × in_ℓ,p}, B_ℓ,p,t ∈ ℝ^{out_ℓ,p × r}

Post-block output replaces the layer-20 → layer-21 edge. Only the LoRA tensors
`{A,B}_ℓ,p,t` and LTI params `{log_A, log_dt, B_lti}` are trainable; base frozen.

## Partition-QR Grassmannian A-init (per-projection)

For fixed (ℓ, p), build A_ℓ,p,· ∈ ℝ^{N·r × in_ℓ,p}:

  W = Gaussian(in_ℓ,p, N·r); Q, _ = QR(W); Q ∈ ℝ^{in_ℓ,p × N·r}
  A_ℓ,p,t = Q[:, (t−1)·r : t·r].T        (r × in_ℓ,p)

By Q^T Q = I_{N·r} in exact arithmetic, A_ℓ,p,i A_ℓ,p,j^T = 0_{r×r} for i≠j.

## Kill criteria (pre-registered)

**K1740** (target, full-scale): Looped-T=3 variant beats base Gemma 4 E4B by ≥ +5pp
on GSM8K-Hard (100 problems, greedy decoding). **Full-scale only.**

**K1741** (target, full-scale): MMLU (57 subjects) within 1pp of base Gemma 4 E4B.

**K1742** (behavioral, full-scale): Quality follows saturating exponential
y(T) = y∞ − (y∞ − y0)·exp(−T/τ) on T ∈ {1..6}, R² > 0.90 on GSM8K-Hard.

**K1743** (structural, smoke-safe): At init, max_{i≠j, ℓ, p} |cos(A_ℓ,p,i, A_ℓ,p,j)| < 0.1.
Measured in float32 on the real Gemma 4 in-dims (320→2560 hidden, 2048 out).

**K1744** (stability, smoke-safe): During 50 Adam training steps on a synthetic
reconstruction loss, max_{batch, t} ρ(A_d,t) < 1 at every step (reuses F#667
primitive in the composition context — LTI inside loop-indexed LoRA block).

## Theorem 1 (Orthogonality at init)

**Claim.** K1743 holds with |cos| ≤ 100·sqrt(r / in_ℓ,p)·u_f32 ≈ 10⁻⁷ ≪ 0.1.

**Proof.** By Theorem 1 of F#562, partition-QR gives Q^T Q = I in exact arithmetic,
with Higham §19.3 float-32 deviation bounded by c·sqrt(N·r)·u_f32.
For r=16, N=6, u_f32 ≈ 6e−8: bound ≈ 6e−7 ≪ 0.1. QED.

Note (tautological-structural, acknowledged per F#498/F#666): K1743 passes **by
construction** under partition-QR init — this is not evidence of orthogonality
as a learned property, only that the *initialization* respects structural
orthogonality. Saves the scale-safety antipattern by pairing it with K1740 (target).

## Theorem 2 (Recurrent stability)

**Claim.** K1744 holds throughout training in exact arithmetic;
float32 degrades to ρ=1 only at the clamp boundary s = -20 (F#667 Lemma 1).

**Proof.** By F#667 Theorem 1: A_d = exp(-exp(clamp(log_dt+log_A, -20, 20))) ∈ (0,1)
in exact arithmetic, independent of the surrounding composition (LTI is applied
element-wise before the TransformerBlock call). Adam updates on log_A, log_dt
affect only their values, not the clamp structure; as long as optimiser trajectory
stays away from s ≤ -20 (verified empirically in F#667: max rms 0.64 over 1000
steps), ρ < 1 is preserved. QED modulo boundary (see F#667 caveat).

## Theorem 3 (Target KCs require full-scale)

**Claim.** K1740, K1741, K1742 are untestable in smoke mode and must defer to
a full-scale follow-up.

**Reasoning.** K1740/K1741 require (i) a reasoning-corpus training phase
(GSM8K + MATH train split, ~10k samples × multi-epoch), (ii) benchmark eval on
100+ problems × 6 loop counts. Under M5 Pro 48GB and 2-hour scope constraint
(researcher hat), this is not achievable in a single iteration. K1742 requires
K1740 at 6 loop counts. Scope: this experiment verifies **K1743 + K1744 in the
composition context** (i.e., extends F#667 primitive to loop-indexed LoRA + LTI
stacked on real Gemma 4 layers 12–20) and marks `is_smoke=true`, yielding
`provisional` status with a structured follow-up for the macro claim.

## Antipattern self-audit

- (f) Tautological KC? K1743 is tautological by construction — **acknowledged**
  above and paired with K1744 (dynamics) + target K1740 (full-scale deferred).
- (h) LORA_SCALE unsafe? α=2 (safe default per audit guidelines, matches Pierre v8).
- (i) Tautological routing? No routing — loop index is a scheduled hyperparameter,
  not a learned router.
- (k) Hardcoded pass? No — all KC measurements come from runtime tensors.
- (m) proxy-model? Base model is the product target (Gemma 4 E4B-it-4bit).
- (n) KC vs target? K1743/K1744 are scaffolding KCs; K1740 is the target KC;
  verdict split acknowledges this (smoke → provisional, not supported).
- F#452/F#453/F#1564 "reproduces or refutes"? K1744 explicitly extends F#667 to
  a new composition context; content of the extension (loop-indexed LoRA between
  LTI applications) is novel, so not tautological-duplicate.
- F#138 is_smoke-as-full: `is_smoke=true` set in `results.json`; status kept at
  `provisional` per verdict-consistency rule #4.

## Prediction vs measurement (for PAPER.md)

| KC | Prediction (proof) | Measurement (smoke) |
|---|---|---|
| K1743 | |cos|_max < 1e-6 (float32, Higham) | TO FILL by run_experiment.py |
| K1744 | max ρ(A) < 1 across 50 steps | TO FILL |
| K1740 | ≥ +5pp GSM8K-Hard at T=3 | NOT MEASURED (full-scale) |
| K1741 | |ΔMMLU| ≤ 1pp | NOT MEASURED (full-scale) |
| K1742 | R² > 0.90 saturating-exp | NOT MEASURED (full-scale) |

Smoke-mode success = K1743 PASS ∧ K1744 PASS → verdict `provisional`;
pipeline OK, macro claim deferred to a follow-up experiment.

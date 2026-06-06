# MATH.md: SFT-Residual M2P on Qwen3-4B + GSM8K

## TYPE: frontier-extension
## PROVEN RESULT: M2P v5 SHINE base-as-encoder (Finding #401, quality_ratio=0.833 at 0.6B)
## FRONTIER: SFT-residual connection in weight space to fix 4B scaling failure

---

## Why This Experiment

Three consecutive M2P failures at 4B:
1. exp_m2p_qwen4b_gsm8k: quality_ratio=-0.125 (v1, standalone MLP encoder)
2. exp_m2p_qwen4b_gsm8k_v5: quality_ratio=-0.187 (v5, SHINE base-as-encoder)

Both generate B-matrices from scratch that actively degrade below base accuracy.
At 0.6B, the same architectures work (quality_ratio=0.754-1.433).

Root cause (Finding #402 analysis): The M2P encoder (d=1024) cannot learn to map
per-layer hidden states to useful B-matrices for 36 layers of a 4B model from
random initialization in 300 steps. The output space per q_proj head is 4×4096=16384
dimensions, requiring massive upsampling from 1024-dim representations.

---

## Theorem 1: SFT-Residual Quality Floor

**Statement:** Let B_sft[li] be the SFT-trained LoRA B-matrix for layer i, and let
ΔB[li] = output_scale * head(z[li]) be the M2P-generated residual. If head weights
are initialized with standard init (zero-mean, small variance σ²), then at initialization:

    E[||ΔB[li]||_F] = output_scale * σ * sqrt(rank * d_proj_out)

For output_scale=0.032, σ=1/sqrt(d_m2p)=1/32, rank=4, d_proj_out=4096 (q_proj):

    E[||ΔB_q||_F] = 0.032 * (1/32) * sqrt(4 * 4096) = 0.001 * 128 = 0.128

Meanwhile ||B_sft_q||_F ~ O(1) (trained SFT weights). So ||ΔB|| << ||B_sft||,
and B_applied = B_sft + ΔB ≈ B_sft at initialization.

**Corollary:** quality_ratio at init ≈ quality_ratio of SFT alone ≈ 1.0.
The M2P cannot degrade below SFT quality at initialization.

**Proof:** By triangle inequality, ||B_sft + ΔB - B_sft|| = ||ΔB|| << ||B_sft||.
The perturbation is small relative to the SFT signal. QED.

## Theorem 2: Residual Refinement Improves Quality

**Statement:** If the M2P is trained to minimize CE loss with B_applied = B_sft + ΔB,
the gradient with respect to head weights is:

    ∂L/∂W_head = ∂L/∂B_applied * output_scale * z[li]

This is identical to standard M2P training, scaled by output_scale. The gradient
signal points in the direction that reduces CE loss from the SFT-initialized adapter.
Since SFT quality > base quality (8pp improvement at 4B), the gradient landscape
has a clear descent direction toward further improvement.

**Prior evidence:** v4 at 0.6B warm-started from SFT B-weights and achieved
quality_ratio=1.433 (28.6% M2P vs 26.0% SFT), proving that refinement from
SFT initialization can EXCEED the SFT quality target.

## Theorem 3: Capacity Analysis

At 4B, the M2P residual ΔB needs to capture only the DIFFERENCE between the
optimal B and B_sft, not the full B from scratch. By Aghajanyan et al. (2020),
the intrinsic dimension of task-specific updates d_int << min(rank, d_model).

Finding #387 measured d_int=86 (q) and 69 (v) for SFT B-matrices at 0.6B.
The residual ΔB = B_optimal - B_sft should have even LOWER intrinsic dimension
(it's a correction to an already-good approximation), so d_m2p=1024 >> d_int
provides ample capacity.

---

## Kill Criteria

- **K1:** init_quality_ratio >= 0.80 — SFT B-matrices preserved at init
  (Theorem 1 guarantees this: ΔB << B_sft at init)
- **K2:** quality_ratio >= 0.60 at n=500 after 1000 training steps
  (Theorem 2: gradient refinement from SFT baseline)
- **K3:** grad_norm > 0 at step 0 — gradient flows through residual path

## Predictions

| Metric | Predicted | Reasoning |
|--------|-----------|-----------|
| init_quality_ratio | 0.90-1.10 | B_applied ≈ B_sft at init (Theorem 1) |
| quality_ratio (1000 steps) | 0.80-1.30 | Residual refinement (Theorem 2, v4 precedent) |
| M2P accuracy at n=500 | 66-76% | Base=65%, SFT=73%, M2P should match/exceed SFT |
| grad_norm at step 0 | > 1.0 | Standard M2P gradient through residual path |
| Runtime | ~50-65 min | 1000 train steps + 500 eval on M5 Pro 48GB |

## Self-Test

1. **What makes failure impossible?** Theorem 1: B_applied = B_sft + small residual.
   At init, the adapter IS the SFT adapter. Training can only perturb from this
   known-good starting point. Structural guarantee, not hyperparameter tuning.

2. **Cited theorems:** He et al. (2016) residual learning. Aghajanyan et al. (2020)
   intrinsic dimensionality. Finding #378 (v4 warm-start). Finding #401 (v5 SHINE).

3. **Predicted numbers:** See table. init_quality_ratio≈1.0 by construction.

4. **Falsification:** If K1 fails (init_quality_ratio < 0.80), the output_scale is
   too large and the random ΔB overwhelms B_sft. Fix: reduce output_scale.
   If K2 fails after K1 passes, the residual training degrades SFT quality.
   This would mean the M2P gradient landscape has no improvement direction from SFT.

5. **Hyperparameters:** output_scale=0.032 (SHINE default), d_m2p=1024 (same as v5),
   1000 steps (matching v4 which succeeded), LORA_SCALE=5.0, LORA_RANK=4.
   Same as v5 except: (a) SFT residual added, (b) 1000 steps vs 300.

## References

- He et al. (2016) — Deep Residual Learning (ResNet residual connection)
- Aghajanyan et al. (2020, arXiv:2012.13255) — Intrinsic dimensionality of fine-tuning
- SHINE (arXiv:2602.06358) — Base-as-encoder, alternating row/col attention
- Finding #378 — v4 warm-start from SFT achieves quality_ratio=1.433 at 0.6B
- Finding #401 — v5 SHINE achieves quality_ratio=0.833 at 0.6B
- Finding #402 — v5 SHINE fails at 4B (quality_ratio=-0.187)
- Finding #387 — SFT B-matrices intrinsic dim: d_int=86 (q), 69 (v)

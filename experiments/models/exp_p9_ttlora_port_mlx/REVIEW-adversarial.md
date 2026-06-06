# Adversarial Review: exp_p9_ttlora_port_mlx

**Verdict: PROCEED**

## What Was Tested

Port of TT-LoRA (arXiv:2504.21190) tensor train decomposition to MLX for Gemma 4 E4B attention projections (q/v/o_proj). Three kill criteria: forward pass consistency, parameter count, latency.

## Assessment

**K1 (Consistency): PASS** -- 0.0 diff across all projections. Both paths compute the same matmul after reconstruction, so exact agreement is expected. Sufficient for a porting verification.

**K2 (Param Count): PASS** -- 5,584 params (q+v) vs 40K threshold. 8.3x compression over LoRA r=6. Measured values correct.

**K3 (Latency): PASS** -- Cached mode max 1.36x vs LoRA. Well within 2x threshold.

## Minor Issue (Non-blocking)

MATH.md Theorem 1 prediction table computes q_proj as `8(5+8) + 64*6*8 = 3,176`, using average interior factor size s_bar=8. But the actual TT shape [5,8,8,8,4,8,8,8] has one interior core with factor 4, giving s_bar=7.33 and true count 2,920. PAPER.md correctly acknowledges this discrepancy. The formula derivation itself is correct; only the numeric application in the prediction table is off.

## Status Appropriateness

SUPPORTED is correct. This is a porting experiment (guided exploration, Type 2). Math predictions mostly verified, with explained discrepancy. Behavioral quality testing is correctly deferred to exp_p9_ttlora_quality.

## Composition Claim

PAPER.md states "TT-LoRA adapters compose identically to LoRA after reconstruction." This is correct by construction: once you materialize Delta-W, composition is W + sum(alpha_i * Delta-W_i) regardless of how Delta-W was parameterized. No additional proof needed.

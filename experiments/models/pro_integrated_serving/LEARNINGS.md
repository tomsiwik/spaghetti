# LEARNINGS.md: exp_pro_integrated_serving

## Core Learning

**The integrated serving pipeline (block-diagonal masking + per-token MLP routing +
DARE + ridge router) replicates from BitNet-2B-4T to Qwen3-4B-4bit without degradation.**
K821 PASS (behavioral 0.364 >= 0.3). The sign flip replicates: integrated is -3.4% BETTER
than isolated oracle and -6.2% BETTER than per-sequence baseline. However, the most likely
explanation for the vs-isolated improvement is an attention LoRA asymmetry confound (integrated
uses base attention, baselines use attention LoRA at scale=5), not pipeline composition benefit.

## What Worked

1. **Cross-architecture replication.** The pipeline transfers from BitNet-2B-4T (ternary,
   squared ReLU, 24 layers) to Qwen3-4B-4bit (4-bit quantized, SiLU, QK-norm, GQA, 36
   layers) without any modifications to the composition mechanism. The structural guarantees
   (RoPE invariance, MLP token-independence, DARE unbiased estimator) hold for both
   architectures as predicted by the math.

2. **Ridge router at d=2560.** 98.0% accuracy (49/50) with zero iterative training.
   Higher hidden dimension provides excellent domain separation. Slightly lower than
   BitNet's 100% (50/50), possibly due to 4-bit quantization noise in hidden states.

3. **Scale=5 composition is safe.** No MMLU catastrophe (consistent with Finding #320/#330).
   Behavioral quality maintained at 0.364 across 5 domains. Scale=5 is the validated
   operating point for multi-adapter composition.

4. **Prefill speed measured.** 1209.5 tok/s for the integrated forward pass (218 tokens).
   This is the first actual measurement of integrated pipeline throughput (tiny experiment
   only measured single-adapter generation). Block-diagonal mask + per-token MLP routing
   overhead is negligible at prefill time.

## What Didn't Work

1. **Weak domains persist.** Legal (0.072) and finance (0.086) behavioral scores are
   near-zero. This is a training quality issue (SFT loss 3.1/3.3 per Finding #319),
   not a pipeline issue. Scale=5 makes these even weaker than at scale=20, but scale=20
   destroys MMLU. Better training data/procedures needed for these domains.

2. **Sign flip remains unexplained.** The conjecture predicted additive degradation;
   measurement shows improvement. The -3.4% vs isolated improvement is MOST LIKELY due
   to the attention LoRA asymmetry confound: integrated pipeline uses base attention (no
   LoRA on q/k/v/o_proj), while both baselines apply RuntimeLoRA to all 7 projections.
   At scale=5 (adapters trained at scale=20), attention LoRA perturbation is poorly
   calibrated and likely harmful. This is NOT a proven explanation -- it requires
   an MLP-only isolated control to confirm.

3. **Generation speed unknown.** 32.5 tok/s measures single-adapter mlx_generate, not
   the integrated pipeline's autoregressive generation. The critical speed question
   for production serving remains unanswered.

## What We Still Don't Know

1. **Is attention LoRA at scale=5 harmful?** If yes, MLP-only adapters are strictly
   better for composition at inference-time scale mismatch. This would simplify the
   architecture (fewer LoRA modules) and explain the sign flip across both experiments.

2. **Autoregressive generation speed.** Prefill is fast (1209 tok/s) but generation
   requires per-token routing decisions and possibly per-token adapter switching.

3. **K > 2 segments.** Only 2-segment pairs tested. The 4 untested pairs include
   legal+finance (both weak) and math+legal (strong+weak) which may behave differently.

4. **Boundary detection in production.** Both tiny and pro use oracle domain labels.
   Automatic boundary detection is required for real serving and is unsolved.

## Literature Connections

- **"A Note on LoRA" (2404.05086):** MLP-only LoRA is competitive with or superior
  to full attention+MLP LoRA in multiple settings. Attention-only LoRA "significantly
  underperforms MLP-only LoRA and does not further improve performance on top of
  LoRA-on-MLP." This directly supports the attention LoRA asymmetry hypothesis: if
  MLP-only is sufficient, adding attention LoRA at mismatched scale could be harmful.

- **"LoRA Learns Less and Forgets Less" (2405.09673):** LoRA on different module types
  has different forgetting/learning tradeoffs. MLP adapters learn domain knowledge;
  attention adapters learn task formatting. At reduced inference scale, the formatting
  signal from attention LoRA may produce harmful perturbations.

- **rsLoRA "Rank Stabilization" (2312.03732):** Scale mismatch between training and
  inference is a known failure mode. Our scale=5 at inference vs scale=20 at training
  is a 4x mismatch, which could disproportionately affect attention projections
  (which have different spectral properties than MLP projections).

- **Block-Attention (2409.15355):** Our implementation matches their block-diagonal
  masking. Their re-encoding step (which we proved unnecessary in Finding #322)
  was for KV-cache portability, not quality.

- **Finding #292 (Pierre v6):** Attention-only adapters dropped code behavioral by
  67% (0.844 -> 0.281). "MLP adapters critical for code domain." This is CONFIRMING
  evidence that attention adapters are not universally helpful and MLP is the
  critical component for domain specialization.

- **Finding #323 (tiny integrated serving):** Same sign flip (-2.8% on BitNet).
  Same code-path confound. The replication on a completely different architecture
  (different activation, quantization, head structure) rules out architecture-specific
  artifacts. The confound is systematic.

- **Finding #330 (scale calibration):** Scale=5 is the validated operating point.
  Scale=13 gives -4pp MMLU (marginal). Scale=20 catastrophic. The pipeline correctly
  uses the proven safe scale.

## Generalizable Insight

**Inference-time scale mismatch affects attention and MLP LoRA differently.** When
adapters are trained at scale=20 and applied at scale=5, the 4x reduction may be
beneficial for attention (where the out-of-calibration perturbation is harmful) while
merely weakening MLP (where the perturbation is in the right direction but reduced).
This suggests MLP-only adapters may be the correct choice for composition at mismatched
scale -- simpler architecture, fewer parameters, and potentially better quality.

This is a hypothesis, not a proven result. The MLP-only isolated control experiment
is the critical test.

## Recommended Follow-ups

1. **exp_pro_mlp_only_isolated_control (P0):** Run isolated oracle with MLP-only
   adapters (detach attention LoRA). If integrated vs MLP-only isolated is ~0%, the
   sign flip is from dropping attention LoRA, not composition benefit. This is the
   single most important experiment for understanding the pipeline's actual mechanism.
   *Motivation:* Findings #332/#323 (sign flip replicates), Finding #292 (attention
   LoRA harmful for code), "A Note on LoRA" (2404.05086, MLP-only competitive).

2. **exp_pro_integrated_generation_speed (P0):** Measure actual autoregressive
   generation speed with the integrated forward pass (per-token routing + block-diagonal
   mask). This is the critical production viability question unresolved since Finding #323.
   *Motivation:* Finding #323 K819 FAIL, Finding #332 generation speed still unmeasured.

3. **exp_scale_per_module_type (P1):** If MLP-only isolated control confirms attention
   LoRA is harmful at scale=5, test whether per-module-type scale (scale=5 for MLP,
   scale=0 for attention) improves quality without the full MLP-only simplification.
   *Motivation:* rsLoRA (2312.03732, scale sensitivity), Finding #326-328 (scale is
   the disease, not adapter directions).

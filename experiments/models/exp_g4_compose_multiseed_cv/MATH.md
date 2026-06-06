# MATH — exp_g4_compose_multiseed_cv

## Claim (KC #1590)
`CV(MMLU-Pro accuracy) <= 2%` across 3 independent seeds of identical 5-adapter
Gemma 4 E4B 4-bit composition on the same test split.

## Theorem (Reproducibility under Lipschitz composition)
Let `f_s : X → Y` be the Gemma-4 base with deterministic forward at seed `s`, and
`A_i^{(s)}` the five LoRA adapters trained with seed `s` (data order, init noise,
dropout). Under equal-weight composition `W_s = W_0 + Σ_i A_i^{(s)}` and a
Lipschitz evaluation map `M` from logits to accuracy,

```
|M(f_{s₁}) − M(f_{s₂})| ≤ L · ||W_{s₁} − W_{s₂}||
```

with `L` bounded by the gradient norm of `M` on the test distribution. For CV ≤
2% we need the three composed weight matrices to lie within a ball of radius
`ε ≤ 0.02·μ/L`. Whether this holds is an *empirical* claim about adapter
variance; the proof merely establishes the chain — if the variance is larger than
`ε/L` the KC KILLs.

## Preconditions (must ALL hold before claim is measurable)
- **P1** — 3 seeds × 5 Gemma 4 r=6 LoRA adapter checkpoints on disk
  (15 safetensors total).
- **P2** — upstream `exp_p1_t2_single_domain_training` landed at `LORA_SCALE=5`,
  `max_tokens ≥ 512`, rank sweep logged, grad-SNR logged (cohort blocker, see
  Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618).
- **P3** — MMLU-Pro evaluator harness that reproduces the baseline number used
  by prior cohort experiments.

## Pre-registered tripwire
If *any* of P1/P2/P3 fail at probe time → `K1590 UNMEASURABLE → status=killed`.
Heavy training (~6 h across 3 seeds) is **forbidden** until preconditions pass.

## Expected probe cost
< 5 s wall, pure file-existence + upstream-verdict read. No MLX.

## References
- Finding #43 (bitnet_multiseed_validation)
- Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618 (cohort blocker)

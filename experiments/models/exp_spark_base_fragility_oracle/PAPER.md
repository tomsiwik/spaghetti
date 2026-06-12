# PAPER — exp_spark_base_fragility_oracle

## Claim under test
H1: A per-token interference predictor reading **zero adapter information** — the frozen base
model's local curvature (KL response to fixed-norm random Gaussian displacement of the mid-layer
hidden state) — ranks which tokens the (1/N)-average LoRA composition damages most.

If true, this would escape the dead class of delta-reading predictors (F#864, F#867, F#868):
interference would be a property of the **frozen base geometry**, not the adapter delta.

## Setup (as run)
- Base: `mlx-community/gemma-4-e4b-it-4bit`, 42 decoder layers, d=2560. REAL model, REAL adapters.
- Adapters: math (gsm8k), python (codealpaca), medical (medmcqa), r=6 q_proj, scale s=6.0.
- Composition: `y = Wh + (1/N) Σ sᵢ Bᵢ Aᵢ`, N=3 (the average merge, never (ΣB)(ΣA)).
- Mid layer ℓ*=21. fragilityₜ = mean over K=8 seeded random draws of KL(Pᵦ(·|hₜ) ‖ Pᵦ(·|hₜ+εₜₖ)),
  ε = ρ·‖hₜ‖·u, ρ=0.08, EPS_SEED=20250609. No adapter tensor enters fragilityₜ.
- damageₜ = base_logprob(y_{t+1}) − composed_logprob(y_{t+1}) on held-out domain text.
- n_tokens_scored = 3828. is_smoke = false. Wall clock 42.4 s.

## Prediction vs. measurement

| Quantity                          | Predicted (H1) | Measured | Kill floor | Crossed? |
|-----------------------------------|----------------|----------|------------|----------|
| Spearman ρ(fragility, damage)     | ≥ 0.30         | **0.0085** | < 0.30   | YES      |
| Top-decile-damage AUC             | ≥ 0.62         | **0.5738** | < 0.62   | YES      |

Supporting measurements: mean_damage = −0.1032 (composition net-*helps* on average),
frac_tokens_damaged = 0.352, top-decile split = 387 positive / 3441 negative.

## Pre-registered kill 2305 (verbatim)
> Spearman rho < 0.30 between base-fragility and per-token interference damage, OR
> top-decile-damage AUC < 0.62

- clause_rho_below_0_30: **TRUE** (0.0085 < 0.30) — fired
- clause_auc_below_0_62: **TRUE** (0.5738 < 0.62) — fired

Either clause alone is sufficient to kill; **both** fired.

## Verdict
**KILLED.** Base-geometry curvature carries no rank information about per-token composition damage.
Spearman ρ = 0.0085 is statistically indistinguishable from zero (chance = 0), and top-decile AUC
= 0.574 is barely above the 0.50 chance line and well below the 0.62 floor. The frozen-base
isotropic local Lipschitz response does **not** predict where the adapter merge hurts.

## Interpretation
The hypothesized mechanism — "where the base output map is locally steep, *any* fixed-norm
displacement (random or adapter) produces large output movement, hence large damage" — is refuted.
Interference is **not** dominated by isotropic base curvature: a random-direction probe that knows
nothing about the structured Σ(BᵢAᵢ)hₜ displacement direction cannot rank damage. This is direct
evidence that composition damage depends on the **alignment** between the adapter delta direction
and the base geometry, not on the base's direction-averaged steepness alone. The escape from the
delta-reading predictor class therefore fails: a zero-adapter-information predictor recovers no
signal. Combined with prior delta-reading failures, no per-token interference oracle survives here.

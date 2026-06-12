# MATH — exp_spark_base_fragility_oracle

## Disease being prevented
The entire dead class of composition-interference predictors reads the **adapter delta**
(magnitude / sign / loudness / timing — F#864, F#867, F#868). If interference is instead a
property of the **frozen base geometry**, then a predictor that touches **zero adapter bytes**
should still rank which tokens composition damages most. We pre-register a falsifiable test.

## Setup
Frozen base `mlx-community/gemma-4-e4b-it-4bit` (42 decoder layers, hidden d=2560).
Three r=6 q_proj adapters (math/gsm8k, python/codealpaca, medical/medmcqa), scale s=6.0.

Let layer ℓ* = 21 (mid stack). For a token t with mid-layer hidden state hₜ ∈ ℝ^d:

- **Base next-token distribution** Pᵦ(·|hₜ): push hₜ through layers ℓ*+1..41, final norm, tied
  head, softcap — i.e. the unmodified base model's output at position t.
- **Perturbed distribution** Pᵦ(·|hₜ+εₜ): identical downstream stack, but hₜ replaced by hₜ+εₜ.

## Predictor (ZERO adapter information)
ε is a **fixed-norm RANDOM Gaussian** direction (NOT any adapter delta), seeded `EPS_SEED=20250609`.
Per token, per draw k∈{1..K}, K=8:

    uₜₖ ~ N(0, I_d)/‖·‖ ,  εₜₖ = ρ · ‖hₜ‖₂ · uₜₖ ,  ρ = 0.08 (fixed noise fraction)

    fragilityₜ = (1/K) Σₖ KL( Pᵦ(·|hₜ) ‖ Pᵦ(·|hₜ+εₜₖ) )

Averaging over K=8 seeded draws removes single-draw direction artifacts. No adapter tensor,
shape, sign, or magnitude enters fragilityₜ — it is a curvature probe of the frozen base alone.

## Label (per-token composition damage)
Composed model = base + (1/N) Σᵢ sᵢ Bᵢ Aᵢ on q_proj, N=3 (math+python+medical), the (1/N)
average merge. For the **true next token** y_{t+1} of held-out domain text:

    damageₜ = base_logprob(y_{t+1}) − composed_logprob(y_{t+1})

(positive = composition hurt that token). Held-out text = unseen prompts from gsm8k / medmcqa /
code_alpaca (same sources the adapters trained on, disjoint items).

## Prediction
H1: base-curvature fragility (random noise, no adapter info) ranks per-token composition damage.
Predicted: Spearman ρ(fragility, damage) ≥ 0.30 AND top-decile-damage AUC ≥ 0.62.

Mechanism: composition perturbs hₜ by a structured Σ(BᵢAᵢ)hₜ of bounded norm; where the base
output map is locally steep (high KL response to *any* small displacement of fixed norm), *any*
perturbation of comparable size — random or adapter — produces large output movement, hence large
damage. Fragility is the isotropic (random-direction-averaged) local Lipschitz response of the
frozen base; it lower-bounds nothing about the adapter but should *correlate* with realized damage
if base curvature is the dominant factor.

## Refutation threshold (pre-registered kill 2305, verbatim)
> Spearman rho < 0.30 between base-fragility and per-token interference damage, OR
> top-decile-damage AUC < 0.62

If EITHER clause is crossed → verdict = killed. Thresholds fixed before the run; no post-hoc move.

## Guards
- ε seed reported (`EPS_SEED=20250609`); no `Math.random`, no `Date`, no time-derived seed.
- Composition is Σᵢ (1/N) sᵢ Bᵢ Aᵢ, never (ΣB)(ΣA). s=6.0 ≤ 8.
- AUC threshold is two-sided informative only via the floor; chance AUC = 0.50.
- is_smoke:false; real model, real adapters, real held-out tokens.

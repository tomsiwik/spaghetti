# MATH — exp_spark_logit_decile_clash

## Frame-break claim

Off-domain composition damage is NOT a static weight-space property of the adapter delta
(three geometric signals already died: magnitude F#864, isotropic curvature F#867/868,
alignment-angle F#869). It is an **output-distribution clash measured at the logit head**:
damage spikes exactly where the adapter's top positive logit-shift mass lands on tokens the
**frozen base already pruned** (placed in its bottom probability decile). The adapter
*resurrects base-pruned tokens* instead of re-ranking plausible ones.

## Definitions (per token t, single off-domain prompt set)

Let `z0_t = base logits` (frozen base, single forward) and `zA_t = composed logits`
((1/N) Σ sᵢ BᵢAᵢ on q_proj, single forward). Define the composed logit shift

    dz_t[v] = zA_t[v] − z0_t[v]          (V-vector, the adapter's effect at the head)

Let `p0_t = softmax(z0_t)` (base distribution). Bottom-decile set of the base:

    D_t = { v : p0_t[v] ≤ q10(p0_t) }    (tokens the base assigned bottom-10% probability)

Top-K positive-shift set (K = 128):

    Kset_t = topK_v relu(dz_t[v])

**Clash signal** (fraction of the adapter's positive logit-shift mass aimed at base-pruned tokens):

    clash_t = Σ_{v ∈ Kset_t ∩ D_t} relu(dz_t[v])  /  Σ_{v ∈ Kset_t} relu(dz_t[v])   ∈ [0,1]

## Labels (damage), composed = (1/N) merge, consistent with F#627/2305 line

- Primary:  `kl_damage_t = KL(p0_t ‖ pA_t)`  where `pA_t = softmax(zA_t)`.
- Secondary: `nll_damage_t = base_logprob(y_{t+1}) − composed_logprob(y_{t+1})`.

## In-script geometric baseline (computed on the SAME tokens, SAME prompts)

To make kill 2306 well-defined WITHOUT importing a number from another token set, we compute
geometric predictors in THIS script on identical (prompt, token) data:

- `align_angle_t`  = angle between the composed mid-layer hidden delta Δh_t and the base hidden
  h_t at L* (the F#869 alignment-angle predictor). Signal = −cos(angle) (or |·|; we report the
  better-correlating sign, magnitude of rho).
- `delta_mag_t`    = ‖Δh_t‖ / ‖h_t‖ at L* (the F#864 delta-magnitude predictor).

    best_geometric_predictor_rho = max( |rho(align_angle, kl_damage)|,
                                        |rho(delta_mag,  kl_damage)| )

This is the in-script baseline the clash signal must beat by +0.15.

## Prediction (what success looks like)

The hypothesis predicts the clash relocates the predictor from weight geometry (where 3 signals
died, expected |rho| ≲ 0.30) to the base's own output prior:

    rho(clash, kl_damage) ≥ 0.55, and ≥ best_geometric_predictor_rho + 0.15.

## Pre-registered KILL 2306 (verbatim)

> "Spearman(clash_signal, per-token KL damage) < 0.45 OR clash_rho < best_geometric_predictor_rho + 0.15"

Refutation thresholds (numeric, fixed before run):
- clause A: `rho(clash, kl_damage) < 0.45`
- clause B: `rho(clash, kl_damage) < best_geometric_predictor_rho + 0.15`
- killed if (A OR B). supported otherwise. `is_smoke:false`.

## Off-domain prompt set (explicit, pre-registered)

The adapters are trained on math/python/medical. "Off-domain" damage is measured on prompts
OUTSIDE all three training domains. Fixed set, real HF datasets, held-out splits:
- general English / commonsense:  `wikitext-2-raw-v1` (test split, first N paragraphs)
- open-domain QA prose:           `squad` (validation, context paragraphs)
- everyday instruction-following: `tatsu-lab/alpaca` items whose text contains none of
  {def, return, import, mg, dose, patient, =, solve, theorem} (filters out code/medical/math).

These contain no math/code/medical adapter-domain content, so any composed shift is pure
off-domain interference, not on-domain reinforcement.

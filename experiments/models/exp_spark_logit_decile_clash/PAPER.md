# PAPER — exp_spark_logit_decile_clash

## Claim under test

Off-domain composition damage is an **output-distribution clash at the logit head**: damage
spikes where the adapter's top positive logit-shift mass (top-K=128 of relu(dz)) lands on tokens
the frozen base already pruned (its bottom-decile probability mass). If true, the clash signal
should predict per-token KL damage better than weight-space geometry, relocating the predictor
from the adapter delta to the base's own output prior.

## Method (real, not mock)

- Base: `mlx-community/gemma-4-e4b-it-4bit`. Adapters: math, python, medical (real
  `adapters.safetensors`), composed as the (1/N) average merge `y = Wh + (1/N) Σ s·BᵢAᵢ` on
  q_proj, lora_scale 6.0.
- Off-domain prompts (48) drawn from wikitext-2-raw-v1/test, squad/validation, and
  domain-filtered tatsu-lab/alpaca — none containing math/code/medical content.
- 4117 tokens scored. Per token: base logits z0 and composed logits zA from real forward passes;
  clash_t per MATH.md; labels kl_damage = KL(p0‖pA) and nll_damage.
- In-script geometric baselines computed on the SAME (prompt, token) data: alignment-angle
  (F#869) and delta-magnitude (F#864).
- `is_smoke: false`, wall clock 24.6 s.

## Prediction vs measurement

| Quantity | Predicted | Measured |
|---|---|---|
| rho(clash, kl_damage) | ≥ 0.55 and ≥ best_geo + 0.15 | **0.0241** |
| rho(clash, nll_damage) | (secondary) | −0.0158 |
| best_geometric_predictor_rho | ≲ 0.30 (geometry already dead) | 0.2076 (delta_mag) |
| threshold_to_beat (best_geo + 0.15) | — | 0.3576 |

### In-script geometric baselines (same tokens)

- alignment-angle vs KL: rho = **−0.0513**
- delta-magnitude vs KL: rho = **0.2076**  ← best geometric predictor
- best_geometric_predictor_rho = **0.2076** (predictor = `delta_mag`)

## KILL 2306 evaluation

> "Spearman(clash_signal, per-token KL damage) < 0.45 OR clash_rho < best_geometric_predictor_rho + 0.15"

- **Clause A** (rho < 0.45): 0.0241 < 0.45 → **FIRED (true)**
- **Clause B** (clash_rho < best_geo_rho + 0.15 = 0.3576): 0.0241 < 0.3576 → **FIRED (true)**

Both clauses fired. The criterion is (A OR B), so either alone is sufficient; both triggered.

## Verdict

**KILLED.** The logit-decile clash signal carries essentially zero rank correlation with
per-token KL damage (rho = 0.0241, and −0.0158 vs NLL). It does not approach the 0.45 floor and
does not beat even the weak surviving geometric baseline (delta-magnitude, rho = 0.2076) — it is
~9× weaker. The hypothesis that damage is the adapter "resurrecting base-pruned tokens" is
refuted: the clash fraction is also small in aggregate (mean_clash = 0.081; only 2.2% of tokens
have clash > 0.5), so the proposed mechanism is both weakly correlated and rarely active.
Weight-space geometry (delta_mag) remains the least-bad predictor, but at rho ≈ 0.21 the
off-domain damage signal is still not explained by any tested geometric or output-distribution
quantity.

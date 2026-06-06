# MATH.md — Beehive→mlx_lm Export Pipeline Validation

## Experiment type
**Verification** — PoLAR training pipeline on real beehive data, confirming convergence and Stiefel maintenance.

## Hypothesis
A PoLAR adapter (rank-6, scale-6, q_proj, joint Stiefel retraction every 20 steps) trained on 104 approved beehive trajectories will:
1. Converge without divergence over 1000 steps
2. Not regress standard benchmarks (GSM8K, HumanEval, MedQA) by more than 2pp
3. Improve principle-following on held-out beehive eval by ≥5pp vs base
4. Maintain Stiefel manifold constraints post-final-retraction

## Theoretical grounding

### PoLAR convergence (F#442, F#444)
Joint Stiefel retraction via SVD projection maintains ||A^T A − I||_F < ε after retraction. With retraction every 20 steps and gradient clipping at 1.0, the adapter delta ΔW = scale · A · B stays bounded. The scale=6 is within the safe range (≤8 per PLAN.md antipattern #003).

**Citation:** F#442 verified joint Stiefel on Gemma 4 native dims. F#444 confirmed PoLAR scale stability.

### No-regression guarantee
The adapter modifies only q_proj (F#421 baseline). With rank=6 on d_model=2048, the rank-6 perturbation affects at most 6/2048 ≈ 0.3% of the projection's effective dimensionality. Cross-entropy loss is Lipschitz-continuous w.r.t. weight perturbations under bounded activation norms, so small-rank perturbations produce bounded output changes.

### Principle-following as target metric
Raw loss decrease is a proxy. The target metric is principle-following score on held-out trajectories — does the model produce outputs aligned with beehive principles (format, keywords, structure)? This satisfies the target-gated kill rule (Finding #666).

## Predictions

| Metric | Prediction | Source |
|--------|-----------|--------|
| Final loss | < first loss (monotonic trend, window-20 smoothed) | PoLAR convergence theory |
| NaN/divergence | None | Gradient clipping + Stiefel retraction |
| GSM8K regression | < 2pp | Rank-6 perturbation bound |
| HumanEval regression | < 2pp | Rank-6 perturbation bound |
| MedQA regression | < 2pp | Rank-6 perturbation bound |
| Principle-following gain | ≥ 5pp over base | Supervised signal from approved trajectories |
| ||A^T A − I||_F post-retraction | < 0.01 | SVD projection guarantee (F#442) |
| ||B B^T − I||_F post-retraction | < 0.01 | SVD projection guarantee (F#442) |

## Kill criteria (pre-registered)

| KC | ID | Threshold | Type |
|----|-----|-----------|------|
| K1 | 2069 | Loss decreases monotonically (window-20); no NaN/divergence | proxy |
| K2 | 2070 | No benchmark regression > 2pp on any of GSM8K/HumanEval/MedQA | target |
| K3 | 2071 | Principle-following ≥ base + 5pp on held-out 20% | target |
| K4 | 2072 | max ||A^T A − I||_F < 0.01 AND max ||B B^T − I||_F < 0.01 | proxy |

K2 and K3 are target metrics; K1 and K4 are proxy metrics. Per Finding #666, kill requires both proxy AND target failure.

## Data
- 104 approved beehive trajectories (score 8–9)
- 80/20 stratified split by skill+type
- Types: act (62 total), full (59), integrate (51), prepare (9) — approved subset
- Format: `{messages: [{role: "user", content: skill+type+principle}, {role: "assistant", content: trajectory}]}`

## Platform
- `mlx-community/gemma-4-e4b-it-4bit` on M5 Pro 48GB
- MLX lazy evaluation with mx.eval at step boundaries
- `mlx_lm` version: current installed

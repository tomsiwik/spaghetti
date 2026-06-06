# E1: Function Vector Strategy Extraction — Results

## Summary

Extracted activation difference vectors (strategy prompt - neutral prompt) from Gemma 4 E4B o_proj outputs across 42 layers, for 3 strategies (systematic, step-by-step, conservative). Measured consistency, injected as LoRA, and evaluated GSM8K behavioral change.

## Prediction vs Measurement

| Prediction | Threshold | Measured | Verdict |
|---|---|---|---|
| P1: Strategy vectors consistent across prompts (K#2017) | cos > 0.1 | cos=0.81–0.97 (mean across layers) | PASS |
| P2: Injected LoRA produces GSM8K improvement (K#2018) | > 2pp | 0.0pp (30% → 30%) | FAIL |
| P3: Cross-prompt consistency (K#2019) | cos > 0.3 | cos=0.95 (mean) | PASS |

## Critical Finding: Vectors Are Strategy-Undifferentiated

Cross-strategy cosine similarities at the best extraction layer (41):

| Pair | Cosine |
|---|---|
| systematic vs step_by_step | 0.9922 |
| systematic vs conservative | 0.9883 |
| step_by_step vs conservative | 0.9883 |

The extracted vectors are nearly identical across all three strategies. This means the mean-difference method captures "system prompt present vs absent" — not the strategy-specific content. K1 and K3 pass trivially because they measure within-strategy consistency of a signal that is actually strategy-invariant.

## Mechanism Analysis

The Function Vectors paper (2310.15213) extracts vectors from attention heads that encode specific input-output functions (e.g., "capitalize", "translate to French"). These are narrow, well-defined transformations. Problem-solving strategies are not narrow functions — they are broad behavioral modes that modulate reasoning across many tokens and layers. The mean-difference approach conflates:

1. **Format signal**: The presence of a system prompt changes token distributions at every layer (instruction-following mode vs. bare-question mode). This dominates.
2. **Strategy signal**: The specific reasoning approach (decompose vs. verify vs. be-cautious) is a much smaller perturbation on top of the format signal.

cos(format_signal, strategy_signal) ≈ 1.0 because format >> strategy in activation magnitude.

## KC Verdicts

- **K#2017 (cos > 0.1)**: PASS — but vacuously, because the signal is format, not strategy.
- **K#2018 (> 2pp GSM8K)**: FAIL — 0.0pp delta. Injected adapter produced no behavioral change.
- **K#2019 (cross-prompt cos > 0.3)**: PASS — but again vacuously, because all strategies produce the same vector.

## Verdict

**PROVISIONAL (smoke, N=10 GSM8K)** — Strong kill signal. The cross-strategy cos > 0.98 shows the extraction method does not isolate strategy-specific directions. K2 FAIL at 0pp confirms no behavioral transfer. A full run is unlikely to change the 0pp delta or the 0.99 cross-strategy similarity.

## What Would Fix This

To salvage the function-vector approach for strategy extraction:
1. **Contrastive extraction**: Instead of (strategy - neutral), use (strategy_A - strategy_B) to cancel the shared format signal. This requires a pairwise design.
2. **Residual extraction**: First subtract the mean-of-all-strategies vector, then measure per-strategy residuals.
3. **Head-level selection**: Function Vectors identifies specific attention heads per function. Averaging across all heads dilutes strategy-specific signals.

## Run Details

- Model: mlx-community/gemma-4-e4b-it-4bit (42 layers)
- Smoke: 10 prompts/strategy, 10 GSM8K problems
- Adapter: r=8, scale=6.0, targets=v_proj+o_proj
- Best extraction layer: 41 (final)
- Runtime: 43.6s

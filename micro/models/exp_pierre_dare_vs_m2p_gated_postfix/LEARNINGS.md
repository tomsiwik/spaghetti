# Learnings: DARE vs M2P-gated

## Core Finding

Continuous-weight composition (M2P-gated) underperforms stochastic binary masking (DARE) by 10pp even with perfect routing (99.6% accuracy). The gate learns *which* adapters to use but not *how much* — gate confidence has zero correlation with correctness (ρ=0.097).

## Why

DARE's 90% drop rate acts as implicit regularization: it zeroes weak/noisy parameters, forcing the model onto robust weight directions. Continuous weighting preserves everything equally, and without explicit regularization the gated sum overflows at scale (observed NumPy overflow in `compute_gated_deltas`).

## Implication

For Pierre's adapter composition: use DARE (binary mask + rescale) as the production merge strategy. If continuous weighting is revisited, it needs (1) explicit weight regularization and (2) numerical stability (clamp/normalize before scaling). But the complexity is not justified — DARE already achieves 73.3% avg across 3 benchmarks with zero learned parameters in the merge step.

## Reusable Findings

- TF-IDF+Ridge gate routes with 99.6% holdout accuracy at 0.07ms — validated zero-cost routing primitive (confirmed across two experiments now)
- Gate confidence is a domain classifier, not a difficulty estimator — don't use it for calibration
- DARE drop=0.9 is stable: 73.3% here matches exp_pierre_per_task_routing_math baseline exactly

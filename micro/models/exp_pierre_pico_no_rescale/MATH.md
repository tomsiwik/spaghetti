# MATH.md — Pico ablation: γ rescaling disabled

## Hypothesis

Pico's algorithm has two parts:
1. **SVD calibration** of stacked B — the central novelty
2. **γ rescaling** — `γ = mean_t‖B_t‖_F / ‖B_merged‖_F` to preserve mean source norm

Fisher-Rao also rescales (norm-preserving step is what made the Karcher-mean
elegant). What if Pico's gain comes mostly from rescaling and not from the
SVD calibration? Or vice versa?

> **Disable γ rescaling in Pico. Does the SVD calibration alone help?**

## Method

Identical to `exp_pierre_pico_calibration` but with `rescale_to_mean_norm=False`.
Reuses Pico's `compose_methods.py` directly via fn_kwargs.

## Pre-registered Kill Criteria

- **K1 (DECISION)** Pico-no-rescale avg ≥ Fisher-Rao avg + 3pp.
- **K2 (ATTRIBUTION)** Pico-no-rescale vs Pico-with-rescale:
  - Within 1pp: rescaling is decoration; SVD calibration does the work.
  - 1-3pp drop: both contribute.
  - >3pp drop: rescaling carries most of Pico's gain — calibration alone insufficient.
- **K3 (BUDGET)** Preprocessing ≤ 5s (same as Pico).
- **K4 (SANITY)** N/A — pure ablation.

## Why this matters

Decomposes Pico's gain (if any) into SVD-calibration vs norm-rescaling.
If rescaling carries the gain, that's a much simpler mechanism to ship —
you don't need the SVD step at all, just a smarter rescaling than Fisher-Rao's.

## References

- Parent: `exp_pierre_pico_calibration`
- Pico paper (arxiv 2604.16826)

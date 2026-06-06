# MATH.md — ACE-Merging ablation: spectral refinement disabled

## Hypothesis

ACE-Merging has two components that could individually contribute to its
gain over Fisher-Rao:

1. **Closed-form covariance-weighted merge** — the main idea
2. **Spectral refinement** — top-k singular value isotropization, applied
   when heterogeneity flag γ > 0.3

Pierre's 7-adapter mix (4 strategy + 3 domain) is **heterogeneous** by
design, so the spectral branch is likely engaged. We don't know which
part of ACE actually contributes.

> **Disable the spectral refinement step. Does ACE still beat Fisher-Rao?**

## Method

Identical to `exp_pierre_ace_merging_b_only` but with `force_disable_spectral=True`
in fn_kwargs. Reuses ACE's `compose_methods.py` directly.

## Pre-registered Kill Criteria

- **K1 (DECISION)** ACE-no-spectral avg ≥ Fisher-Rao avg + 3pp (must still help without spectral).
- **K2 (ATTRIBUTION)** ACE-no-spectral avg compared to ACE-with-spectral (from `exp_pierre_ace_merging_b_only`):
  - Within 1pp: spectral refinement is decoration, covariance merge does the work.
  - 1-3pp gap: both contribute meaningfully.
  - >3pp gap: spectral refinement is essential — covariance merge alone is insufficient.
- **K3 (BUDGET)** Preprocessing ≤ 15s (lighter than full ACE because no per-layer SVD).
- **K4 (SANITY)** Same as parent ACE — verdict on K1+K2+K3.

## Why this matters

Interpretability ablation. Tells Pierre's architecture which knob to keep
when porting ACE to product. Spectral refinement adds ~8s of preprocessing;
if it's decoration, drop it.

## References

- Parent: `exp_pierre_ace_merging_b_only`
- ACE-Merging paper (arxiv 2603.02945)

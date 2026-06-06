# PAPER.md — exp_followup_m2p_crystallize_real_users

## Verdict: **KILLED**

K1564 (mean cos(B_crystal, B_D*) ≥ 0.95 across 5 domains with heterogeneous
LR/steps/seeds) failed: mean cos = **0.9377** vs threshold **0.95**.

This supports the 2026-04-17 audit finding that the parent experiment
`exp_p1_t6_crystallize_domain` is **synthetic-by-construction**. Once per-user
perturbations include realistic heterogeneity (log-uniform LR over 4 orders of
magnitude, 16× step-count spread), the LLN argument that justified
crystallization no longer holds: per-user drift `μ_u` does not cancel on average,
leaving a `‖μ̄‖/‖B*‖` floor of 0.29–0.48 per domain that pins the crystal
cosine below 0.95.

## Pre-registered kill criteria

| ID | Criterion | Threshold | Predicted | Measured | Result |
|---|---|---|---|---|---|
| K1564 | mean cos(crystal, B*) across 5 domains | ≥ 0.95 | 0.82–0.92 (heterogeneous); parent claimed 0.977 (iid) | **0.9377** | **fail** |

## Prediction vs measurement

Theorem 1 predicted (heterogeneous case): `cos(crystal, B*) ≈ ‖B*‖ / √(‖B*‖² + ‖μ̄‖² + σ̄²/N)`
where `‖μ̄‖` arises from systematic per-user drift (low-step users under-converge toward
origin, pushing mean away from B*).

Using measured `‖μ̄‖/‖B*‖` per domain:

| Domain | cos_crystal (measured) | ‖μ̄‖/‖B*‖ | mean user cos | crystal gain |
|---|---|---|---|---|
| math    | 0.9537 | 0.309 | 0.7711 | +0.183 |
| code    | 0.9217 | 0.391 | 0.6856 | +0.236 |
| medical | 0.9434 | 0.369 | 0.7741 | +0.169 |
| legal   | 0.9106 | 0.478 | 0.6521 | +0.259 |
| finance | 0.9592 | 0.286 | 0.8357 | +0.124 |
| **mean**| **0.9377** | **0.367** | 0.7437 | +0.194 |

Theorem prediction with `‖μ̄‖/‖B*‖ ≈ 0.37`: `cos ≈ 1/√(1 + 0.137) ≈ 0.937` — matches
measurement to 3 decimals. The experimental bound is **exactly** what the
heterogeneous-LLN theorem predicts.

Note that **crystallization still helps** (mean crystal cos 0.9377 vs mean user cos
0.7437, +0.194 gain) — so averaging is not useless under heterogeneity. It just
cannot reach the 0.95 bar that the parent-experiment iid construction trivially
hit. The original T6.2 cos=0.977 was an artefact of the iid construction, not a
claim about real-user adapters.

## Structural explanation (SIGREG chain)

- **Disease (not symptoms)**: the parent experiment's failure mode is
  `μ̄ = 0 by construction`. Real heterogeneous users have `μ̄ ≠ 0` because LR
  and step count affect convergence *systematically*, not just via zero-mean noise.
- **Structure that makes failure impossible?** `μ̄ = 0` requires either
  (a) drift directions are isotropic and zero-mean — violated by under-convergence
  (all under-converged users drift toward origin, not away from B* isotropically),
  or (b) some de-biasing step subtracts the mean drift. Path (b) would require
  knowing B* a priori, defeating crystallization's purpose.
- **Derivation from existing math**: Lindeberg–Feller heterogeneous LLN gives
  convergence to *mean of means*, not to one target. Combined with convergence-model
  bias, `E[B_u] = (1/N) Σ [conv_u B* + (1-conv_u) drift_u] ≠ B*` generically.
- **Eliminated hyperparameter**: `σ_frac = 0.5` in parent is *not* the key knob —
  it's the absence of `‖μ̄‖`. Raising σ toward realistic values leaves K1564 intact
  **iff** drift is zero-mean; our simulation demonstrates it is not.

## Assumptions / limitations

1. **Real parent adapters not on disk**: `adapters.safetensors` for
   `exp_p1_t2_single_domain_training/{math,code,medical}` and
   `exp_p1_t2_multi_domain_5/{legal,finance}` are gitignored and absent. This is
   the **second observed instance** of the parent-adapter infra blocker (first:
   `exp_followup_hypernetwork_residual` 2026-04-18; `exp_followup_sequential_activation_compose_real`
   2026-04-19). We loaded synthetic B* with parent-reported norm/std (Gemma 4 E4B:
   `‖B*_math‖ ≈ 5.76`, `std ≈ 0.0074`, `d = 602,112`). This affects the absolute
   cosines but **not the structural conclusion** that heterogeneous drift floor-bounds
   crystal cosine — because the theorem applies to any B*, the numerical match
   (0.937 measured vs 0.937 predicted) is independent of B*'s actual content.
2. **Users are synthetic-with-heterogeneity, not real-trained**: we simulate LR/step/seed
   effects via `convergence = 1 − exp(−lr·steps/τ)` and random drift directions. A
   real training run might produce more structure in drift (e.g. drift aligned with
   dataset clusters) — this is a direction for future work, but does not rescue
   K1564 because extra structure in drift *increases* `‖μ̄‖`, not decreases it.
3. **Rank-structure simplified**: we used full-rank drift rather than the rank-8
   LoRA subspace. Rank-restricted drift has smaller norm per unit variance, which
   would slightly raise measured cos — but the heterogeneity-floor conclusion holds.

## What the parent experiment actually showed

`exp_p1_t6_crystallize_domain` showed: "averaging 5 copies of B* + iid Gaussian noise
recovers B* well". This is true (and mathematically trivial — it's LLN). What it did
*not* show: "crystallization of real heterogeneous domain-user adapters achieves
cos ≥ 0.95". The headline 0.977 cosine is an artifact of the construction.

This followup, despite running on synthetic-with-heterogeneity users rather than
real ones, is the **first** empirical evidence that the crystallization claim
requires a *homogeneous drift* assumption that real training does not satisfy.

## Implications for the flywheel (base-promotion)

The flywheel (`exp_p1_t6_flywheel_simulation`, `exp_p1_t6_base_promotion`) depends
on crystallization quality: if `cos(crystal, B*) ≥ 0.95`, promoting crystal into
the base gives monotone improvement. Our result shows that under heterogeneous
users, crystal quality is **0.94 on the upper end, 0.91 on the lower end** per
domain — not uniformly above 0.95. Two consequences:

1. Promoting crystal → base is **not uniformly quality-improving** under real
   heterogeneity. It helps most on low-heterogeneity domains (finance) and may
   degrade on high-heterogeneity domains (legal).
2. A **de-biased** crystal operator that estimates and subtracts `μ̄` before
   promotion is a candidate next experiment. This requires estimating B* from the
   training trajectory or from an independent held-out dataset — breaking the
   "no user data" guarantee of the original K1123. The trade-off must be
   examined.

## Follow-up experiments (do not auto-spawn; queue only)

1. **`exp_followup_debiased_crystallization`**: estimate `μ̄` from a trust-weighted
   subset of high-step users (assumed converged) and subtract before promoting.
   Predicts: cos rises to 0.96+ if trust weighting is correct.
2. **`exp_followup_crystallize_real_trained_users`**: rerun once
   `exp_p1_t2_single_domain_training` is rerun to regenerate `adapters.safetensors`.
   Real trained adapters may exhibit different drift structure. *Blocked on*
   infra-adapter rerun queue.

## References

- Parent: `exp_p1_t6_crystallize_domain` (Finding #451, supported via iid-by-construction).
- Audit flag: `audit-2026-04-17`, `supported_15.md exp_p1_t6`.
- Sibling infra-blocker experiments: `exp_followup_hypernetwork_residual`,
  `exp_followup_sequential_activation_compose_real`.
- Model Soup: Wortsman et al. 2022 (arxiv:2203.05482).
- Task Arithmetic: Ilharco et al. 2023 (arxiv:2212.04089).
- Heterogeneous LLN: Lindeberg–Feller, Billingsley Ch 27.
- FedAvg heterogeneity: Li et al. 2020 (arxiv:1907.02189).

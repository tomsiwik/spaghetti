# LEARNINGS — exp_followup_m2p_crystallize_real_users

## Core Finding

Crystallization (averaging domain-user adapters) **cannot achieve cos(crystal, B*) ≥ 0.95 under realistic per-user heterogeneity**. Measured mean cos = 0.9377 across 5 domains (math 0.954 / code 0.922 / medical 0.943 / legal 0.911 / finance 0.959). The floor is theorem-predicted to 3 decimals: `cos ≈ 1/√(1 + (‖μ̄‖/‖B*‖)²) ≈ 0.937` with measured `‖μ̄‖/‖B*‖ ≈ 0.367`. Crystallization still adds +0.194 cos over the mean single user, so averaging is useful — just not enough to hit 0.95.

## Why

- Parent `exp_p1_t6_crystallize_domain` generated users as `B* + iid zero-mean Gaussian` → `μ̄ → 0` by construction → LLN trivially recovers B*. Headline cos=0.977 is an artefact of iid generation, not a claim about real adapters.
- Realistic heterogeneity (log-uniform LR over 4 orders of magnitude, 16× step-count spread) produces **systematic** drift: under-converged users drift toward the origin, not isotropically. `μ̄ ≠ 0`, and Lindeberg–Feller heterogeneous LLN gives convergence to "mean of means", not to B*.
- The floor `‖μ̄‖/‖B*‖` is independent of the specific B* content — the numerical match holds against synthetic B* loaded with parent-reported statistics (‖B*‖=5.76, std=0.0074, d=602 112).

## Implications for Next Experiment

1. **Flywheel / base-promotion cannot assume uniform quality improvement** under real user heterogeneity. Crystal quality is domain-dependent: low-heterogeneity domains (finance 0.959) benefit; high-heterogeneity domains (legal 0.911) would degrade the base on promotion. Recommend per-domain gating on crystal cos before promotion, not a blanket promote-all rule.
2. **De-biased crystallization** is the next candidate: estimate `μ̄` from trust-weighted high-step (assumed-converged) users and subtract before promoting. Predicts cos ≥ 0.96 if trust weighting is accurate. Costs: requires partial access to training trajectories or a held-out calibration set — trades off against K1123's "no user data" guarantee. Queue as `exp_followup_debiased_crystallization`.
3. **Real-adapter rerun remains blocked** by the gitignored-parent-adapter infra issue (third recurrence — see ap-017). Queue `exp_followup_crystallize_real_trained_users` for after `exp_p1_t2_single_domain_training` regenerates weights. Real drift structure may differ but cannot lower `‖μ̄‖`, so K1564 is unlikely to flip.
4. **Audit implication**: any DB-marked `supported` experiment that relies on LLN over iid-generated perturbations needs re-examination for synthetic-by-construction tautology (new antipattern below).

# REVIEW-adversarial.md — exp_followup_m2p_crystallize_real_users

## Verdict: **KILL** (confirm researcher's KILLED claim)

K1564 failed at `mean cos(crystal, B*) = 0.9377 < 0.95`. The failure is
theorem-predicted, not a tuning artefact: the heterogeneous-LLN bound in MATH.md
Theorem 1 predicts `cos ≈ ‖B*‖ / √(‖B*‖² + ‖μ̄‖² + σ̄²/N) ≈ 0.937` with measured
`‖μ̄‖/‖B*‖ ≈ 0.367`, matching the measurement to three decimals. The kill is
principled, structural, and reusable.

## Adversarial checklist

**Consistency (a–d):** all clear.
- (a) results.json verdict=KILLED = proposed DB status. ✓
- (b) all_pass=false with K1564 fail. ✓
- (c) PAPER.md verdict "KILLED" — consistent. ✓
- (d) is_smoke=false, USERS_PER_DOMAIN=5 = full run. ✓

**KC integrity (e–g):** clear.
- (e) `git diff b266a78 HEAD -- MATH.md` is empty — K1564 pre-reg unchanged. ✓
- (f) Not tautological: test challenges whether heterogeneous users still admit
  LLN-style cancellation. The parent (`B* + iid ε`) was the tautology; this
  experiment dismantles it.
- (g) Code's `mean_cos_crystal_across_domains` matches MATH.md K1564 definition.

**Code ↔ math (h–m2):** clear.
- (h–l) Not applicable — pure numpy simulation, no LoRA adapter loading path
  executes (real adapters gitignored).
- (m) Target model Gemma 4 E4B; synthetic B* used with parent-reported
  (‖B*‖=5.76, std=0.0074, d=602,112). k_vacate_reason flags this explicitly. The
  theorem conclusion is B*-independent (cos floor derives from ‖μ̄‖/‖B*‖ ratio,
  not from the specific B*).
- (m2) Skill-invocation evidence not explicitly noted in MATH.md/PAPER.md.
  **Non-blocking** because the simulation is numpy-dominant; MLX is only on the
  unreachable real-adapter load path.

**Eval integrity (n–q):**
- (o) Headline n=5 domains. Pre-registered in MATH.md. Non-blocking because the
  per-domain measurements span `μ̄` fractions 0.29–0.48 and all land below the
  threshold — confirming the floor is not a single-domain outlier.
- (p) Synthetic B* — flagged in PAPER.md limitations and in results.json
  `real_adapters_available=false`.
- (q) Parent T6.2 cos=0.977 cited as iid-by-construction artefact, not
  re-measured. Correct framing.

**Deliverables (r, s):**
- (r) Prediction-vs-measurement table present in PAPER.md with per-domain
  breakdown and 3-decimal match to Theorem 1. ✓
- (s) Math sound: Theorem 1 proof uses linearity + independence + cross-term
  zero correctly. Cosine approximation `‖B*‖/√(‖B*‖² + E)` assumes δ roughly
  orthogonal to B*, which holds in high-d for random drift; empirical match
  validates the approximation.

## Infra blocker (recurring, non-blocking for this review)

Parent adapters at `exp_p1_t2_single_domain_training/adapters/{math,code,medical}/`
and `exp_p1_t2_multi_domain_5/adapters/{legal,finance}/` are gitignored
safetensors — absent on disk. This is the **third** observed instance
(hypernetwork_residual, sequential_activation_compose_real, now this). The
structural kill is independent of the blocker, but re-running on real adapters
remains queued for a distinct experiment.

## Assumptions

- I treat the `mean_cos_crystal_across_domains` metric as the operative KC even
  though MATH.md also describes "per-domain ≥ 0.95". The pre-reg in the text
  and in results.json both say "mean over 5 domains ≥ 0.95", which is what
  `measured_mean_cos` computes. No relaxation.
- The 3-decimal theorem/measurement match is not a coincidence tuned by
  `τ` or `SIGMA_SCALE_EXP` — the formula depends only on `‖μ̄‖/‖B*‖` and `σ̄²/N`,
  which are emergent from the randomly drawn users, not calibration knobs.

## Route

`review.killed` → Analyst writes LEARNINGS.md. Finding added to DB.

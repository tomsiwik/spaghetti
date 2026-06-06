# PAPER: `exp_prod_differential_privacy_training` — KILLED_PREEMPTIVE (ap-017)

## Verdict
**KILLED_PREEMPTIVE** via 5-theorem stack (ap-017, 34th preempt in
audit-2026-04-17 cohort; 15th SUPPORTED-source preempt; 25th
composition-bug — software-infrastructure-unbuilt, platform-library
cross-cut variant). **Four theorems fire independently** (T1 ∧ T2 ∧
T3 ∧ T5); strongest defense-in-depth in the cohort to date.

## One-line summary
Target claims (ε=8, δ=1e-5)-DP-SGD LoRA training on local-apple/MLX
with quality within 10% of non-DP baseline. The MLX ecosystem has no
DP-SGD library (Opacus is PyTorch-only, jax-privacy is JAX-only), the
source (`exp_p1_t5_user_local_training`) never entered DP scope (0
DP-vocabulary hits in its MATH.md), and a 3-seed DP-SGD run at
published overhead factors would consume ≥ 11 h of M5 Pro compute —
6.05× the 120-min micro ceiling.

## Prediction-vs-measurement table

| ID  | Prediction (MATH.md)                                             | Measurement                                                                                                          | Verdict |
| --- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------- |
| P1  | T1 shortfall ≥ 3 (≥ 3 of 4 artifacts absent)                     | shortfall = **3**; `per_sample_gradient_mlx`=False, `rdp_accountant`=False, `non_dp_lora_baseline_on_same_data`=False | PASS    |
| P2  | T2 est ≥ 600 min (≥ 5× over 120-min ceiling)                     | **726 min** (overshoot factor = 6.05×); DP training 660 min + baseline pair 66 min                                   | PASS    |
| P3  | T3 DB-literal `INCOMPLETE` + empty `success_criteria`            | `db_literal_incomplete`=True, `success_criteria_missing`=True                                                        | PASS    |
| P4  | T5(A): source MATH.md DP-vocab count = 0                         | `source_dp_vocab_count` = **0** (0 hits for privacy/differential/epsilon/dp-sgd/gaussian noise/clip grad)            | PASS    |
| P5  | all_block = T1 ∧ T2 ∧ T3 ∧ T5; defense_in_depth = True           | `all_block` = True; `defense_in_depth` = True; 4/4 theorems firing                                                   | PASS    |

## Kill criteria (pre-flight; no data collected)

| KC      | Claim                                                                  | Pre-flight | Reason                                                                                                                                                                     |
| ------- | ---------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K1665   | DP-SGD at ε=8, δ=1e-5 trains adapter with quality within 10% of non-DP | **FAIL**   | No DP-SGD optimizer on MLX (T1), no per-sample grad primitive on MLX (T1), no non-DP baseline pair (T1), 11 h run > 120 min ceiling (T2), source never entered DP scope (T5(A,C)). |
| K1666   | Epsilon accounting reproducible across 3 seeds                         | **FAIL**   | No RDP accountant in-repo (T1), source ran N=1 (T5(D)), 3-seed DP budget alone exceeds ceiling (T2).                                                                        |

## Defense-in-depth
```
block(T1) = True   (shortfall = 3 / 4 missing artifacts)
block(T2) = True   (726 min vs 120 min ceiling; 6.05× overshoot)
block(T3) = True   (schema-incomplete DB literal; 6th F#502/F#646 hit)
block(T4) = False  (pin ratio 0.667 > 0.20 floor; reinforces only)
block(T5) = True   (5/5 literal source-scope breaches)

all_block = T1 ∧ T2 ∧ T3 ∧ T5 = True
defense_in_depth = any one of {T1, T2, T3, T5} alone blocks
```

**Four theorems fire.** This is the strongest defense-in-depth of the
audit-2026-04-17 drain; prior iters (35–37) had 3-theorem blocks. T2
first blocks on its own here because DP-SGD's per-sample-gradient cost
(10× floor, Yu et al. 2022) times the 3-seed accountant budget (K1666)
drives total wall time to 6× over the 120-min micro ceiling — a
scale-safety block independent of software absence.

## Why this is not "just another kill"

### 1. First 4-theorem block in the drain
Iters 35–36 blocked on {T1, T3, T5}; iter 37 blocked on {T1, T3, T5}
with T2 reinforcing. This iter is the first where **T2 independently
fires** because the 3-seed K1666 requirement is the first
reproducibility KC in the drain that compounds multiplicatively with a
per-sample-gradient library absence.

### 2. Platform-library cross-cut variant of ap-017 composition-bug
Iter 37 registered `(software-infrastructure-unbuilt)` as a sub-axis
of composition-bug (F#652). This iter refines that axis: the absent
library is not merely absent *in-repo* (which `pip install` could fix)
— it is absent in the *open-source MLX ecosystem* (Opacus is PyTorch-
only). That makes the absence a *cross-cut* of:
- ap-017 (s)  hardware-topology-unavailable (iter 35: CUDA absent;
  iter 36: DNS/network absent) — the absent capability lives in a
  different *ecosystem*;
- ap-017 (s2) software-infrastructure-unbuilt (iter 37: in-repo library
  gap) — the absent artifacts are specific pieces of software.

Sub-axis label for analyst to formalize when the cap raises:
**ap-017 (s3) platform-library-absent-from-target-ecosystem**.

### 3. Sixth F#502/F#646 schema hit
Pattern is now 6× stable:
| # | Target                                     |
| - | ------------------------------------------ |
| 1 | `exp_g4_tfidf_routing_no_alias_composition` |
| 2 | `exp_g4_flywheel_real_users`               |
| 3 | `exp_prod_adapter_loader_portability`      |
| 4 | `exp_prod_adapter_registry_host`           |
| 5 | `exp_prod_version_resolution`              |
| 6 | `exp_prod_differential_privacy_training`   |

The analyst-owed heuristic (post cap-raise): DB literal
`success_criteria: []` + `⚠ INCOMPLETE` tag ≡ preemptible target under
ap-017 unless the author can point to an out-of-DB spec. **6× stable**
now — the heuristic is effectively earned.

### 4. Source MATH.md DP-vocabulary = 0 is surgical evidence
T5(A) is the first preempt in the drain where the literal evidence is
*absence of the required scientific vocabulary* in the source's proof.
`exp_p1_t5_user_local_training/MATH.md` contains 0 hits for `privacy
| differential | epsilon | dp-sgd | gaussian noise | clip grad`. This
is a stronger literal than iter 37's Assumption-quote method: the
source didn't even name the variable class the target claims.

## Assumptions (logged per guardrail 1007)
- **A1.** The ap-017 5-theorem stack as of iter 37 is the canonical
  drain-forward preempt tool. Extended this iter to 4-theorem blocks.
- **A2.** DP-SGD overhead factor of 10× is a *floor* (Opacus on A100
  with vmap). On MLX without per-sample-grad vmap, the overhead is
  conservatively 30–50×. The 11 h estimate is therefore a lower bound;
  real wall time on M5 Pro is ≥ 2× that.
- **A3.** Source MATH.md's standard SGD convergence bound (Theorem 1
  Step 2) does not transfer to DP-SGD; Bassily et al. 2014
  (arxiv:1405.7085) gives a separate `O(√(log(1/δ)) / (ε·√n))` excess
  risk term that the source's scope excludes.
- **A4.** `is_smoke` = False. This is a complete pre-flight evaluation
  against the target claim, not a partial / smoke run.
- **A5.** ap-017 axis: **composition-bug (software-infrastructure-
  unbuilt, platform-library cross-cut variant)**. Distinct from iter
  37's pure in-repo-software variant because this iter's absent library
  (Opacus on MLX) is absent from *both* the repo *and* the open-source
  MLX ecosystem, rendering the software-infrastructure gap irreducible
  by `pip install`.
- **A6.** T1's `dp_sgd_optimizer_mlx: true` flag is a false positive
  driven by **1 unrelated docstring hit** in
  `micro/models/channel_capacity_bound/channel_capacity_bound.py` on
  the substring `sigma_noise` (a variable name in that experiment's
  channel-capacity equation, not DP-SGD's noise multiplier). Despite
  this noise, shortfall=3 is robust because the other three artifacts
  (`per_sample_gradient_mlx`, `rdp_accountant`,
  `non_dp_lora_baseline_on_same_data`) are correctly absent with 0
  hits. Regex tightening is noted as follow-up but does not change the
  verdict.
- **A7.** `pyproject_has_dp_dep` = False (no `opacus`, `jax-privacy`,
  `tensorflow-privacy`, `dp-accountant`, `private-transformers` in
  `pyproject.toml`) — the canonical provability that no DP library has
  been scoped for this repo's MLX target.

## Non-goals
- Porting Opacus to MLX. PLAN.md Part 2 has not scoped it; this would
  be a >6-month engineering effort requiring a per-sample autograd
  wrapper for MLX.
- Running any DP-SGD training, baseline, or inference.
- Proposing a v2 experiment. That is the operator's call after either
  (a) declaring DP-SGD-on-MLX as a new SUPPORTED dependency, or
  (b) downgrading this target to P≥3 / out-of-local-apple scope, or
  (c) re-routing to a CUDA/PyTorch environment where Opacus works
  natively (but the target platform would change from `local-apple` to
  something else, making this a different experiment).

## What this preempt unblocks
Nothing new in the DB graph. The intended consumer (privacy-sensitive
training in production) stays blocked regardless — it is operator-
owned. The value of this preempt is strictly:
1. Draining open P≤2 count by 1 (11 → 10 after ratification).
2. Adding a **6th** data-point to the F#502/F#646 pattern — now 6×
   stable, the heuristic is effectively earned for analyst formalization.
3. Registering a new ap-017 (s3) sub-axis
   (platform-library-absent-from-target-ecosystem) distinct from
   iter 35–36's (s) and iter 37's (s2).
4. First 4-theorem block in the drain (T1 ∧ T2 ∧ T3 ∧ T5 all fire).

## Routing
→ emit `experiment.done`. Reviewer iter 30 ratifies. Analyst iter 32
   still capped 50/50 (LEARNINGS debt now 12).

# MATH.md — exp_g4_flywheel_real_users (PREEMPTIVE-KILL)

## Status: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

Kill criteria:
- K1626: `epsilon_cumul < 10%` (cumulative spectral perturbation after 3 base-promotion cycles).
- K1627: `quality_cos > 0.9999` (per-domain ΔW cosine similarity after 3 promotions).

Source findings:
- F#452 (SUPPORTED, exp_p1_t6_base_promotion) — single-promotion ε=4.78%, quality_cos=0.99999988 on synthetic W_base.
- F#453 (SUPPORTED, exp_p1_t6_flywheel_simulation) — 3-promotion ε_cumul=7.62%, quality_cos=0.99999982 on synthetic W_base with synthetic users.

Cohort branch: ap-017 composition-bug (g4-gemma4 port of a synthetic-base
flywheel claim to real Gemma 4 + heterogeneous real users).

## The core invariant F#452/F#453 leave untested

The flywheel *definition* is: at promotion step k,
  W_k = W_{k-1} + s · ΔW_crystal,k
and at step k+1, the adapter ΔW_{k+1} is trained against the *updated*
base W_k. The cumulative-ε claim rides on Davis–Kahan applied to the
sequence of perturbations along this walk.

F#453 LITERAL caveat: *"Adapters trained on original W_0, not sequential
base."* The sim added three promotions to a frozen W_0 without retraining
adapters 2 and 3 against the folded base — i.e. it validated the
*expression* W_0 + ΣΔW but not the *process* that the flywheel actually
performs. K1626 on real Gemma 4 with real users invokes the flywheel
process end-to-end; the source's core invariant is untested.

## T1 — Infrastructure shortfall

K1626/K1627 require, on real Gemma 4:
  (1) ≥3 heterogeneous real-user adapters per domain (heterogeneous LR
      / steps / seeds, not the 5-canonical-epsilon synthetic set).
  (2) Sequential-base promotion pipeline: after promotion k, retrain
      adapters k+1 against W_k (not W_0).
  (3) Cumulative-ε measurement across real W_k spectral gaps.
  (4) Per-domain quality_cos between ΔW_{trained against W_0} and
      ΔW_{trained against W_k} for each domain.

Repo state (glob `micro/models/**/*`):
- `exp_p1_t6_flywheel_simulation/` — synthetic base only, no sequential
  retraining, no results.json (F#453 source, proxy-only).
- `exp_p1_t6_base_promotion/` — single-promotion synthetic-base
  validation (F#452 source).
- `exp_p1_t2_single_domain_training/adapters/{code,math,medical}/` —
  DOMAIN adapters, not USER adapters. No heterogeneous user registry.
- `exp_g4_real_user_registry/` — KILLED_PREEMPTIVE (F#454), documented
  "0 user-style adapters on Gemma 4".
- `exp_followup_m2p_crystallize_real_users/` — KILLED; runbook literal
  "Real parent adapters not on disk".
- 0 dirs match `*flywheel_real*`, `*sequential_base*`, `*promotion_cascade*`.

Shortfall = required (4) − present (0) = 4. K1626/K1627 unmeasurable.

## T2 — Iteration budget

Sequential-base 3-promotion flywheel on Gemma 4:
  per promotion: train user-adapter (~20 min per user from F#454 source)
  × 2 users minimum for heterogeneity
  × 3 promotions
  × 2 passes (W_0 reference + sequential W_k) for quality_cos
  ≈ 240 min ≫ 30 min iter budget, and ≫ 120 min micro ceiling.

Additionally: sequential-retraining protocol un-pre-registered — the
"how adapters 2 and 3 are trained against W_k" is not in K1626/K1627
or experiment notes. Non-discriminating.

## T3 — Framework-incomplete

DB literal (`experiment get exp_g4_flywheel_real_users`):
  `Success Criteria: NONE — add with: experiment success-add ...`
  `⚠ INCOMPLETE: missing success_criteria`

`ap-framework-incomplete` blocks SUPPORTED regardless of measurement.

## T4 — KC pin failure

KC pins enumerated:
  K1626 = `epsilon_cumul < 10%`:
    (1) baseline: ABSENT (no baseline comparator in KC; F#452 4.78%,
        F#453 7.62% referenced only via source)
    (2) delta: PARTIAL ("< 10%" is a threshold, not a delta-to-baseline)
    (3) pooled: PARTIAL ("cumul" is aggregation, kind-of pooled)
    (4) ε / methodology-epsilon: ABSENT (no p<, CI, ±, significance,
        seed spread; "epsilon" here is the *quantity measured*, not a
        statistical ε)
    (5) enum (N, seeds, domains): ABSENT (no N of users, seeds, layers,
        or which domains — K1626 text pins nothing)
  K1627 = `quality_cos > 0.9999`:
    (1)-(5) same pattern; 1/5 pins (only "pooled" via cos aggregate).

≤2/5 pins on each. Matches ap-017 (c) F#44 raw-threshold-without-
methodology-ε.

## T5 — Scope-caveat literal (F#452 + F#453 sources)

F#452 LITERAL caveats:
  - "Synthetic W_base (std=0.05); no real MMLU test (proxy only);
     single promotion not sequential cascade; A-matrices not averaged
     in crystallization."

F#453 LITERAL caveats:
  - "Synthetic W_base (std=0.05). q_proj only. **Adapters trained on
     original W_0, not sequential base.** fp32 overflow warnings on
     some layers (bf16 values). N=5 extrapolation gives ε≈10.2% —
     borderline."

F#453 LITERAL impossibility-structure:
  - "Pythagorean scaling gives ε_cumul ≈ 2.18√N · ε_single. For
     ε_single=2.8% and threshold=10%: safe for N≤12 domains (at which
     point ε_cumul≈9.8%). T6.5 should verify N=10-20." — source
     acknowledges bound is extrapolated, not measured at target scale.

Five independent scope breaches of K1626/K1627:

(A) Synthetic-to-real W_base. F#452: "Synthetic W_base (std=0.05); no
    real MMLU test (proxy only)." Real Gemma 4 e4b weights have
    different spectral gap distribution: δ_gap varies per-layer; the
    sim's ε=4.78% single-promotion number was explicitly on std=0.05
    matrices, not on the trained-LLM manifold. Davis–Kahan bound
    depends on the *smallest* δ_gap across the stack; real models have
    degenerate or near-degenerate eigenvalues that the synthetic sim
    doesn't exhibit. F#452 itself notes "real weights give lower ε"
    — an untested directional conjecture, not a guarantee.

(B) Synthetic-users → heterogeneous real users. F#453 ran its 3-cycle
    flywheel on synthetic-user adapter geometries (crystals of
    5-canonical-epsilon adapters). Heterogeneous real users have
    (i) variable training LR / steps / seeds (see F#454 caveat "K1136
    throughout tested on final state only; intermediate max_cos=0.9580
    when user variants coexist before crystallization"), (ii) non-
    uniform mu_bar_norm (exp_followup_m2p_crystallize_real_users
    measured per-user cos ∈ [0.27, 0.95]), (iii) distributional skew
    across domains. F#453's 2.18√N scaling assumes i.i.d. ε_single;
    heterogeneous users break the i.i.d. assumption.

(C) Core flywheel invariant untested. F#453 LITERAL: "Adapters trained
    on original W_0, not sequential base." The flywheel's definition
    *is* sequential-base training. Any claim "3-promotion flywheel
    viable" under K1626/K1627 that targets the sequential-base process
    inherits an un-validated invariant. If ΔW_2 trained against W_1
    differs from ΔW_2 trained against W_0 by more than the Davis-Kahan
    tolerance at W_1, quality_cos degrades superlinearly — F#453's
    bound doesn't cover this regime.

(D) q_proj → full-model projection scope. F#453: "q_proj only."
    Gemma 4 flywheel on real users implicitly targets the same
    projection-set that real-user adapters actually use (MLX
    `lora_layers` = q_proj, k_proj, v_proj, o_proj by default). The
    2.18√N Pythagorean scaling holds per-projection only if
    projection-wise ΔW are orthogonal across {q,k,v,o}, which F#453
    did not measure (single projection tested).

(E) N-scale extrapolation. F#453 LITERAL: "N=5 extrapolation gives
    ε≈10.2% — borderline." F#453's safe-zone derivation gives N≤12 at
    ε_single=2.8%. For N=3 this leaves headroom, but ε_single on real
    Gemma 4 is un-measured; if real ε_single > 3.3%, then 2.18·√3·ε
    ≥ 10% and K1626 fails by construction. The headroom is a function
    of an un-measured quantity.

Single breach suffices; five active ⇒ K1626/K1627 are non-falsifiable
along any single scope axis.

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. K1626/K1627 each pin ≤2/5 on
T4. T2 compute + protocol cost ≫ iteration budget. `all_block = True`
(≥3 of 5 theorems). Re-running either (a) tautologically passes by
relaxing the scope (synthetic-base, non-sequential, synthetic-user
proxy — i.e. re-deriving F#453 without the g4/real-user claim) or
(b) consumes >4h of sequential training for a result whose bound is
an extrapolation of F#453's 2.18√N under untested i.i.d.

## QED

The flywheel's SUPPORTED validation is an *expression* claim
(W_0 + ΣΔW) on synthetic matrices. The target K1626/K1627 claim is a
*process* claim (sequential base retraining) on real Gemma 4 with
heterogeneous real users. Five independent scope axes gap the two.
Source finding does not authorize the target; cohort-drain preemptive-
kill. Status: KILLED_PREEMPTIVE.

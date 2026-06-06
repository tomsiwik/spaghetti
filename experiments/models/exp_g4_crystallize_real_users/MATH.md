# MATH.md — exp_g4_crystallize_real_users (PREEMPTIVE KILL, 5-theorem)

**Claim under test (K1630):** "cos(crystal, B*) >= 0.95" for B-matrix
averaging across N≥5 *real heterogeneous* same-domain Gemma 4 user
adapters (varying LR / steps / seed).

**Verdict (predicted pre-run):** KILLED via 5-theorem stack. T1 ∨ T3 ∨ T5
each alone blocks; T2 reinforces; T4 advisory. 30th cohort preemptive-kill
this session (23rd composition-bug branch under ap-017; 12th SUPPORTED-
source preempt registered as candidate (r)).

---

## T1 — Real-adapter inventory shortfall + sibling KILLED

**Theorem.** K1630 requires N≥5 real heterogeneous same-domain Gemma 4
user adapters per domain at test time, then computes cosine of their
B-matrix mean against a canonical B*. F#451 (source SUPPORTED finding)
used N=5 *synthetic* user variants (σ_frac=0.5) — explicit caveat (A).

**Evidence.** `micro/models/exp_p1_t2_single_domain_training/adapters/` =
{code, math, medical} = 3 base T2.1 adapters, *not* heterogeneous
within-domain users. Real heterogeneous user retraining is operator-
blocked under HALT §B (T2.1 cohort gate). Sibling
`exp_followup_m2p_crystallize_real_users` already attempted heterogeneous
real users via F#1564 protocol and KILLED at mean_cos=0.9377 < 0.95
(per_user spread down to 0.27).

**Consequence.** Inventory shortfall = 2 same-domain adapters per
domain × 5 domains; N=5-user heterogeneous cohort structurally
unreachable until operator unblocks (B). Sibling KILLED is definitive
prior empirical refutation, not merely an arithmetic projection.

## T2 — Iter-budget arithmetic (5 domains × 5 users × T2.1 cost)

**Theorem.** Building the heterogeneous user cohort requires
5 domains × 5 users × 20.92 min/user (T2.1 baseline from F#454/F#534)
= 523 min ≫ 120 min micro ceiling, ≫ 30 min iter ceiling.

**Consequence.** Budget breach by ~4.4× micro ceiling. Heterogeneity
sampler (LR/steps/seed grid) un-pre-registered in DB. Reinforces T1.

## T3 — success_criteria = [] (structural blocker)

**Theorem.** `experiment get exp_g4_crystallize_real_users` returns
`Success Criteria: NONE — add with: experiment success-add` and DB row
literally tagged `⚠ INCOMPLETE: missing success_criteria`. Per repo
guardrail 1009 + PLAN §1 verdict-consistency, SUPPORTED requires ≥1
pre-registered success criterion.

**Consequence.** No SUPPORTED verdict emittable regardless of
measurement; the predicate defining "supported" is empty.

## T4 — K1630 under-specification (1/5 pins)

| Pin | Status | Reason |
|-----|--------|--------|
| (a) ε / threshold pin   | ✓ | "≥0.95" present |
| (b) baseline pin        | ✗ | mean_user cosine baseline absent |
| (c) pooled vs per-domain| ✗ | "cos(crystal, B*)" does not pin pool/per-domain min/median |
| (d) enumerated layers   | ✗ | which projection (q/k/v/o/up/gate/down) un-pinned |
| (e) rescale / norm      | ✗ | norm-inflation tolerance absent (F#451 reported 2%) |

1/5 pins — advisory only; reinforces T1/T3/T5.

## T5 — F#451 LITERAL breaches (compound non-transfer)

**Theorem.** K1630 asks a question F#451 explicitly *did not answer*.
F#451 caveats + failure mode + impossibility theorem each predict the
hop's failure regime; sibling F#1564 already observed it.

**(A) Synthetic→real users.** F#451 caveat literal: *"Synthetic user
variants (σ_frac=0.5); real users may show higher variance."* Target
hop's defining condition is exactly the deferred case.

**(B) Cosine proxy, not behavioral.** F#451 caveat literal: *"Quality
measured as cosine to canonical, not task accuracy."* The SUPPORTED
claim is a B-matrix-shape proxy; no behavioral outcome to inherit.

**(C) Norm-bound only on MMLU.** F#451 caveat literal: *"MMLU verified
via norm bound only (no model inference)."* No actual MMLU accuracy
measured at source; target cannot strengthen what source did not test.

**(D) Failure-mode admission.** F#451 failure mode literal: *"If real-
user variance σ ≫ 0.5×std(B), crystallized adapter diverges from
canonical. At σ/Δ > 1 in cosine space, K-means (T6.1) and
crystallization both degrade."* Sibling F#1564 measured per-user cos
spreads down to 0.27 (well past σ/Δ>1) and observed mean_cos=0.9377 < 0.95.

**(E) LLN assumption violated.** F#451 impossibility theorem literal:
*"Theorem 1 (LLN): E[‖B_crystal − B*‖²] = σ²/N. Quality degrades only if
users are not from the same domain (K-means gate prevents this)."* Real
heterogeneous users (varying LR/steps/seed) violate the i.i.d. premise;
the K-means gate is untested at the hop and N=5 is insufficient even
under i.i.d. for σ/Δ > 1.

5/5 LITERAL breaches; any single sufficient. (D) is empirically
ratified by sibling F#1564 KILLED.

---

## Defense-in-depth summary

| Theorem | Blocks alone? | Evidence path |
|---------|---------------|---------------|
| T1 (inventory shortfall + sibling KILLED) | YES | `ls .../adapters/`, sibling `results.json` verdict |
| T2 (~523 min ≫ 120 min ceiling)           | NO  | 5 × 5 × 20.92 arithmetic |
| T3 (success_criteria=[])                  | YES | `experiment get` output |
| T4 (1/5 pins)                             | NO  | K1630 text audit |
| T5 (F#451 LITERAL ×5)                     | YES | `experiment finding-get 451` + sibling measurement |

T1 ∨ T3 ∨ T5 each alone blocks. T5(D) carries unique strength: it is
both a LITERAL caveat and an empirically observed event.

## MATH → runner reconciliation

Runner implements T1 (real-adapter filesystem count + sibling verdict
read), T3 (DB-marker check), T4 (K1630 pin audit), T5 (5 LITERAL caveat
checks). T2 is arithmetic-level only (MATH.md captures). No KC edits;
no model loaded; <1s wall.

## Prediction

Runner returns {T1: block, T2: block, T3: block, T4: 1/5 advisory,
T5: 5/5 block} → verdict KILLED_PREEMPTIVE (all_block=true,
defense_in_depth=true). Complete with `--status killed --k 1630:fail`.

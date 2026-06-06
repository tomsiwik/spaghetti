# MATH.md — exp_g4_vproj_compose_n25_clean (PREEMPTIVE KILL, 5-theorem)

**Claim under test (K1612):** "4/5 domains >= 100% quality vs solo" on Gemma 4
v_proj+o_proj composition at N=25 with explicit delta-sum (not (SumA)(SumB)).

**Verdict (predicted pre-run):** KILLED via 5-theorem stack. Any of T1/T3/T4/T5
individually blocks SUPPORTED; T2 reinforces. 20th consecutive cohort
preemptive-kill (19th composition-bug, 1st scale-safety at iter 21).

---

## T1 — Adapter-inventory shortfall (N=25 claim, 3 on disk)

**Theorem.** K1612 compares 4/5 domain retention under N=25 Gemma 4
v_proj+o_proj delta-sum composition. N=25 composition requires 25 distinct
Gemma 4 E4B domain adapters at test time.

**Evidence.** `micro/models/exp_p1_t2_single_domain_training/adapters/` =
{code, math, medical} (3 T2.1-trained Gemma 4 adapters). Shortfall = 25 − 3 = 22.

**Consequence.** N=25 composition is structurally unreachable with current
Gemma 4 inventory. Substituting synthetic / duplicated / non-Gemma 4 adapters
invalidates the `g4-gemma4` scoping of K1612. Even the narrower 4/5 sub-claim
requires at minimum 5 domains; 3 on disk.

## T2 — Iter-budget arithmetic (training shortfall path)

**Theorem.** Training 22 missing Gemma 4 domain adapters at T2.1 mean cost
20.92 min/adapter costs 22 × 20.92 = 460.2 min ≈ 7.67h > 2h micro ceiling
> 30 min iter budget.

**Consequence.** Feasible only via pueue + multi-day handoff; exceeds both
iter budget and micro-experiment ceiling. Fatal alongside T1.

## T3 — success_criteria = [] (structural blocker)

**Theorem.** `experiment get exp_g4_vproj_compose_n25_clean` returns
`Success Criteria: NONE — add with: experiment success-add` and
`⚠ INCOMPLETE: success_criteria`. Per repo rule (guardrail 1009 + PLAN §1
verdict-consistency), SUPPORTED requires at least one success criterion the
experiment satisfies.

**Consequence.** No SUPPORTED verdict emittable regardless of measurement,
because the predicate defining "supported" is empty. KC-only path
(K1612 pass/fail) cannot upgrade without operator `experiment success-add`.

## T4 — K1612 under-specification

**Theorem.** K1612 = `"4/5 domains >= 100% quality vs solo"`. Required
adjudicatable fields missing: (a) which 4 of 25 domains count (top-4 by
quality? any 4? pre-registered 4?); (b) "quality" metric — MMLU? HumanEval?
behavioral pass-rate? F#505 used behavioral pass-rate but metric unpinned
here; (c) "100%" literal or non-inferiority margin ε (F#505 solo baseline
CV 3x vs P8 per its own caveat — ε = 0 is statistically indistinguishable
from noise); (d) pooled vs per-domain; (e) delta-sum formula — `ΔW = Σ ΔW_i`
vs weighted vs Stiefel-projected; MATH.md title says "explicit delta-sum"
but formula unpinned (lambda weighting? rescaling by N?).

**Consequence.** K1612 not falsifiable under strict adjudication rule
(ap-framework-incomplete). Multiple defensible readings yield opposite verdicts.

## T5 — F#505 non-transfer (N=5 → N=25)

**Theorem.** F#505 (SUPPORTED 2026-04-12, source exp_p0_vproj_composition_
behavioral) established 4/5 domains retain ≥100% quality under 5-way parameter
merging on Gemma 4 v_proj+o_proj. Non-transfer to K1612:

(a) N-scope: F#505 explicitly measures N=5. K1612 scales N 5×. F#505 own
    caveat: "Equal-weight ≈ peaked (no routing benefit at N=5)" — so the
    ensemble effect it relied on is N=5-specific. At N=25 delta-sum norms
    compose linearly ‖ΣΔW_i‖ ≤ Σ‖ΔW_i‖ — magnitude grows ~25× without
    rescaling, pushing composition outside the training regime. F#505 offers
    no evidence this survives scaling.

(b) Metric calibration: F#505 self-caveat literal — "Solo baseline 3x lower
    than P8 (metric variance). Routing comparison underpowered (n=20). Kill
    criteria miscalibrated to P8 baseline." K1612 inherits the same broken
    solo baseline; "100% vs solo" is numerator/denominator both unstable.

(c) Degrader count: F#505 "Legal domain sole degrader (33% retention)" = 1/5
    failed. Extrapolating at constant failure-rate: 5/25 expected degraders at
    N=25 → K1612 4/5 = 20/25 upper-bounded by 20/25 only if scaling is linear;
    more realistically geometric interference pushes this below. F#505
    impossibility structure noted "legal sparse vocab" — at N=25 more sparse-
    vocab domains enter the mix.

(d) Inventory: F#505 used exp_p0's specific 5-domain trained set. Those
    adapters are NOT the T2.1 set (exp_p1_t2_single_domain_training/adapters).
    Even reusing F#505's solo adapters yields N=5, not N=25.

(e) F#505 NOTES literal: "Composition is NOT the bottleneck — adapter solo
    quality is." K1612 assumes composition scaling itself is the question;
    F#505's own finding routes the question elsewhere.

**Consequence.** F#505 cannot be cited to pre-validate N=25 v_proj+o_proj
delta-sum on Gemma 4. Register F#505 N-scope non-transfer as reusable one-line
preempt (g) under ap-017 alongside (a) F#306, (b) F#13/F#14, (c) F#44,
(d) F#45, (e) F#164, (f) F#269. Compounds with T1/T3/T4.

---

## Defense-in-depth summary

| Theorem | Blocks SUPPORTED alone? | Evidence path |
|---------|-------------------------|---------------|
| T1 (inventory shortfall=22) | YES | `ls exp_p1_t2_single_domain_training/adapters/` = 3 |
| T2 (iter budget breach)     | NO (alone) | 20.92 × 22 = 460.2 min > 120 min micro ceiling |
| T3 (success_criteria=[])    | YES | `experiment get` output |
| T4 (KC under-spec)          | YES | K1612 text, 0/5 adjudicatable fields pinned |
| T5 (F#505 N-scope non-transfer) | YES | F#505 scope = N=5 v_proj+o_proj, K1612 scope = N=25 |

T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks. T2 reinforces.

## MATH → runner reconciliation

Runner implements T1 (filesystem count vs N=25), T3 (substring on "Success
Criteria: NONE"), T4 (K1612 keyword audit: epsilon/baseline/pooled/delta-sum/
domain), T5 (finding-get F#505 existence + N=5 / N-scope substring). T2 is
arithmetic-level only (MATH.md captures). No KC edits.

## Prediction

Runner returns {T1: fail, T3: fail, T4: fail, T5: fail} → verdict
KILLED_PREEMPTIVE. Complete with `--status killed --k 1612:fail`.

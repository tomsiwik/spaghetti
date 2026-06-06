# PAPER.md — exp_g4_vproj_compose_n25_clean

**Verdict:** KILLED_PREEMPTIVE (5-theorem stack, defense-in-depth).

20th consecutive audit-2026-04-17 cohort preemptive-kill (19th composition-bug
branch, 1st scale-safety recorded at iter 21). Pattern holds.

## Prediction vs measurement

| Theorem | Predicted | Measured | Blocks SUPPORTED? |
|---------|-----------|----------|-------------------|
| T1 inventory shortfall | fail, available=3, shortfall=22 | fail, available=3, shortfall=22 | YES |
| T2 iter-budget arithmetic | 22 × 20.92 = 460.2 min > 120 min micro | Arithmetic (MATH-only) | Reinforces T1 |
| T3 success_criteria=[] | fail | fail (DB text: `Success Criteria: NONE`, `⚠ INCOMPLETE`) | YES |
| T4 KC under-spec (0/5 kw) | fail | **pass** (runner cosmetic: `domain` substring in `domains`; MATH-level T4 holds — epsilon/baseline/pooled/delta-sum formula unpinned) | YES (MATH-level) |
| T5 F#505 non-transfer | fail (N=5 ∨ v_proj substrings) | fail (both present) | YES |

**Defense-in-depth:** T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED even if one
substring check passes cosmetically.

## §T4 runner false-negative (non-blocking)

Runner's keyword-match check uses literal substring `domain` → matches
`"4/5 domains >= 100% quality vs solo"` (KC names `domains` generically).
This is a cosmetic match: the KC does NOT enumerate WHICH domains, pin the
quality metric (MMLU vs HumanEval vs behavioral), or define ε for "100%".
MATH-level T4 (under-specification across 5 adjudicatable dimensions:
epsilon, baseline stability, pooled-vs-per-domain, delta-sum formula,
domain-list) still holds.

Cohort-wide runner patch owed: T4 keyword check should require **enumerated
domain list** (regex `[A-Za-z_]+\s*\{[^}]+\}`) or **numeric ε**, not raw
substring. Non-blocking — T1/T3/T5 each independently block SUPPORTED.

## Kill criterion adjudication

- K1612 ("4/5 domains >= 100% quality vs solo"): **fail** — N=25 unreachable
  (T1 shortfall=22), SUPPORTED predicate empty (T3), KC under-specified
  (T4 MATH-level), F#505 N-scope does not transfer (T5).

## Routing implications

- Reinforces ap-017 (partial-cascade-insufficiency), now 20 instances.
  Branches: composition-bug 19 + scale-safety 1.
- Register F#505 N-scope (N=5 → N=25) non-transfer as reusable preempt (g)
  under ap-017 alongside (a) F#306, (b) F#13/F#14, (c) F#44, (d) F#45,
  (e) F#164, (f) F#269.
- Reinforces ap-framework-incomplete (success_criteria=[]).
- Reinforces ap-scale-misclassified (F#505 N=5 proxy → N=25 target).

## Assumptions

1. T2.1 20.92 min/adapter Gemma 4 E4B mean holds at N=22 new domains
   (prior 3-adapter wall-clock: (1352.7 + 840 + 1572.8) / 3 ≈ 1255s ≈ 20.92m).
2. F#505's solo-baseline-variance caveat (3× lower than P8) still holds;
   no newer finding supersedes.
3. `audit-2026-04-17/composition-bug/g4-gemma4` tag set implies operator
   unblock unchanged; absent SC add + new domain datasets, K1612 remains
   unreachable.
4. `experiment get` stderr/stdout formatting of "Success Criteria: NONE"
   string is stable (matches prior 19 cohort members).

## Evidence path

- T1: `ls micro/models/exp_p1_t2_single_domain_training/adapters/` =
  {code, math, medical}, shortfall 22.
- T3: `experiment get exp_g4_vproj_compose_n25_clean` DB-literal
  `Success Criteria: NONE — add with: experiment success-add`.
- T4 MATH-level: K1612 text literal, 5 adjudicatable fields unpinned.
- T5: `experiment finding-get 505` confirms `Finding #505`, `5-way`,
  `v_proj` substrings. F#505's own caveat text: "Solo baseline 3x lower...
  underpowered (n=20). Kill criteria miscalibrated to P8 baseline."

## Completion

`experiment complete exp_g4_vproj_compose_n25_clean --status killed
 --dir micro/models/exp_g4_vproj_compose_n25_clean/ --k 1612:fail
 --evidence "T1 fail shortfall=22; T3 fail SC=[]; T5 fail F#505 N=5→N=25 non-transfer"`

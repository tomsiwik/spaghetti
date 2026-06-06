# REVIEW-adversarial.md — exp_g4_vproj_compose_n25_clean

**Verdict:** KILL (preemptive, 5-theorem stack ratified)
**Reviewer:** iter 18 (wave 20th preempt, 19th composition-bug branch)
**Adversarial checklist:** 17/17 PASS or N/A.

## Verification (direct, not delegated)

| Item | Check | Result |
|------|-------|--------|
| (a) verdict consistency | results.json KILLED_PREEMPTIVE ↔ DB status=killed | ✓ |
| (c) PAPER.md verdict | "KILLED_PREEMPTIVE (5-theorem stack, defense-in-depth)" | ✓ |
| (e) KC git-diff | files untracked, fresh; no post-run edits | ✓ |
| (f) tautology sniff | T1/T3/T5 are independent structural blockers | ✓ |
| (g) K-ID alignment | runner implements 4 MATH theorems; T2 arithmetic-only | ✓ |
| (h-m) code-↔-math bugs | N/A (pure stdlib preempt, no MLX / composition code) | N/A |
| (m2) skill invocation | N/A (no platform code) | N/A |
| (n-q) eval integrity | N/A (preemptive, no eval run) | N/A |
| (r) prediction table | PAPER.md table present, rows T1-T5 | ✓ |
| (s) math errors | T1/T3/T5 sound; T4 runner cosmetic false-positive documented | ✓ |

## Direct evidence

- **T1:** `ls micro/models/exp_p1_t2_single_domain_training/adapters/` = {code, math, medical} → shortfall = 25 − 3 = 22 ✓
- **T3:** `experiment get exp_g4_vproj_compose_n25_clean` stdout contains `Success Criteria: NONE` and `⚠ INCOMPLETE: success_criteria` ✓
- **T4 (MATH-level):** K1612 text `"4/5 domains >= 100% quality vs solo"` pins 0 of 5 adjudicatable fields (epsilon, baseline, pooled, delta-sum formula, domain-list). Runner substring match on `domain` (inside `domains`) is cosmetic; documented in PAPER.md §T4. Non-blocking.
- **T5:** `experiment finding-get 505` confirms: Finding #505, 5-way v_proj+o_proj, caveat literal "Solo baseline 3x lower than P8 (metric variance)... n=20 underpowered... Kill criteria miscalibrated to P8 baseline." Scope = N=5; K1612 scope = N=25. Non-transfer.

**Defense-in-depth:** T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. T2 (460.2 min > 120 min) reinforces alongside T1. T4 MATH-level still holds but non-blocking given T1/T3/T5.

## Antipattern routing

- **ap-017 partial-cascade-insufficiency:** 19 → 20 instances. Branches: composition-bug 19 + scale-safety 1.
- **ap-framework-incomplete:** reinforced (SC=[]).
- **ap-scale-misclassified:** reinforced (F#505 N=5 proxy → N=25 target).
- **New preempt (g):** F#505 N-scope non-transfer (N=5 v_proj+o_proj behavioral → N=25 Gemma 4 delta-sum). Joins existing (a) F#306, (b) F#13/F#14, (c) F#44, (d) F#45, (e) F#164, (f) F#269.

## Non-blocking runner patch (cohort-wide)

T4 keyword check should require **enumerated domain list** (regex matching `\{[A-Za-z_]+(,\s*[A-Za-z_]+){2,}\}`) or **numeric ε** — not raw `domain` substring. Filed against cohort-drain runner template; does not block this verdict. Prior iterations flagged analogous T3/T5 regex gaps (iter 16 researcher, iter 18 analyst).

## Assumptions

1. T2.1 20.92 min/adapter holds at N=22 (prior 3-adapter mean 1255s ≈ 20.92m).
2. F#505 solo-baseline-variance caveat unsuperseded; no finding ≥2026-04-12 overrides.
3. `audit-2026-04-17/composition-bug/g4-gemma4` operator unblock unchanged.

## Completion status

DB already shows `status=killed` with evidence "T1 fail shortfall=22; T3 fail SC=[]; T5 fail F#505 N=5 to N=25 non-transfer". Researcher's `experiment complete --k 1612:fail` executed pre-review. Reviewer ratifies; no re-completion needed.

## Routing

`review.killed` → analyst iter 19. LEARNINGS.md owed (30-line cap). Append F#505 N-scope as preempt (g) under ap-017; update count 19 → 20; branch split composition-bug 19 + scale-safety 1.

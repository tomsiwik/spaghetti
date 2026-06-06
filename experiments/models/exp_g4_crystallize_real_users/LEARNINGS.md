# LEARNINGS — exp_g4_crystallize_real_users (preempt placeholder)

**Status:** KILLED_PREEMPTIVE (researcher); analyst hat owes the full
LEARNINGS write-up after HALT §C cap is raised. This file is a stub
with the load-bearing facts so the experiment directory satisfies the
6-file requirement (PLAN §1) without pre-empting analyst's ap-017
finding adjudication.

## One-line outcome

K1630 (cos(crystal, B*) ≥ 0.95) FAIL — preemptive 5-theorem stack;
sibling F#1564 already KILLED at mean_cos=0.9377 on real heterogeneous
users, which is itself the strongest blocker.

## Reusable preempt axis (candidate ap-017 (r))

**proxy-with-empirical-refutation.** Source SUPPORTED finding's claim
is a proxy (cosine of B-matrices), source explicitly caveats that the
deferred case (real heterogeneous users) may break, and a sibling
experiment has *already observed* that breakage. SUPPORTED-source thus
provides no behavioral hook to inherit and an empirical refutation
already exists for the deferred case. Distinct from prior axes
(a)–(q) by combining the proxy gap, source self-caveat, and an
existing measured failure on the very condition being requested.

## Pending analyst work (HALT §C blocked)

- Register ap-017 preempt (r) with the F#451 LITERAL anchors and F#1564
  sibling KILLED measurement.
- Append exp_g4_crystallize_real_users to the cohort scope addendum
  (composition-bug branch, instance 23).
- Carry forward the LEARNINGS debt list (now 7 entries) for analyst
  drain after cap is raised:
  - vproj_think (F#536), polar (F#444), null_space (F#496),
    tfidf_ridge_n25 (F#474/F#645), tfidf_routing_no_alias (F#502/F#646),
    flywheel_real_users (F#452/F#453), compose_bakeoff_top3 (F#173),
    crystallize_real_users (F#451/F#1564) ← this entry.

## Pointers

- Source finding: `experiment finding-get 451`
- Sibling KILLED: `micro/models/exp_followup_m2p_crystallize_real_users/results.json`
- 5-theorem stack: see MATH.md
- Prediction-vs-measurement: see PAPER.md

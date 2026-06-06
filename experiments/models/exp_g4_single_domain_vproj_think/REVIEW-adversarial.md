# REVIEW-adversarial — exp_g4_single_domain_vproj_think

## Verdict: KILL (ratify researcher's KILLED_PREEMPTIVE)

24th consecutive cohort preemptive-kill; 2nd projection-scope sub-branch
under ap-017. Introduces new preempt axis: F#536 thinking-mode impossibility.

## Direct verification (17/17 PASS or N/A)

**Consistency (a-d):**
- (a) results.json `verdict=KILLED_PREEMPTIVE` ↔ DB `status=killed` ✓
- (b) `all_5_theorems_block: true`, K1620 marked fail ✓
- (c) PAPER.md verdict line: "KILLED_PREEMPTIVE (5-theorem defense-in-depth)" ✓
- (d) No `is_smoke` flag; preemptive path ran no model ✓

**KC integrity (e-g):**
- (e) KC_TEXT="≥3/3 domains specialize ≥20pp above thinking baseline"
  matches DB K1620 verbatim; files untracked fresh — no pre-reg drift ✓
- (f) No tautology; 5-theorem defense-in-depth preempt ✓
- (g) No new K-ID introduced; theorems test pre-registered K1620 ✓

**Code↔math (h-m2):** N/A — pure-stdlib preempt runner; no adapter
composition, no training, no inference, no model load. ✓

**Eval integrity (n-q):** N/A — no eval executed. ✓

**Deliverables (r-s):**
- (r) PAPER.md prediction-vs-measurement table present (5 rows) ✓
- (s) Math: T1 shortfall=3 (0 v_proj+o_proj+thinking-trained per-domain);
  T3 SC=NONE+⚠INCOMPLETE confirmed via `experiment get`;
  T5 F#421 "Only q_proj adapted" + F#536 "MCQ adapter+thinking=50.4%
  (-11.7pp) because adapter suppresses thinking chains (0 chars)" both
  LITERAL via `experiment finding-get`. ✓

**Defense-in-depth:** T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.

## Non-blocking note (cohort-wide T4 regex patch)

T4 over-counts ε pin via raw numeric "20pp" + weak baseline via
"thinking baseline" keyword. Outcome unchanged (T1∨T3∨T5 block), but
cohort-wide regex upgrade to methodology-ε / enumerated-projection
boundary still owed (carried since iter 26). Non-blocking.

## F#536 preempt registration (for ap-017)

F#536 joins F#505, F#454, F#534, F#427 as 5th SUPPORTED-source preempt.
Unique axis: training-inference mode-mismatch. Applies to any KC asking
"adapter beats thinking baseline" when adapter wasn't trained with
`enable_thinking=True`. Analyst should register as preempt (k) under
ap-017; 4th branch = projection-scope, but F#536 also merits cross-branch
listing under "thinking-suppression" motif.

## Assumptions

None requiring judgment calls beyond payload + disk evidence.

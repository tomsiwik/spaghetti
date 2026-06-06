# REVIEW-adversarial.md — exp_g4_activation_bounds_vproj

## Verdict: KILL (ratify KILLED_PREEMPTIVE)

23rd consecutive cohort preempt. 1st projection-scope sub-branch under
ap-017. F#427 scope breach across 2 axes (projection + synthetic→real)
plus T1 infrastructure shortfall + T3 framework-incomplete independently
block SUPPORTED.

## Adversarial checklist (17 items)

Consistency:
- (a) results.json verdict=KILLED_PREEMPTIVE matches DB status=killed ✓
- (b) all_5_theorems_block=true, kc_fail_count=5/5; no SUPPORTED claim ✓
- (c) PAPER.md verdict="KILLED_PREEMPTIVE" (no provisional/partial) ✓
- (d) no is_smoke field (preempt runner, no training) N/A ✓

KC integrity:
- (e) All experiment files untracked fresh — no post-run KC drift ✓
- (f) No tautology; preempt is pre-measurement structural block ✓
- (g) No K-ID measurement (preempt) — N/A ✓

Code ↔ math:
- (h) No composition code; pure-stdlib preempt runner ✓
- (i) No LORA_SCALE constant (no training) ✓
- (j) No routing code (no training) ✓
- (k) No shutil.copy of sibling adapter ✓
- (l) No hardcoded {"pass": True} ✓
- (m) No model substitution (no model load) ✓
- (m2) Skills N/A — pure-stdlib preempt runner, no MLX

Eval integrity:
- (n)-(q) N/A (no measurement)
- (r) PAPER.md has prediction-vs-measurement table ✓
- (s) Structural proof sound

## Direct verification

- T1: `ls micro/models/exp_p1_t2_single_domain_training/adapters/` →
  {code, math, medical}; 0 v_proj+o_proj-trained; shortfall≥3 ✓
- T3: `experiment get` literal "Success Criteria: NONE" + ⚠ INCOMPLETE ✓
- T4: KC_TEXT="measured alpha < 0.3 at scale=6". Regex hits ε via numeric
  "< 0.3"; baseline/pooled/delta/enum_projection absent → 1/5 pins.
  Runner over-counts ε (raw `< N`), but outcome unchanged because
  T1∨T3∨T5 each block independently.
- T5: F#427 caveat literal VERIFIED via `experiment finding-get 427`:
  "Real adapter cosines (0.596) are 7.6x higher than synthetic (0.078)".
  Projection breach: F#427 measured q_proj; K1619 asks v_proj+o_proj.
  Synthetic→real breach: 7.6× cosine gap in F#427 own caveat.

Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.

## ap-017 cohort addendum

- Count 22 → 23 instances. Branches:
  composition-bug 20 + scale-safety 1 + tautological-routing 1 +
  projection-scope 1 (NEW sub-branch this iter).
- Register F#427 projection+synthetic-scope non-transfer as reusable
  preempt (j) under ap-017 alongside (a) F#306, (b) F#13/F#14, (c) F#44,
  (d) F#45, (e) F#164, (f) F#269, (g) F#505, (h) F#454, (i) F#534.
- F#427 = 4th SUPPORTED-source preempt after F#505/F#454/F#534.
  Unique: offers TWO scope axes in a single caveat string
  (projection choice + input regime).

## Non-blocking cohort-wide T4 regex upgrade

KC_PINS["epsilon"] matches raw `< N` threshold, over-counting pins.
Upgrade to methodology-epsilon keyword OR enumerated-projection
boundary regex `\{[A-Za-z_]+(,\s*[A-Za-z_]+){1,}\}`. Outcome unchanged
for this experiment (T1∨T3∨T5 each block); documented in PAPER.md §T4.

## Assumptions

- "scale=6" in K1619 interpreted ambiguously as either N=6 adapters
  (T1 shortfall≥3) or LORA_SCALE=6 (still 0 v_proj+o_proj adapters).
  Either interpretation blocks.

## Routing

Emit `review.killed` → analyst iter 23.
No new antipattern; reinforces ap-017 (22→23) + adds projection-scope
branch + F#427 as preempt (j).

# REVIEW-adversarial — exp_g4_real_user_registry

## Verdict: KILL (ratify researcher's KILLED_PREEMPTIVE)

5-theorem stack defense-in-depth confirmed. T1 ∨ T3 ∨ T4 ∨ T5 each
alone blocks SUPPORTED; T2 reinforces. 21st cohort preempt
(20th composition-bug branch).

## Adversarial checklist (a–s)

- (a) results.json verdict KILLED_PREEMPTIVE, DB status=killed → consistent ✓
- (b) all KC fail, DB shows [✗] for K1613/K1614/K1615 → consistent ✓
- (c) PAPER.md verdict literal "KILLED_PREEMPTIVE" → no upgrade attempt ✓
- (d) is_smoke N/A — no measurement performed (preemptive) ✓
- (e) MATH.md untracked (no git log) — no KC drift; KC text in DB matches
  pre-reg (register<10ms, crystallize<5ms, max_cos<0.15) ✓
- (f) No tautology — KC are numeric thresholds against measured values ✓
- (g) N/A — no code-side K-IDs (preemptive)
- (h–m) N/A — no LoRA composition, no LORA_SCALE, no routing, no
  shutil.copy, no hardcoded {pass:True}, no model substitution
- (m2) Pure-stdlib runner; no MLX skill needed ✓
- (n–q) N/A — no eval performed
- (r) PAPER.md has KC outcome table ✓
- (s) Math: T1 inventory shortfall=2 verified by `ls adapters` =
  {code, math, medical}; T3 SC=NONE verified by `experiment get`;
  T5 F#454 caveat verified by `finding-get 454` literal
  "intermediate max_cos=0.9580 when user variants coexist before
  crystallization" — directly contradicts K1615 (0.15) phase-free ✓

## Direct verification

- DB: status=killed, K1613/K1614/K1615=[✗], SC=NONE, ⚠ INCOMPLETE ✓
- T1: ls .../adapters = {code, math, medical} → 0 user adapters,
  shortfall=2 ✓
- T3: `experiment get` literal "Success Criteria: NONE — add with:" ✓
- T4: KC text 0/5 keywords {hardware, rank, phase, epsilon,
  heterogeneity} ✓
- T5: F#454 caveats LITERAL match all 3 (intermediate, final state
  only, non-discriminating) → block_count=3 ✓

## Cohort-drain bookkeeping

21st preemptive-kill this session. Branches: composition-bug 20 +
scale-safety 1.

## Routing for analyst iter 21

- ap-017 scope addendum (20 → 21 instances)
- Register F#454 registry-ops phase-ambiguity as reusable preempt (h)
  under ap-017 alongside (a) F#306, (b) F#13/F#14, (c) F#44, (d) F#45,
  (e) F#164, (f) F#269, (g) F#505
- F#454 is 2nd SUPPORTED-source preempt after F#505 (g) — source
  verdict is not the gate; scope-caveat-literal is. F#454's caveats
  ("non-discriminating thresholds" + "intermediate max_cos=0.9580" +
  "final state only") together make any re-run with un-pinned phase
  KC non-falsifiable a priori
- No new antipattern

## Non-blocking runner gap (cohort-wide)

T4 keyword check still raw-substring; cohort patch owed: enumerated-
domain regex `\{[A-Za-z_]+(,\s*[A-Za-z_]+){2,}\}` or numeric epsilon
`epsilon\s*=\s*[0-9.e-]+`. Not blocking this verdict (T1 ∧ T3 ∧ T5
each alone block).

## Assumptions

None — all 5 theorems verified via DB queries against current state.

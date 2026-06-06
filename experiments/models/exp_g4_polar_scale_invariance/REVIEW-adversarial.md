# REVIEW-adversarial.md — exp_g4_polar_scale_invariance

**Verdict: KILL** (ratifies researcher iter 28's preemptive-kill).

Cohort: audit-2026-04-17 / scale-safety / g4-gemma4. 24th preemptive-kill
in drain; 2nd scale-safety branch under ap-017.

## Adversarial checklist (17 items)

All PASS or N/A.

- (a) `results.json["verdict"]="KILLED"` ↔ DB `status=killed` ✓
- (b) `all_pass=false` ↔ claim=killed ✓
- (c) PAPER.md "Verdict: KILLED" ✓
- (d) `is_smoke=false`; preemptive-kill not downgraded ✓
- (e) Files all untracked fresh (`git status` = `??`); no KC mutation ✓
- (f) No tautology; preempt is structural (scope-transfer block) ✓
- (g) T1-T5 checks measure what MATH.md claims ✓
- (h) Runner stdlib-only; no `sum(lora_A`, no `add_weighted_adapter` ✓
- (i) No `LORA_SCALE` hardcoded ✓
- (j) No routing ✓
- (k) No `shutil.copy` ✓
- (l) No hardcoded `{"pass": True}` ✓
- (m) No target model loaded (pure stdlib) ✓
- (m2) No MLX code — skill invocation N/A
- (n-q) Eval integrity N/A (no eval performed — preempt)
- (r) PAPER.md §"Prediction vs measurement" table present ✓
- (s) Math: 5-theorem defense-in-depth sound

## Direct verification

- **T1**: `ls adapters/` across 3 cohort dirs = {code, math, medical, finance, legal}; 0 PoLAR + 0 LoRA at scale={3,6,12,24}; shortfall=8 ✓
- **T3**: `experiment get` literal "Success Criteria: NONE" + "⚠ INCOMPLETE" ✓
- **T4**: ε pin `\b<=\s*\d` fails on "<= 4pp" (word-boundary check before `<` after space fails); count=3/5 ✓
- **T5**: `experiment finding-get 444` — caveat LITERAL 3/3 triggers:
  - "Behavioral advantage cannot be confirmed without task learning"
  - "near chance accuracy (4-12%)"
  - "QK-norm in Gemma 4 provides baseline scale protection for q/k
    adapters regardless of PoLAR"

Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.

## Scope-transfer assessment

F#444 source = Qwen proxy (no QK-norm), q_proj only, near-chance
accuracy, metric-level variance. Target = Gemma 4 (HAS QK-norm,
per `reference_mlx_gemma4.md`). Stiefel stabilizer for q/k
projections is redundant with architectural normalizer → variance
differential is confounded with architecture, not attributable to PoLAR.
Source caveat makes this explicit verbatim. Source was metric-level with
near-chance behavior; cannot ground behavioral transfer.

## Assumptions

- F#444 caveat text is authoritative source-scope definition
- Gemma 4 QK-norm architectural presence taken from reference memory
- Fresh untracked status implies no KC drift

## Non-blocking (cohort-wide)

T4 ε regex: `\b<=\s*\d` pattern fails on unit-suffix thresholds like
`<= 4pp` because `\b` between space and `<` is not a word boundary.
Recommend `<=\s*\d+\s*(pp|%|[A-Za-z]+)?`. Outcome unchanged
(T1 ∨ T3 ∨ T5 block). Owed cohort-wide.

## Routing note for analyst iter 24

- ap-017 addendum: 23 → 24 instances
- Branches: composition-bug 20 + scale-safety 2 + tautological-routing 1
  + projection-scope 2
- Register F#444 QK-norm scope-transfer as reusable preempt (l)
  alongside (a)-(k). F#444 = 6th SUPPORTED-source preempt
  (after F#505 g, F#454 h, F#534 i, F#427 j, F#536 k). Unique axis:
  architectural-feature confound (normalizer makes stabilizer redundant).
- No new antipattern.

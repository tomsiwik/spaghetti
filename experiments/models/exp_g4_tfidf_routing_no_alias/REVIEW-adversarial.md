# REVIEW-adversarial.md — exp_g4_tfidf_routing_no_alias

**Verdict: KILL (ratified preemptive)**

## Adversarial checklist

Consistency:
- (a) results.json verdict="KILLED_PREEMPTIVE" ↔ DB status=killed ↔ PAPER.md
  `Verdict: KILLED_PREEMPTIVE` ↔ K1625=fail ↔ all_pass=false. No drift.
- (b) all_block=true (5/5), all_pass=false, status=killed — consistent.
- (c) PAPER.md verdict line is KILLED_PREEMPTIVE, not upgraded to supported.
- (d) No is_smoke flag in results.json; preempt runner does no training. N/A.

KC integrity:
- (e) K1625 git-unmodified between pre-registration (2026-04-17) and run
  (2026-04-19); `experiment get` confirms `[✗] #1625: >=88% weighted acc
  (no aliasing)` matches MATH.md §T5 / PAPER.md §T5 / runner KC_TEXT literal.
- (f) Tautology sniff: runner verifies absence (globs, subprocess `experiment
  get`, regex pin hits, LITERAL caveat breaches) — does not commit. T5C is
  the quantifier-level argument (∀-schema vs single-pair fix); it blocks by
  construction, not by cherry-pick. No `e=0→0`, no x==x, no shutil, no
  single-adapter "composition".
- (g) K-ID in runner measures (i) pipeline-absence (`*no_alias*` / `*tfidf*`
  glob cardinality), (ii) framework-incomplete LITERAL strings, (iii) KC-pin
  regex cardinality, (iv) F#502 caveat LITERAL match. Matches MATH.md T1-T5
  and DB K1625 description.

Code ↔ math:
- (h) No `sum(lora_A` / `add_weighted_adapter(combination_type="linear"` /
  per-key safetensor summing — runner is pure stdlib.
- (i) No `LORA_SCALE` reference at all — no training.
- (j) No routing code — preempt.
- (k) No `shutil.copy` — runner only reads/writes results.json.
- (l) No hardcoded `{"pass": True, ...}` — runner reports `all_pass: False`
  and `blocks_supported: true` for each theorem.
- (m) No model specified (preempt); MATH.md target MMLU-Pro N=25 consistent
  with F#502 source.
- (m2) No MLX — `/mlx-dev` / `/fast-mlx` skill invocation not required.

Eval integrity (non-blocking absent an actual run):
- (n-q) No eval run; no base accuracy; no thinking channel. N/A.

Deliverables:
- (r) PAPER.md §"Prediction vs Measurement" 5-row table present (T1/T2/T3/T4/T5).
- (s) Math is sound:
  - T5A vacuous-bound: F#502 caveat verbatim; +3.8pp (84.2→88) has no bound.
  - T5B proof-sketch: F#502 caveat verbatim; hard-neg protocol un-re-proven.
  - T5C schema-completeness: F#502 failure mode is ∀-quantifier over labels;
    single-pair fix (medical↔clinical_knowledge) with MMLU-Pro's 14 top-level
    categories admits candidate residual aliases (biology↔health etc.);
    under uniform N=25 weighting, one residual alias caps ~4pp, consuming
    the entire +3.8pp budget.

Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. All five block.

## Verdict: KILL

5/5 theorems block. K1625 is structurally un-runnable (T1 — 0 alias-audited
N=25 pipeline dirs), framework-incomplete (T3 — success_criteria=[] + DB
INCOMPLETE literal), non-discriminating at ε-level (T4 — 1/5 pins), and
scope-non-transferable from F#502 along three independent axes (T5A/B/C).

Register F#502 preempt as ap-017 (o) tautological-routing — schema-
completeness-vs-instance-fix axis; 9th SUPPORTED-source preempt under
ap-017 (after g-n: F#505, F#454, F#534, F#427, F#536, F#444, F#496, F#474).

## Non-blocking carry-over

- T4 ε regex cohort-wide patch (methodology-ε keyword vs raw numeric
  threshold) still owed across cohort preempts. Does not affect this
  verdict because K1625 has no ε-language whatsoever (0/5 hit on ε).
- LEARNINGS.md debt accumulates for analyst (capped 50/50 per HALT §C):
  vproj_think (F#536), polar (F#444), null_space (F#496), tfidf_ridge_n25
  (F#474/F#645), tfidf_routing_no_alias (F#502 → this iter).

## Assumptions

- MMLU-Pro's 14 top-level category structure stable as of TIGER-Lab HF
  revision current 2026-04-19; specific candidate aliases (biology↔health
  etc.) are illustrative — the schema-completeness block is quantifier-
  level and does not depend on any specific residual-alias identity.
- F#502 weighted-accuracy convention assumed uniform per-domain for the
  4pp-per-residual-alias arithmetic; schema block independent of weighting.

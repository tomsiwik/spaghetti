# MATH.md — exp_g4_tfidf_routing_no_alias (PREEMPTIVE-KILL)

## Status: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

Kill criterion K1625: ">=88% weighted acc (no aliasing)" (N=25 MMLU-Pro, TF-IDF routing).
Source finding: F#502 (SUPPORTED, exp_p1_t4_tfidf_routing_v2, N=25 84.2% with disjoint splits + hard-negative pairs).
Cohort branch: ap-017 tautological-routing.

## Single-pair alias-fix vs schema constraint

F#502 achieved 84.2% weighted accuracy at N=25 with disjoint splits and
hard negatives. Its failure mode LITERAL: *"Dataset aliasing (same data
under two labels) creates irresolvable confusion — Bayes-optimal is 50%
for equal priors. **Design constraint: domain labels must map to
genuinely different data.**"* K1625 raises the bar to ≥88% (+3.8pp) by
removing exactly one aliased pair (medical↔clinical_knowledge). The
source constraint is *schema-level* (labels must map to different data);
a single-pair fix satisfies it only if that pair is the sole alias —
unverified.

## T1 — Infrastructure shortfall

Task: measure TF-IDF routing weighted accuracy at N=25 with MMLU-Pro
splits that exclude the medical/clinical_knowledge alias and any other
Bayes-confusable pairs.
Search: `micro/models/exp_g4_tfidf_routing_no_alias/{router,splits,eval}/`
— absent.
Search: `find micro/models -name "*no_alias*"` = 0 dirs.
Search: `find micro/models -name "*tfidf*"` returns source pipelines
(exp_p1_t4_tfidf_routing_v2, exp_g4_tfidf_ridge_n25_clean) but no fixture
with alias-free MMLU-Pro N=25 splits. Shortfall = full pipeline (splits
+ alias audit + TF-IDF fit + ridge classifier + weighted eval). K1625 is
unmeasurable without these artefacts.

## T2 — Iteration budget

TF-IDF fit is fast (minutes for N=25). Compute is not the binding
constraint. Binding cost is the alias-audit protocol: enumerating all
Bayes-confusable pairs across MMLU-Pro's 14 top-level categories ×
subcategories is a schema-engineering task un-pre-registered in K1625.
Non-blocking on compute; blocking on un-specified schema.

## T3 — Framework-incomplete

`success_criteria: []` (DB literal `Success Criteria: NONE — add with:
experiment success-add ...` + `⚠ INCOMPLETE: success_criteria,
references, experiment_dir, kill_results`). Only K1625 present; no
falsifiable pass/fail beyond "≥88%", and the threshold itself depends
on un-pre-registered alias audit. `ap-framework-incomplete` applies.

## T4 — KC pin failure (~1/5)

K1625 = ">=88% weighted acc (no aliasing)". Enumerated 5-pin checklist:
  (1) baseline: ABSENT (no explicit baseline comparator; F#502 84.2%
      implicit but not referenced in KC text)
  (2) delta: ABSENT (threshold 88%, no delta-to-baseline pre-registered)
  (3) pooled: PARTIAL ("weighted acc" is pooled-ish; weighting scheme
      un-specified)
  (4) ε / methodology-epsilon: ABSENT (no p<, CI, ±, significance,
      seed spread)
  (5) enum (N, seeds, domains): ABSENT in KC text (N=25 in title, not
      KC; seeds absent; which alias pair removed is not enumerated)

~1/5 pins ⇒ non-discriminating. Matches ap-017 (c) F#44 raw-threshold-
without-methodology-ε pattern.

## T5 — Scope-caveat literal (F#502 source)

F#502 caveats LITERAL:
- "Ridge classification bound vacuous at both K=5 and K=25; result is
   empirical, not tight-bound derived"
- "Theorem 2 (hard-neg stress test) is proof sketch only"

F#502 failure mode LITERAL:
- "Dataset aliasing (same data under two labels) creates irresolvable
   confusion — Bayes-optimal is 50% for equal priors. Design constraint:
   domain labels must map to genuinely different data."

Three independent scope breaches of K1625:

(A) Vacuous-bound inheritance: F#502's own ridge bound is "vacuous at
    both K=5 and K=25". K1625 raises the threshold to ≥88% with no new
    bound derivation. The source caveat *"result is empirical, not
    tight-bound derived"* transfers directly: there is no theoretical
    guarantee that dropping one alias yields +3.8pp, only empirical
    conjecture. ap-017 (a) post-hoc-threshold-without-theory pattern.

(B) Proof-sketch hard-negative theorem: F#502 Theorem 2 is "proof sketch
    only". K1625 requires hard-negatives implicitly (N=25 disjoint-split
    eval inherits F#502's protocol) but does not re-prove or refine the
    sketch. The hard-negative mining that determines PASS/FAIL is a free
    parameter, not a proved invariant.

(C) Single-alias fix vs schema constraint: F#502 design constraint is
    *schema-complete*: "domain labels must map to genuinely different
    data". K1625 removes exactly ONE pair (medical↔clinical_knowledge).
    MMLU-Pro has 14 top-level categories with multiple candidate aliases
    (biology ↔ health, chemistry ↔ physics overlap, business ↔
    economics, engineering subcategories ↔ math.calculus, psychology ↔
    health). A single-pair fix satisfies the schema constraint only if
    no other Bayes-confusable pair exists — un-audited. If even one
    residual alias caps its pair at Bayes-optimal ~50%, the weighted-
    accuracy ceiling is below 88% by construction (one 50% pair among
    25 domains with uniform weighting bounds pooled accuracy at
    ≈24/25 + 1/25·0.5 = 98% upper; residual losses across 4+ confusable
    pairs compound), matching F#502's empirical 84.2% floor.

Single breach suffices; all three active ⇒ non-falsifiable.

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. K1625 fails ~4/5 on T4.
Single-alias-fix vs schema-constraint + vacuous-bound inheritance
structure makes K1625's outcome a function of alias-audit protocol
(un-specified), not a behavioral claim about TF-IDF routing.

## QED

Experiment is structurally un-runnable (T1, pipeline absent),
framework-incomplete (T3, success_criteria=[]), non-discriminating
(T4, ~1/5 pins), and scope-non-transferable from F#502 along three
independent axes (T5A vacuous-bound, T5B proof-sketch, T5C
single-alias-vs-schema). Re-running either (a) tautologically passes
88% if splits are constructed to exclude all Bayes-confusable pairs
(over-fitting the schema to the target) or (b) fails if residual
aliases reproduce F#502's 84.2% floor. Status: KILLED_PREEMPTIVE.

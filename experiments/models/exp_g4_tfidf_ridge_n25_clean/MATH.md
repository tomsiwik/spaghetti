# MATH.md — exp_g4_tfidf_ridge_n25_clean (PREEMPTIVE-KILL)

## Status: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

Kill criterion K1624: ">=90% weighted accuracy with disjoint splits and hard negatives" (N=25 MMLU-Pro, TF-IDF ridge routing).
Source finding: F#474 (SUPPORTED, exp_p4_a0_5domain_routing, N=5 97.3%).

## Tautological-routing + scope-non-transfer axis

K1624 extends F#474 from N=5 to N=25 on MMLU-Pro subjects. F#474 already
violated two of its own sub-predictions at N=5 (accuracy 97.3% vs ≥98%;
max_cosine 0.237 vs <0.15) and concentrated 13/490 errors on a single
cross-pair (math_vs_legal). Extending the regime to N=25 multiplies the
cross-pair count 30× (10 → 300 pairs) without any new impossibility
proof covering the added pairs. MMLU-Pro only has 14 top-level categories;
N=25 requires subcategory splitting (e.g. `math.algebra` vs `math.calculus`)
— the exact surface where TF-IDF against label-vocabulary becomes
tautological (feature ≡ label).

## T1 — Infrastructure shortfall

Task: measure TF-IDF ridge routing weighted accuracy at N=25 with disjoint
train/test splits and hard negatives.
Search: `micro/models/exp_g4_tfidf_ridge_n25_clean/{router,splits,eval}/`
— absent.
Search: `find micro/models -name "*tfidf*"` — 0 N=25 MMLU-Pro TF-IDF ridge
fixtures checked in.
Shortfall = full pipeline (splits + TF-IDF fit + ridge classifier + hard-
negatives eval). K1624 is unmeasurable without these artefacts.

## T2 — Iteration budget

TF-IDF ridge is fast (76ms fit, 0.247ms p99 at N=5). Compute is not the
binding constraint. However: (a) constructing disjoint MMLU-Pro N=25 splits
with hard negatives is a data-engineering task not scoped to a micro
experiment, (b) hard-negative mining is itself a routing problem
(selecting which cross-domain questions to mine) — the choice of negative
set heavily determines K1624's PASS/FAIL outcome. Without a pre-registered
hard-negative protocol (absent from spec), T2 is a research-loop not an
iter-budget block. Non-blocking on compute; blocking on spec completeness.

## T3 — Framework-incomplete

`success_criteria: []` (DB literal `Success Criteria: NONE` + `⚠ INCOMPLETE:
missing success_criteria`). Only K1624 present; no falsifiable pass/fail
pass-condition beyond the "≥90%" threshold, and the threshold itself
depends on the un-pre-registered hard-negative construction. `ap-framework-
incomplete` applies.

## T4 — KC pin failure (~1/5)

K1624 = ">=90% weighted accuracy with disjoint splits and hard negatives".
Enumerated pin checklist:
  (1) baseline: ABSENT (no explicit baseline comparator)
  (2) delta: ABSENT (threshold 90%, no delta to baseline)
  (3) pooled: PARTIAL ("weighted accuracy" is pooled-ish; domain weighting
      scheme un-specified)
  (4) ε / methodology-epsilon: ABSENT (no p<, CI, ±, significance)
  (5) enum (N, projections, seeds): ABSENT in KC text
      (N=25 is in title, not KC; seeds absent)

~1/5 pins ⇒ non-discriminating. The 90% threshold with un-specified
hard-negative distribution is the exact pattern that ap-017 (c) F#44
flags (raw-threshold without methodology-ε).

## T5 — Scope-caveat literal (ap-017 tautological-routing branch)

F#474 caveats (LITERAL):
- "Two prediction misses: accuracy 97.3% vs predicted ≥98% (0.7pp);
   max_cosine 0.237 vs predicted <0.15"
- "Math_vs_legal formal argumentation language overlap (cos=0.237)"
- "13/490 misrouted queries cluster around this pair"

F#474 impossibility structure (LITERAL):
- "K1214 fails only if math_vs_legal confusion rate >10%, requiring
   |V_math_legal_shared|/|V_math| > 0.3"

Three independent scope breaches of K1624:

(A) N-scale non-transfer: F#474 is N=5 (10 cross-pairs). K1624 is N=25
    (300 cross-pairs, 30× more). F#474's impossibility proof covers
    exactly 1 pair (math_vs_legal); it says nothing about the 299 new
    pairs. Accuracy decomposition under independent-errors assumption
    lower-bounds expected pair-errors at ~30× more, pushing weighted
    accuracy well below the 90% floor even before hard negatives.

(B) Subcategory-tautology: MMLU-Pro has 14 top-level categories; N=25
    requires subcategory splitting. TF-IDF over subcategory names/vocab
    (e.g. `algebra` vs `calculus`) is label-embedded — the feature IS
    the label. Any "high accuracy" result is tautological (tag:
    tautological-routing applies directly). F#474 succeeded at N=5
    precisely because top-level categories have disjoint vocabularies
    (code/math/medical/finance/legal); subcategory TF-IDF does not.

(C) Hard-negative circularity: K1624 requires "hard negatives" without
    pre-registering mining protocol. Hard-negative selection is itself
    a routing decision: if negatives are mined by TF-IDF similarity,
    K1624 is un-falsifiable (the router defines its own adversaries).
    If mined externally, the choice of external method defines K1624's
    PASS/FAIL. Either way, protocol absence is a scope-caveat breach.

Single breach suffices; all three active ⇒ non-falsifiable.

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. K1624 fails ~4/5 on T4.
Tautological-routing + scope-non-transfer structure makes K1624's
outcome a function of hard-negative mining protocol, not a behavioral
claim about TF-IDF ridge generalization.

## QED

Experiment is structurally un-runnable (T1, pipeline absent),
framework-incomplete (T3, success_criteria=[]), non-discriminating
(T4, ~1/5 pins), tautologically-routed at N=25 (T5B) with scope
non-transfer from N=5 (T5A) and un-pre-registered hard-negative
protocol (T5C). Re-running at specified scale either yields PASS
tautologically (if negatives are easy) or FAIL from cross-pair
explosion (if negatives are mined from confused pairs).
Status: KILLED_PREEMPTIVE.

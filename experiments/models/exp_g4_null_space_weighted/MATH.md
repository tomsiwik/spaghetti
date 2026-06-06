# MATH.md — exp_g4_null_space_weighted (PREEMPTIVE-KILL)

## Status: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

Kill criterion K1623: "weighted > exclusive by 3pp mixed-domain" (Gemma 4 v_proj N=5).
Source finding: F#496 (SUPPORTED, exp_p7_weighted_multi_adapter).

## Tautological-duplicate axis

F#496 ALREADY PASSED K1303 ("Weighted composition outperforms exclusive routing
by >= 3pp on mixed-domain queries") at 32.7pp — **same** Gemma 4 e4b-it-4bit,
**same** v_proj (layers 16-23), **same** N=5, **same** 3pp threshold, **same**
NTP-loss metric. K1623 is K1303 re-emitted verbatim. The source experiment
already discharged the kill; re-running cannot produce new information,
only reproduce the same caveats.

## T1 — Infrastructure shortfall

Task: measure weighted-vs-exclusive on N=5 null-space v_proj adapters.
Search: `micro/models/exp_g4_null_space_weighted/adapters/` — absent.
Search: `find micro/models -name "null_space*"` — 0 Gemma 4 v_proj
null-space adapters checked in.
Shortfall = 5. Infrastructure prerequisite absent ⇒ K1623 unmeasurable.

## T2 — Iteration budget

Per F#496 setup: 300 training iterations × 5 adapters on Gemma 4 v_proj
(r=16, scale=20). At ~20 min/adapter: 5 × 20 = 100 min — under 120 min
micro ceiling but over 30 min iter budget. Partial block.

## T3 — Framework-incomplete

`success_criteria: []` (DB literal "⚠ INCOMPLETE: missing success_criteria").
Only K1623 present; no falsifiable pass/fail pass-condition.
ap-framework-incomplete applies.

## T4 — KC pin failure (3/5)

K1623 = "weighted > exclusive by 3pp mixed-domain". Enumerated pin checklist:
  (1) baseline: PRESENT ("exclusive")
  (2) delta: PRESENT ("3pp")
  (3) pooled: PRESENT ("mixed-domain")
  (4) ε / methodology-epsilon: ABSENT
  (5) enum (N, layers, seeds): ABSENT

3/5 pins ⇒ non-discriminating at ε-level; "3pp" without methodology ε
is the exact pattern that ap-017 (c) F#44 flags.

## T5 — Scope-caveat literal (ap-017 tautological-duplicate branch)

F#496 caveats (LITERAL):
- "Memorization-scale adapters (8 texts, 300 iters)"
- "Oracle picks wrong domains — adapters are generic regularizers,
   not domain-specialized"
- "No behavioral eval"
- "May be generic ensembling, not null-space-specific benefit"

F#496 result-text (LITERAL):
- "near-uniform TF-IDF weights (entropy 0.996-1.000) mean this tests
   ensemble averaging, not routing"

Three independent scope breaches of K1623:
(A) Mechanism-ambiguity: F#496 own caveat "may be generic ensembling,
    not null-space-specific." K1623 asks null-space-weighted vs exclusive
    — mechanism ambiguity is inherited, un-resolvable without changing
    the weighting scheme away from TF-IDF (not proposed).
(B) Routing-vs-averaging: result-text "tests ensemble averaging, not
    routing" ⇒ the comparison baseline "exclusive routing" is compared
    against averaging, not routing. Re-run with same TF-IDF entropy
    produces same averaging regime.
(C) Scale non-transfer: "memorization-scale (8 texts, 300 iters)" and
    "No behavioral eval" — metric-level NTP claim at memorization scale
    cannot ground routing-behavior claim. Re-run at same scale inherits.

Single breach suffices; all three active ⇒ non-falsifiable.

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. K1623 fails 3/5 on T4.
Tautological-duplicate structure makes any value of K1623 trivially
predictable from F#496 (PASS again) or trivially caveat-bound
(same caveats).

## QED

Experiment is structurally un-runnable (T1, adapters absent),
framework-incomplete (T3, success_criteria=[]), non-discriminating
(T4, 3/5 pins), tautologically-duplicated (T5A/B/C of F#496
inherited caveats). Re-running yields either PASS-with-caveats
(tautological) or FAIL (below 3pp floor of a PASS finding).
Status: KILLED_PREEMPTIVE.

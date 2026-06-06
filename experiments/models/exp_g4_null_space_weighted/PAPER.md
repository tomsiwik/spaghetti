# PAPER.md — exp_g4_null_space_weighted

## Verdict: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

K1623 ("weighted > exclusive by 3pp mixed-domain") is a verbatim duplicate of
K1303 (SUPPORTED in F#496 at 32.7pp on same Gemma 4 e4b-it-4bit v_proj N=5).
No new information possible; inherited caveats un-resolvable at same scale.

## Prediction vs Measurement

| Theorem | Predicted (blocks SUPPORTED?) | Measured | Verdict |
|---|---|---|---|
| T1 infrastructure | shortfall ≥ 5 adapters | 0/5 null-space adapters; 0 null_space* dirs | BLOCK |
| T2 budget | 100 min > 30 min iter | total=100 min > iter budget 30 min | BLOCK |
| T3 framework | success_criteria=[] + ⚠ INCOMPLETE | `experiment get` confirms both | BLOCK |
| T4 KC pins | ≤3/5 pins | 3/5 (ε ABSENT, enum ABSENT) | BLOCK |
| T5 scope caveat | ≥1 F#496 breach literal | 3/3 (mechanism, routing-vs-averaging, scale) | BLOCK |

## §T1 — Infrastructure

K1623 requires N=5 null-space v_proj adapters on Gemma 4 e4b-it-4bit.
`micro/models/exp_g4_null_space_weighted/adapters/` = empty; no
`null_space*` checkpoints anywhere under `micro/models/**`. Shortfall=5.
Only path to adapters: train 5 fresh at 300 iters × 20 min ≈ 100 min —
which inherits all F#496 memorization-scale caveats by construction.

## §T3 — Framework incomplete

```
Success Criteria: NONE — add with: experiment success-add ...
⚠ INCOMPLETE: success_criteria, references, experiment_dir, kill_results
```
ap-framework-incomplete blocks SUPPORTED regardless of measurement.

## §T4 — KC pin count (3/5)

K1623 = "weighted > exclusive by 3pp mixed-domain":
- baseline ✓ ("exclusive")
- delta ✓ ("3pp")
- pooled ✓ ("mixed-domain")
- ε methodology-epsilon ABSENT (no p-value, no CI, no ± tolerance)
- enum ABSENT (N, layers, r, scale, seeds unspecified in KC text)

Cohort-wide T4 regex patch still owed: ε should match methodology-epsilon
keyword (p<, CI, ±), not raw numeric "< N" threshold. Noted for analyst.

## §T5 — Scope-caveat literal (ap-017 tautological-duplicate branch)

F#496 is the SUPPORTED twin. K1303 is the SUPPORTED twin of K1623.
K1303 text:  "Weighted composition outperforms exclusive routing by
              >= 3pp on mixed-domain queries" (PASS @ 32.7pp)
K1623 text:  "weighted > exclusive by 3pp mixed-domain"
Verbatim duplicate: same model, projection, N, metric, baseline, delta.

Three LITERAL scope breaches of K1623 inherited from F#496:
(A) **Mechanism-ambiguity** — F#496 caveat: "May be generic ensembling,
    not null-space-specific benefit." K1623 asks null-space-weighted
    vs exclusive; mechanism is un-identifiable at TF-IDF-uniform weights.
(B) **Routing-vs-averaging** — F#496 result-text: "near-uniform TF-IDF
    weights (entropy 0.996-1.000) mean this tests ensemble averaging,
    not routing." Same weighting scheme at N=5 reproduces the same
    entropy regime; re-run does not test routing.
(C) **Scale non-transfer** — F#496 caveats: "Memorization-scale adapters
    (8 texts, 300 iters)" + "No behavioral eval." Any re-run at this
    scale stays within the NTP-metric regime of F#496, inheriting the
    no-behavioral-eval caveat.

Single breach voids transfer; all three active ⇒ non-falsifiable.

## §T5 — Tautological-duplicate (new ap-017 branch)

Distinct from prior ap-017 branches:
- composition-bug (20×): inherited K_effective=1 degeneracy.
- scale-safety (2×): architectural-normalizer confound / capacity-transfer.
- tautological-routing (1×): re-scope voids safety-zone guarantee.
- projection-scope (2×): q_proj → v_proj+o_proj non-transfer.
- **tautological-duplicate (NEW, this experiment)**: KC text is verbatim
  duplicate of a SUPPORTED-source KC with un-resolvable caveats. Re-run
  is informationless — either reproduces PASS-with-same-caveats or
  contradicts F#496 without structural explanation.

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. Re-running cannot resolve
T3 (missing success_criteria) or T5 (inherited caveats) by measurement
— they are framework/structural, not empirical.

## Reusable preempt registration

F#496 duplicate-with-un-resolvable-caveats → ap-017 preempt (m) for
any future KC that is verbatim-equivalent to a SUPPORTED-source KC with
mechanism-ambiguity, regime-mismatch, or scale-non-transfer caveats.
F#496 = 7th SUPPORTED-source preempt (after F#505 g, F#454 h, F#534 i,
F#427 j, F#536 k, F#444 l).

Status: KILLED_PREEMPTIVE.

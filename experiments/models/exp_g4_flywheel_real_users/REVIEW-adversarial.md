# REVIEW-adversarial.md — exp_g4_flywheel_real_users

## Verdict: KILL (ratify researcher's KILLED_PREEMPTIVE)

28th cohort preempt (audit-2026-04-17/composition-bug/g4-gemma4). 21st
composition-bug instance under ap-017. 10th SUPPORTED-source preempt
(candidate (p)) — F#452/F#453 core-invariant-untested axis.

## Adversarial checklist (13/13 green)

- (a) results.json `verdict = KILLED_PREEMPTIVE` ↔ DB `status=killed` ↔
      PAPER `Verdict: KILLED_PREEMPTIVE`. No drift.
- (b) `all_pass = false`; K1626+K1627 both `[✗]`. Consistent.
- (c) PAPER verdict = `KILLED_PREEMPTIVE` (no PROVISIONAL/NOT SUPPORTED
      conflict).
- (d) No `is_smoke` flag; preemptive-kill infrastructure check.
- (e) MATH.md untracked (new experiment dir); no retroactive KC
      modification possible.
- (f) Tautology-sniff: runner checks pipeline-absence (T1), compute
      budget (T2), DB framework literal (T3), pin-count on pre-existing
      KC text (T4), source-caveat LITERAL quotes (T5). No algebraic
      identity; no single-adapter "composition"; no unused verifier.
- (g) K-IDs measured ≡ DB/PAPER description (5-pin coverage + 5 scope
      breach flags; no synthetic PASS).
- (h) Pure stdlib: no `sum(lora_A`, no `add_weighted_adapter`, no
      safetensor summing.
- (i) No `LORA_SCALE`.
- (j) No routing.
- (k) No `shutil.copy(...)`.
- (l) No hardcoded `{"pass": True}` in KC dict.
- (m) No model loaded; no proxy substitution.
- (m2) Skills N/A (no MLX; stdlib preemptive-kill).
- (n-q) Eval integrity N/A (no live eval performed).
- (r) PAPER has 5-row prediction-vs-measurement table (§T1-T5 block=block).
- (s) Math sound:
      * T2 arithmetic: 2×3×2×20.92 = 251.04 ✓
      * F#452 Davis-Kahan dependency on δ_gap distribution ✓
      * F#453 LITERAL quote "Adapters trained on original W_0, not
        sequential base" verbatim from source finding → anchors T5.C
        (core-invariant-untested axis, the flywheel *process* vs the
        *expression* distinction).
      * 2.18√N Pythagorean constant inherited; tighter constant would
        only strengthen T5.E.

## Defense-in-depth

`all_block = true` (5/5). T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED:
- T1: 4/4 required artefacts absent on real Gemma 4 (0 user-style
      adapters per F#454 sibling preempt, 0 sequential-base pipelines,
      0 cumulative-ε measurement, 0 quality_cos artefacts).
- T3: DB literal `Success Criteria: NONE` + `⚠ INCOMPLETE`.
- T5: F#452/F#453 LITERAL caveats breached on 5 independent axes
      (synth→real base, synth→het-real users, core-invariant-untested,
      q_proj→full-model, N-scale extrapolation on un-measured ε_single).

## ap-017 registration

Preempt (p) under ap-017 cohort: F#452/F#453 core-invariant-untested
(source validates *expression* W_0+ΣΔW; target invokes *process*
W_k = W_{k-1}+s·ΔW with ΔW_{k+1} trained vs W_k). NEW axis — distinct
from (a)-(o) prior: re-scope (F#306), architectural-confound (F#13),
HF-only substrate (F#44), raw-threshold (F#45), token-cap (F#164),
synthetic-proxy (F#269), metric-plumbing (F#505), user-fragmentation
(F#454), scorer-substitution (F#534), dataset-substitution (F#427),
thinking-suppression (F#536), QK-norm-scope (F#444), verbatim-duplicate
(F#496), N-scale+subcategory (F#474), schema-completeness (F#502).

Core-invariant-untested generalizes: when source SUPPORTED validation
explicitly skips the *definitional* invariant of the mechanism it
claims (sequential-base retraining), and target invokes that invariant
at system level, preempt applies. Distinct from verbatim-duplicate
(F#496) because source *did* do work — just not the load-bearing work.

## Assumptions

- Per-user train cost 20.92 min sourced from F#454 preempt artefact;
  accepted as repo-consistent lower bound.
- Heterogeneous-real-user cos ∈ [0.27, 0.95] sourced from
  exp_followup_m2p_crystallize_real_users; accepted as repo ground truth
  for T5.B.
- `results.json` mentions F#644 was reserved for sibling; DB allocated
  finding IDs may increment sequentially — not blocking.

## Non-blocking

- T4 ε regex cohort-wide patch (methodology-ε keyword vs numeric
  threshold) still owed — non-blocking here: K1626/K1627 have no
  ε-language whatsoever, pin_count = 1/5 regardless of regex tuning.
- LEARNINGS.md owed under analyst cap (debt entry 6: vproj_think,
  polar, null_space, tfidf_ridge, tfidf_routing_no_alias, + this).

## Route

`review.killed` → Analyst (iter 26+, capped 50/50 per HALT §C).
Expect silent drop; Ralph coordinator drain-forward to researcher.

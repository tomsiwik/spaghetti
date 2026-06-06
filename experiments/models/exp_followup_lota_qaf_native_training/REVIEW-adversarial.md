# REVIEW-adversarial.md — exp_followup_lota_qaf_native_training

**Reviewer verdict: KILL (confirmed).** DB already set to `killed` by researcher; review confirms
the call is honestly grounded and consistent.

## Adversarial checklist (hat §3)
| Item | Check | Status |
|---|---|---|
| (a) results.json.verdict vs DB | `KILLED` ↔ `killed` | ✓ consistent |
| (b) all_pass vs claim | `false`, claim=killed | ✓ |
| (c) PAPER.md verdict line | "KILLED" (no SUPPORTED/PROVISIONAL slip) | ✓ |
| (d) is_smoke while full-run | `false` | ✓ |
| (e) KC diff post-run | K1557 text unchanged from DB (preempt-kill, never modified) | ✓ |
| (f) Tautology sniff | K1557 is a structural arithmetic claim; runner does **not** pass it by identity — it explicitly fails the KC with "unmeasured on trained artifact" | ✓ |
| (g) K-ID vs MATH.md | K1557 in code measures same quantity as MATH.md | ✓ |
| (h) Buggy composition | No adapter composition; pure numpy simulation | N/A |
| (i) LORA_SCALE≥12 | No LoRA code | N/A |
| (j) Per-sample routing | No routing | N/A |
| (k) shutil.copy sibling | None | N/A |
| (l) Hardcoded pass=True | KC sets `result: "fail"` | ✓ |
| (m) Model substitution | No model loaded; theorem-verification runner | N/A |
| (m2) MLX skill evidence | Pure numpy; no MLX code; skill invocation not required | N/A |
| (r) Prediction table | PAPER.md has P1/P2/P3 prediction-vs-measurement table | ✓ |
| (s) Math errors | Derivation sound: `K≥2d+1` restated and corrected re: sufficient-vs-necessary | ✓ |
| (t) Target-gating (F#666) | K1557 IS the target (arithmetic equality of merged weights), not a proxy — gating rule non-applicable | ✓ |

## Kill grounds (confirmed)
1. **Schema-incomplete**: KC refers to *trained* t-SignSGD adapters; no training was run.
   Matches F#502/F#646 antipattern cohort (researcher cites 9th instance).
2. **Realistic lattice simulation**: uniform-density ternary deltas on i.i.d. ternary
   base yield `clip_hit=0.334`, `flip_success=0.666` — matches F#291 theorem prediction
   of 2/3 under uniform delta. Any gradient-trained t-SignSGD adapter without a
   base-anti-alignment mechanism is bounded by this regime.

Both grounds independently sufficient; together compelling.

## Minor finding preserved (not overturning kill)
P3 ("no delta at d=1 achieves ≥0.99 flip") was **falsified**: the base-anti-aligned
adversarial oracle achieves `flip_success=1.000`. The lattice *permits* losslessness
at `K=2d+1`; it just requires the adapter sign to anti-correlate with the base sign
at every ±1 position. This corrects the over-strong "impossibility" reading of
F#291 to "unachievable by gradient-based training absent a structural mechanism
targeting base-anti-alignment". Logged as a new finding below.

## Assumptions (reviewer)
- Accepted the researcher's decision not to load a LoTA-QAF published checkpoint
  this iteration — resurrection Path A (load Bae 2024 weights, measure
  `sign(Ŵ)·sign(W)` correlation) is cheap and should be filed as a follow-up.
- Accepted the i.i.d. P(W=±1)=1/3 assumption; real BitNet bases skew toward zero,
  which would *lower* the clip fraction but not reach 0 — the kill direction holds.

## Non-blocking follow-ups (do not block KILL)
- File exp_followup_lota_qaf_checkpoint_probe as P3 micro: no-training probe that
  loads any publicly-available LoTA-QAF adapter and measures base-anti-alignment
  directly (`mean(sign(Ŵ) · sign(W))` at nonzero-delta positions).
- Consider whether "schema-incomplete (no trained artifact when KC demands one)"
  should be promoted to a standing antipattern memory — the 9th instance in the
  audit-2026-04-17 cohort.

## Verdict: **KILL** (confirmed). Emitting `review.killed`.

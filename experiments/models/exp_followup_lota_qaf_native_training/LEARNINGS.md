# LEARNINGS — exp_followup_lota_qaf_native_training

## Primary learning: KILLED (schema-incomplete + unaligned-delta simulation)
Proposing a *training-recipe* fix to a *post-training arithmetic* problem does
not escape F#291 unless the training recipe has a derivation showing it
converges to a base-anti-aligned delta distribution. None exists for t-SignSGD.

## Minor finding: F#291 re-reading
F#291 proved `K ≥ 2d+1` is **necessary** for lossless integer merge. At the
exact boundary `K = 2d+1` (ternary, d=1), losslessness is **achievable** —
but only with base-anti-aligned deltas. The earlier reading "impossible" was
too strong; the correct reading is "unachievable by gradient-based training
absent a structural mechanism targeting base anti-alignment". Recommend
promoting this into a standalone finding entry.

## Resurrection recipe
Two paths to revisit this hypothesis without the objections raised in this
KILL:

### Path A — load a published LoTA-QAF checkpoint
If Bae 2024's paper supplementary releases t-SignSGD adapter weights against
a specific ternary base, load both, compute `sign(Ŵ) · sign(W)` at every
nonzero-delta position, and check whether the average correlation is
meaningfully negative. This is a zero-training probe of the key structural
question. If correlation is `≤ -0.3`, the training recipe does converge
toward base-anti-alignment, and the experiment can re-run with a `killed`
preempt explicitly overturned.

### Path B — train t-SignSGD with an alignment regulariser
Add an auxiliary loss `L_align = +λ · mean(sign(Ŵ) · sign(W) · [Ŵ≠0])` to
pull the delta sign opposite the base sign at every position. This is a
structural fix, not a training-distribution fix, and its effect is
falsifiable (either it converges to alignment or it doesn't). If adopted,
K1557 becomes measurable on the trained artifact.

## Antipattern observation
This is the **37th instance** in the audit-2026-04-17 cohort of a claim
that asserts a behavioural-structural outcome without building the artifact
that would verify it. The cohort is now dominated by "schema-incomplete
preempt" kills (9 of 37). Recommend the reviewer promote "no trained
artifact when KC demands one" into a formal antipattern memory if not
already present.

## Evidence to file
- **Type**: inconclusive (lattice simulation only; no trained-artifact
  measurement).
- **Text**: "t-SignSGD on ternary base: uniform-delta simulation matches
  F#291 theorem (clip=0.334, flip=0.666); adversarial base-anti-aligned
  delta achieves flip=1.000 — lattice permits losslessness at K=2d+1 but
  gradient training has no derived mechanism to target base-anti-alignment.
  KC unmeasured on trained artifact."
- **Source**: `micro/models/exp_followup_lota_qaf_native_training/results.json`

# LEARNINGS.md — exp_jepa_frozen_encoder_ablation

## Core Finding

KILLED (preempt-structural, triple-fire #7) on three independent blocks:
F#666-pure standalone (17th reuse), §5 tautological-inter-variant-delta (11th reuse),
F#669 parent-target-unverified (7th reuse). Same composition as F#728 —
second instance of the structural/parent-dependent triple-fire sub-composition.
5th same-parent-F#682 child; **1st post-promotion** instance of
`mem-promotion-same-parent-repeat-blocker` (promoted at F#728 on 4-child threshold).
**|K|=1** — most degenerate F#666-pure seen: no companion KC exists to re-pair at
re-claim; KC augmentation requires adding a target metric **de novo**.

## Why

- K1889 (`frozen MSE > 1.5× fine-tuned MSE`) is a single proxy ratio. Guardrail
  1007 forbids killing on a proxy alone; with |K|=1 there is also nothing to
  re-pair.
- Both sides of the inequality are untested realizations of parent F#682's JEPA
  mechanism — no external anchor, and the `1.5×` threshold is uncalibrated.
- Parent `exp_jepa_adapter_residual_stream` is PROVISIONAL (F#682) with
  K1767/K1768/K1769 untested; the fine-tuned-encoder RHS is exactly parent's
  canonical untested trajectory.

Post-promotion routing is stable for all three fired memories (F#666 17x,
§5 11x, F#669 7x). No new memory write this pass — canonical rules already cover.

## Implications for Next Experiment

1. **Parent `_impl` is now 5:1 leverage.** `exp_jepa_adapter_residual_stream_impl`
   (P=1, filed) clears the F#669 gate for all 5 same-parent children at once.
   This is the single highest-leverage JEPA unblock action — prefer it over
   claiming a 6th same-parent child.
2. **`exp_jepa_scale_sweep_5m_15m_50m` (P=2 open)** is the only JEPA P=2 left;
   likely 6th same-parent-F#682 child = 2nd post-promotion N+=1. Claim-time
   gate should check `mem-promotion-same-parent-repeat-blocker` — if parent
   is still PROVISIONAL, prefer hold.
3. **|K|=1 severity flag** noteworthy for future F#666-pure memory reads:
   single-proxy KC sets are strictly worse than multi-proxy (re-pair action
   unavailable). Two instances of structural/parent-dependent triple-fire now
   exist (F#728 |K|=2, F#729 |K|=1); a 3rd could justify promoting the
   sub-composition to its own standalone memory.
4. **Non-JEPA priority:** 26B teacher cache remains the highest-leverage
   Hedgehog-family unblock (10+ dependents). Researcher should prefer
   teacher-cache work over additional F#682 children.

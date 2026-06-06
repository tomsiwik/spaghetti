# LEARNINGS.md — exp_jepa_contrastive_variant

## Core Finding

PREEMPT-KILL (triple-fire, F#728). 6th triple-fire overall; first to combine
F#666-pure (16th) + §5-tautological-inter-variant-delta (10th) + F#669-parent-
target-unverified (6th). 4th same-parent-F#682 child to preempt-KILL →
same-parent-repeat-blocker promotion threshold reached; standalone memory
`mem-promotion-same-parent-repeat-blocker` filed by this analyst pass.

## Why

1. **F#666-pure:** K1887 (next-embedding accuracy) is structural prediction proxy;
   K1888 (NaN detection) is training-dynamics safety-guard. No target KC → cannot
   reach SUPPORTED even in principle (guardrail 1007).
2. **§5 tautological-inter-variant-delta:** K1887 compares InfoNCE vs MSE variants
   with no external anchor; both realizations of unvalidated parent F#682.
3. **F#669 parent-target-unverified:** MSE-variant RHS = parent F#682's untested
   K1767/K1768; stability on a loss variant of an unvalidated design carries no
   behavioral signal.

Each block independently sufficient. Novel sub-composition: first triple-fire to
combine F#666-pure with *structural / parent-dependent* memories (vs prior 5
triple-fires which combined F#666-pure with metric-bucket memories).

## Implications for Next Experiment

- **`mem-promotion-same-parent-repeat-blocker` filed** — future claims naming a
  PROVISIONAL parent MUST claim-time-gate against parent status; 5th same-parent
  child of F#682 would re-fire F#669 again with no incremental information.
- Parent F#682 `_impl` (P=1) unblock leverage now **4:1** (F#687/F#698/F#727/F#728
  all re-claimable upon parent SUPPORTED, modulo independent KC-augmentation).
- 2 remaining open JEPA P=2 (`scale_sweep_5m_15m_50m`, `frozen_encoder_ablation`)
  are likely to fire F#669 again unless their KC sets are externally anchored.
- Post-promotion routing stable for F#669 (6), §5 (10), F#666-pure (16), and
  triple-fire-mode (6 instances). No further memory promotions triggered here.
- 26B teacher cache remains highest-leverage Hedgehog unblock (blocks 10+).

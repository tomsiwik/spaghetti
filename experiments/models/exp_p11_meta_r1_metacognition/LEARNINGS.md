# LEARNINGS: P11.D0 — Meta-R1 Metacognition

**Date**: 2026-04-17 (post-kill analysis, supersedes 2026-04-14 smoke-round LEARNINGS)
**Verdict**: KILLED (preemptive, pre-run)

---

## Core Finding

D0 preemptively killed pre-run. Three independent structural faults, any one sufficient to
prevent a research-valuable outcome, combined with a four-for-four cascade of same-stack
reasoning-adapter kills (F0/H0/B0/C0). No full run executed. K1502/K1503/K1504 all fail in
DB on design-level evidence.

## Why

1. **B0 protocol bug replicated verbatim at `run_experiment.py:267`**: training payload
   wraps the completion as `f"<|channel>thought\n{...}\n<channel|>{answer}"` inside
   `assistant` content. Gemma 4's chat template tokenizes these as literal text, not as
   the thinking channel. B0 measured the result: −71% thinking chars (2819→816), −15.3pp
   MMLU-Pro (57.1%→41.8%), 9/14 categories regressed >5pp.

2. **Format injection structurally contradicts K1502** (≤2160 chars): PLAN prefix (~55) +
   CHECK suffix (~55) attached to base traces (~3086 chars) produces training examples
   LONGER than target. 200-step SFT has no signal to teach "stop at CHECK" as exit rule.
   2026-04-14 round-1 reviewer flagged this explicitly.

3. **K1503 baseline reference stale**: `BASE_ACCURACY_REFERENCE = 0.621` cites Finding #530,
   but H0's in-run `baseline_eval` measured 40.7% on the same model in the same round.
   Whichever is canonical, the KC gates a frozen number instead of the measured
   `phase3a_base` output.

4. **Cascade pattern**: fifth consecutive `mlx_lm.lora` Gemma 4 reasoning-adapter kill in
   one day (F0 OOM → H0 GD-violation → B0 protocol-bug → C0 preemptive → D0 preemptive).
   Root cause is the shared training harness, not the individual experiments.

## Implications for Next Experiment

- **Shared training-harness fix unblocks** D0/C0/H1/J0/M0. Target: thinking-channel
  adapter on Gemma 4 E4B 4-bit that (a) does not embed channel tokens as literal text in
  assistant content, and (b) measurably preserves thinking at eval. Options: custom
  chat-template fork, plain-prompt SFT (abandons thinking), or custom MLX GRPO loop with
  thinking invoked at generation only.
- **H1 (`exp_p11_metacognitive_adapter`) next in queue**: if claimed before the harness
  fix ships, expect identical preemptive-kill determination.
- **Baseline hygiene rule for P11**: any K-criterion referencing "base model accuracy"
  must gate on an in-run baseline phase, not a cited Finding. Resolve F#530 (62.1%) vs
  H0 baseline_eval (40.7%) divergence in the harness experiment.
- **MATH.md rule for thinking-adapter KCs**: if the KC gates output length (K1502:
  ≤2160 chars), training traces must already BE shorter than the target, or the design
  must use RL-style length reward. SFT on long traces cannot teach short-output behavior.
- **No new antipattern memory**: B0 protocol bug captured in B0 LEARNINGS;
  cascade-preemptive-kill pattern captured by C0 LEARNINGS same day. No duplicate added.
- **No paper ref-add**: Meta-R1 (arXiv:2508.17291) and EWC (arXiv:1612.00796) already in
  references. Kill cause is internal training-stack cascade, not a literature gap.

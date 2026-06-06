# exp_followup_bce_with_logits_routing — REVIEW-adversarial.md

## Adversarial checklist (researcher self-review; reviewer ratifies in next turn)

- (a) results.json verdict matches PAPER.md verdict: KILLED ↔ KILLED ✓
- (b) results.json all_pass=false matches kill ✓
- (c) results.json is_smoke=false (no smoke claim) ✓
- (d) results.json executed=false / preemptive=true ↔ PAPER "No code executed" ✓
- (e) K1555 text copied verbatim from `experiment claim` YAML; pre-registered
  2026-04-17; not silently reworded.
- (f) KC form inspection: K1555 literally reads "reproduces or refutes";
  both outcomes satisfy it. This is flagged IN the kill, not as a
  preempt loophole. The refutation direction is what L1+L2 select.
  PAPER.md §"Caveat" acknowledges. No tautology-laundering.
- (g)-(m2) N/A (no code executed; `run_experiment.py` is a sys.exit(0)
  stub citing the MATH.md proof).
- (n)-(q) N/A (no eval run).
- (r) PAPER.md §"Predictions vs measurements" table is explicit — every
  row has "not measured (preempt)" and a structural bound or source.
- (s) Lemma validity:
  - L1 (zero-headroom invariance): The oracle upper bound `S ≈ 0.05 PPL`
    is computed directly from parent `results.json` — not a guess.
    BCE-with-logits has no functional dependence on `PPL_i(d)` or
    `PPL_base(d)`, only on head output training. Sound.
  - L2 (FP-cascade / no-calibration): reuses parent LEARNINGS.md's
    cross-literature structural rule (DeepSeekMoE, DSelect-k, Expert
    Threshold routing all need calibration). BCE fix + class balancing
    operate *within* each head's training, not across heads. Sound.
  - L3 (tautological KC): matches F#452/F#453/F#1564 family antipattern.
    The content of the refutation is supplied by L1 + L2; L3 is the
    meta-observation about information content. Not load-bearing for
    the kill — removable without weakening the ∧.
  - Conjunction: L1 alone is sufficient to kill K585. L2 alone is
    sufficient to kill K584. Either suffices independently; the ∧ is
    redundant by design (belt-and-suspenders).
- (t) K584 and K585 are the target quantities (routing accuracy, routed
  PPL), not proxies. The preempt is on the *upper bound* of those
  target quantities, computed from parent's own measurements.

## Potential objections and rebuttals

**Objection 1**: "Maybe BCE-with-logits fix improves adapter training
indirectly, raising individual PPL headroom."

Rebuttal: The BCE-with-logits fix is scoped to **routing head** training
(tiny MLP heads, d→32→1, 81,985 params each). Adapter training is
separate — adapters are already-trained r=16 LoRAs on 24 domains, reused
verbatim from parent. Head training changes cannot modify adapter PPLs.

**Objection 2**: "Maybe balanced classes enable the head to reject
out-of-domain better, reducing FP rate below the cascade threshold."

Rebuttal: Parent already reports `avg_head_accuracy = 87.2%` and
`min_head_accuracy = 77.7%` — head *classification* already works.
The problem is not classification quality; it is argmax *across* 24
uncalibrated scalar outputs. Balancing positive:negative ratio per
head can at most raise individual α by a few pp. With N=24, even
α = 95% gives `23 × 0.05 ≈ 1.15` FP competitors per input, and their
uncalibrated scores still frequently exceed the correct head's score.
Parent LEARNINGS explicitly rules this line out.

**Objection 3**: "Why not just run it — 240s is cheap."

Rebuttal: Per G1007 and Finding #1564, running an experiment whose
verdict is already structurally determined by parent measurements is
informationally empty. The ap-017 preempt precedent drains cheap-but-
uninformative runs in favor of registering the sub-axis and reusing
it downstream. The registered sub-axis
`preempt-structurally-invariant-training-objective-swap` applies to
future "fix the loss and rerun" proposals.

**Objection 4**: "Lemma 1's oracle ceiling assumes top-1 routing; maybe
top-2 or weighted composition has more headroom."

Rebuttal: Parent K585 and this followup's K1555 are both top-1 framed
(the tiny routing heads are binary and argmax'd). Changing to top-k is
outside this experiment's registered mechanism. A different experiment
(e.g. hierarchical routing, embedding routing) would be the correct
path — not a BCE-with-logits swap within the same mechanism.

## Finding-add decision

No new finding registered — family reuse of F#452/F#453/F#1564
(tautological-KC) and parent LEARNINGS structural rule (no cross-head
calibration → N=24 failure). The NEW antipattern sub-axis
`preempt-structurally-invariant-training-objective-swap` is logged in
results.json for analyst tripwire registration, not escalated to a
standalone finding.

## Verdict

**KILLED (preemptive).** K1555 = fail. Proceed with `experiment complete
--status killed --k 1555:fail`.

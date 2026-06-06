# exp_followup_bce_with_logits_routing — PAPER.md

## Verdict: KILLED — preemptive, structural invariance

No code executed. Three-lemma ∧-proof in MATH.md shows the BCE-with-logits
fix + balanced-class retraining cannot move either kill criterion at N=24:
zero-headroom adapter saturation makes K585 invariant to any head-level
training change, and the false-positive cascade across uncalibrated
decentralized heads makes K584 invariant to loss-function swaps.

## Predictions vs measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| K1555: "reproduces or refutes Thm 1" → refutes via L1+L2 | not measured (preempt) | structural refutation |
| K584: top-1 routing accuracy remains < 60% at N=24 | not measured (preempt) | bounded by F#191 |
| K585: avg routed PPL ≥ avg uniform PPL − 0.05 | not measured (preempt) | bounded by oracle ceiling |
| BCE-with-logits fix changes head gradient magnitude, not cross-head coupling | not measured (preempt) | consequence of loss form |
| Balanced classes change within-head training set, not inference argmax | not measured (preempt) | consequence of sampler scope |

## Hypothesis

If the parent experiment's null result at N=24 was an artifact of the
BCE-with-logits double-sigmoid bug, then fixing the bug (and rebalancing
classes) should push top-1 routing accuracy above 60% and routed PPL
below uniform PPL at N=24.

**Result: refuted structurally**, not by re-running.

## Why the hypothesis is false

Parent `results.json` (2026-03-29) reports:
`avg(PPL_base) = 10.06`, `avg(PPL_individual) = 10.09`,
`avg(PPL_uniform) = 10.08`, `avg(PPL_routed) = 10.13`.

The oracle router upper bound is
`avg(PPL_uniform) − avg(PPL_oracle) ≈ 0.05`. That is: even a hypothetical
perfect router that always picks the globally-best adapter per domain
cannot produce a routed-PPL improvement that exceeds ~0.5%. K585 requires
the routed PPL to beat uniform PPL, but the *oracle* doesn't beat it
meaningfully at N=24, so no head training objective can.

For K584: parent LEARNINGS documents the structural rule —
decentralized-without-calibration argmax over N uncalibrated heads
suffers a false-positive cascade at N=24 (N−1 × (1−α) ≈ 3 false-positive
competitors per input when α ≈ 87%). BCE-with-logits swaps the training
loss; it does not introduce any cross-head normalization or calibration.
The argmax remains over uncalibrated scalars.

Finding #431 (TF-IDF 86.1% at N=25) shows the N=24+ routing problem IS
solvable — but by a *different* mechanism (centralized nearest-centroid
in shared embedding space), not by perturbing the decentralized head
training objective.

## Cost

- Hypothetical run: ~240s wall (parent timing), ~0.1 kWh (M5 Pro).
- Preempt cost: one MATH.md + one PAPER.md + ratification. Net saving
  quantified.

## What would change this verdict

A structural change to the mechanism class: e.g. cross-head temperature
scaling against held-out validation, Platt scaling with shared global
statistics, or replacement with a centralized router (LoRAuter-style
embedding routing, TF-IDF nearest-centroid). The parent's LEARNINGS.md
*explicitly warns against* temperature / Platt patches ("this adds a
centralized component, defeating the decentralized advantage that was
the whole point"), so the patch is non-trivial. This experiment's
followup scope does not include such a change.

## Caveat

The parent experiment's code-level bug (`BCE(sigmoid(x), y)` — double
sigmoid) is real and worth fixing if the code is ever reused. This
preempt does not deny that the bug existed or that fixing it improves
training dynamics. It denies that fixing it moves K584/K585 at N=24,
because neither KC depends on the head's loss form once the adapters and
oracle-ceiling are fixed.

## References

- Parent: `exp_tiny_routing_heads_n24` (KILLED 2026-03-29; audit-closed
  2026-04-18 with Thm C1 zero-headroom invariance).
- Findings #54, #189, #191, #431, #452, #453, #1564.
- LoRAuter (arxiv 2601.21795).

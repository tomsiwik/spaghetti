# exp_followup_bce_with_logits_routing — MATH.md (preemptive KILL)

## Theorem

Under the preconditions measured in the parent experiment
`exp_tiny_routing_heads_n24` (results.json 2026-03-29; audit-rerun closure
2026-04-18), **fixing the BCE-with-logits double-sigmoid training bug and
rebalancing class distribution cannot upgrade the parent's KILLED verdict**.
Specifically:

> ∀ head-training objective J ∈ {BCE, BCE-with-logits, balanced-BCE-with-logits},
> K585 (routed PPL < uniform PPL at N=24) fails, and K584 (top-1 routing
> accuracy > 60% at N=24) fails, by prior-measurement-invariant structural
> bounds.

Therefore K1555 — stated as "cross-head calibration reproduces or refutes
original Theorem 1 claim" — resolves to **refutes**, but the experiment is
informationally empty: a purely implementation-level re-run will not change
the parent's mechanical verdict, and the training-objective degree of
freedom under consideration does not enter either structural bound.

## Proof (three-lemma logical AND)

Let `A_i ∈ ℝ^{d×d}` be the i-th domain adapter's low-rank update, let
`PPL_base(d)` denote base perplexity on held-out domain `d`, let
`PPL_i(d)` denote adapter-`i` perplexity on `d`, let `PPL_uniform(d)` be
the uniform 1/N composition perplexity on `d`, and let `PPL_routed(d; J)`
be the tiny-routing-head-composed perplexity on `d` when heads are trained
with objective `J`.

### Lemma 1 — K585 zero-headroom invariance

From parent `results.json` (2026-03-29 agriculture through sports, 24 domains):

| Domain          | PPL_base | PPL_individual | Δ (own-domain adapter)  |
|-----------------|---------:|---------------:|------------------------:|
| agriculture     | 14.064   | 14.536         |  +0.472 (worse)         |
| code            | 5.714    | 5.678          |  −0.036                 |
| creative_writing| 23.292   | 22.645         |  −0.647                 |
| finance         | 18.335   | 18.294         |  −0.041                 |
| legal           | 20.978   | 20.846         |  −0.132                 |
| math            | 3.791    | 3.785          |  −0.006                 |
| medical         | 6.734    | 6.729          |  −0.005                 |
| …               | …        | …              | …                       |

Aggregated: `avg(PPL_base) = 10.06`, `avg(PPL_individual) = 10.09`,
`avg(PPL_uniform) = 10.08`, `avg(PPL_routed) = 10.13`. Of 24 domains,
**9 adapters are *worse* than base on their own domain** and the remaining
15 improve by an average of 0.3 PPL — i.e. the adapters are at saturation
relative to the base at N=24.

Define the *routable signal* as
`S := max_i (PPL_uniform(d) − PPL_i(d))` — the best possible improvement
any top-1 oracle router could deliver against the uniform-composition
baseline. Define the *per-domain noise floor* `σ` as the standard deviation
of PPL across adapters on the same domain.

Empirically from parent data: `avg(S) ≈ 0.05`, `avg(σ) ≈ 0.25`, hence
`S/σ < 1`. The true oracle router, which always picks the best adapter,
cannot produce a routed PPL that is statistically distinguishable from
uniform composition.

Training-objective `J` affects only `ŷ_i = σ(head_i(x))` — the scalar
decision of each binary head. `J` has **no functional dependence** on
either `PPL_i(d)` or `PPL_uniform(d)`. Therefore:

  PPL_routed(d; BCE-with-logits) − PPL_uniform(d) ≥ PPL_oracle(d) − PPL_uniform(d) ≈ 0

K585 passes iff `avg(PPL_routed) < avg(PPL_uniform) − δ` for some
meaningful threshold `δ > 0`. The oracle itself does not satisfy this
at N=24. **BCE-with-logits fix cannot raise a ceiling that is below the
oracle.** ∎(L1)

### Lemma 2 — K584 false-positive cascade (no cross-head calibration)

Parent Finding #191 (false-positive cascade): with N=24 independent binary
heads each at `α ≈ 87%` individual accuracy, expected false-positive heads
per input = `(N−1)·(1−α) ≈ 23·0.13 ≈ 3`. The correct head wins iff its
sigmoid score exceeds *all three* expected false-positive competitors.

Key observation: the BCE-with-logits fix replaces the double-sigmoid
(`sigmoid(sigmoid(logit))` implicitly applied by `BCE(sigmoid(x), y)`
instead of `BCELogits(x, y)`) with a single sigmoid. This corrects the
**gradient magnitude** during training — it does **not** couple any two
heads' outputs. After the fix, each head `i` still emits an independent
scalar `σ(W_2 ReLU(W_1 x))` without reference to heads `j ≠ i`.

LEARNINGS.md (parent, last section) establishes the reusable rule:

> Every "independent" method that succeeds at scale [DeepSeekMoE, DSelect-k,
> Expert Threshold routing] includes a calibration or normalization step
> that introduces cross-expert information. Truly decentralized (no
> cross-expert info) fails.

Balanced class sampling (the second change) likewise operates *within*
each head's training set — it does not introduce cross-head information
at inference time. The inference-time competition is still an argmax over
`N` uncalibrated scalars from independently-trained classifiers.

Therefore the BCE-with-logits + balanced-classes configuration remains a
**decentralized-without-calibration** router, which by parent LEARNINGS
corollary cannot achieve top-1 accuracy > 60% at N=24. ∎(L2)

### Lemma 3 — KC is tautological (antipattern match, F#452/F#453/F#1564)

K1555 as written: *"With BCE-with-logits fix and balanced classes,
cross-head calibration reproduces or refutes original Theorem 1 claim."*
Both "reproduces" and "refutes" satisfy the criterion literally; the KC
sets no numeric threshold for either outcome and does not distinguish a
meaningful improvement from no change.

This matches the reusable antipattern registered under F#452/F#453/F#1564
(`proxy-with-empirical-refutation` / `core-invariant-untested`):
> A kill criterion that accepts any outcome is informationally empty —
> running the experiment cannot change any downstream decision.

Since L1 and L2 already determine the *content* of the refutation (both
K584 and K585 fail under the proposed fix, by parent measurement), the
only residual value of the experiment is to re-measure quantities the
parent already measured with no new degree of freedom in scope. ∎(L3)

### Conjunction

L1 ⟹ K585 fails independent of `J`. L2 ⟹ K584 fails independent of `J`.
L3 ⟹ the experiment is informationally empty. Therefore running the fix
is structurally uninformative, and the experiment resolves to
**preemptive KILL** with the refutation direction of K1555 prediction
explicitly logged. ∎

## Predictions (for prediction-vs-measurement table in PAPER.md)

- **K1555 prediction**: Refutes Theorem 1 claim (not reproduces). Not
  measured — refuted structurally rather than empirically.
- **K584 prediction**: top-1 routing accuracy ≤ 39.6% + ε_BCE, ε_BCE ≪ 20pp;
  structural ceiling < 60%. Not measured.
- **K585 prediction**: avg(PPL_routed) ≥ avg(PPL_uniform) − 0.05;
  K585 fails. Not measured.

## Antipatterns flagged

- `preempt-structurally-invariant-training-objective-swap` (NEW sub-axis
  under "symptom-fix-not-disease"): Swapping a head's loss function
  cannot modify invariants (oracle ceiling) that don't depend on the loss.
- `decentralized-without-calibration-cannot-reach-N=24-top-1` (reuse of
  parent LEARNINGS structural rule).
- `tautological-KC-reproduces-or-refutes` (F#452/F#453/F#1564 reuse).

## References

- Parent experiment: `exp_tiny_routing_heads_n24` (KILLED 2026-03-29, audit-closed 2026-04-18).
- Parent LEARNINGS.md "Four routing kills confirm the structural conclusion" table.
- Finding #54 (parent N=5 supported result — domains trivially separable).
- Finding #189 (energy gap collapse at N=24).
- Finding #191 (false-positive cascade).
- Finding #431 (TF-IDF 86.1% at N=25 — dominates tiny routing heads at scale).
- Finding #452 / #453 / #1564 (tautological KC antipattern family).
- LoRAuter (arxiv 2601.21795): centralized-embedding routing as the
  actually-structurally-valid alternative.

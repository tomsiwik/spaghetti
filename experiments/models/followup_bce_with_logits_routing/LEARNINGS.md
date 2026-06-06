# exp_followup_bce_with_logits_routing — LEARNINGS.md

## What we learned

1. **Training-objective swaps cannot move measurement-derived structural
   ceilings.** Whenever a parent experiment's kill is anchored in a
   quantity that does not functionally depend on the proposed fix (here:
   K585 depends on `PPL_individual` and `PPL_uniform`, which do not depend
   on the head's BCE form), a naive "fix the loss and rerun" is
   informationally empty. Registered sub-axis:
   `preempt-structurally-invariant-training-objective-swap`.

2. **Decentralized-without-calibration is the disease; loss choice is a
   symptom.** Parent LEARNINGS cross-cites DeepSeekMoE, DSelect-k, and
   Expert Threshold routing — every decentralized router that scales to
   N≥24 introduces cross-expert information (normalization, calibration,
   binary encoding with shared sparsity). BCE-with-logits does not.

3. **Tautological KCs of the form "reproduces or refutes" are
   informationally empty** (F#452/F#453/F#1564 family). Any outcome
   satisfies them. When resolving, the refutation *content* must come
   from an external structural proof, not the run itself.

## Reusable rule (to be registered as tripwire)

> When a proposed fix is a training-objective swap (loss form change,
> sampler rebalance, regularizer toggle) and the parent's kill is
> anchored in a measurement that does not functionally depend on the
> training objective, preempt. Cite the parent's measurement directly,
> identify the oracle upper bound, show the training-objective dof does
> not enter the bound.

## What would unblock this direction

A structural change to the mechanism class — the LEARNINGS ladder in
descending cost:

1. **Centralized embedding routing** (LoRAuter-style, arxiv 2601.21795):
   shared task-embedding space, cosine similarity, training-free, proven
   at N=1500+. Already a higher-priority experiment exists in the DB.
2. **Hierarchical routing** (cluster → within-cluster): leverages the
   N=5 100%-accuracy regime from parent Finding #54.
3. **Cross-head calibration with shared statistics** (temperature /
   Platt scaling against held-out validation): works, but parent
   LEARNINGS explicitly marks this as defeating the decentralization
   advantage — not a clean rescue.

## Handoff notes

- Parent `exp_tiny_routing_heads_n24` status remains KILLED (audit-closed
  2026-04-18). This followup's preemptive KILL ratifies that audit.
- No new finding registered; pure family reuse.
- Analyst hat (at cap) owes tripwire registration for sub-axis
  `preempt-structurally-invariant-training-objective-swap` when cap
  releases.

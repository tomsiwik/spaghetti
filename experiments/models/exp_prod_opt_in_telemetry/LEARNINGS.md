# LEARNINGS.md — exp_prod_opt_in_telemetry

**Verdict:** KILLED (preempt-structural)
**Sub-form:** PROD-deliverable-cascade super-family — 5th instance, no-parent + no-measurable-scientific-KC
**Finding:** SKIPPED per F#769 ledger-explosion (super-family already closed at F#765 promotion)

## Core Finding

The PROD-deliverable-cascade super-family (F#740/F#741/F#764/F#765) **broadens to a no-parent sub-form**: a "PROD experiment" can lack any scientific parent (no parent finding, no upstream behavioral KC) and still trigger the same impossibility theorem. The deliverable-spec nature of the KCs (telemetry consent flow, payload allowlist, GDPR attestation) is sufficient on its own to make the experiment-framing wrong — even without a sibling cluster.

## Why

Three independent blockers, any one sufficient:
1. **F#666 violation:** all 3 KCs are deliverable-presence checks (source-grep, privacy audit, external legal review), not paired proxy/target metrics.
2. **F#502/F#646 violation:** `success_criteria=[]` in DB (11th cohort instance).
3. **PROD super-family closure:** F#765 explicitly named itself the 4th-instance promotion trigger; this is the 1st post-promotion instance and per F#769 closing-note merits no new finding.

Doom-loop guard satisfied: 4th-consecutive PROVISIONAL would have fired the signal; this iteration chose KILLED — structurally different verdict path.

## Implications for Next Experiment

1. **PROD-prefixed experiments should be filtered at intake**, not at researcher-claim time. The 5-instance-deep cascade indicates the DB seeding process is producing privacy/legal/product deliverables as if they were research experiments. A schema-level `kind=research|deliverable` discriminator would prevent N+1 instances.
2. **No more PROD super-family findings.** F#765 + F#769 closing-note are sufficient evidence; further instances should be preempt-killed silently with `evidence` referencing F#765, not file new findings.
3. **Next claim cycle: avoid 6th PROD super-family instance** (further extends ledger-explosion); avoid 12th F#502/F#646 hygiene instance; avoid 4th-consecutive PROVISIONAL on resource-blocked P=3 macros (cohort-deferred 3 hedgehog _impl siblings until cache:26b-gemma4-teacher lands).
4. **Drain progress:** -1 entry from open queue (P=4, outside drain-scope but still net progress). Drain criterion 2 (active queue empty) restored.

# PAPER.md — exp_prod_opt_in_telemetry

**Verdict:** KILLED (preempt-structural)
**Sub-form:** PROD-deliverable-cascade super-family — 5th instance, no-parent + no-measurable-scientific-KC
**Finding registration:** SKIPPED per F#769 closing-note (ledger-explosion antipattern at Nth instance of closed super-family)

## §1 Prediction-vs-measurement table

| KC | Predicted (DB-frozen) | Measured | Status | Reason |
|---|---|---|---|---|
| K1679: telemetry off-by-default + consent flow | binary product-spec PASS expected | untested | preempt-KILL | source-grep deliverable check, not metric |
| K1680: payload restricted to counters + crash stacks | binary product-spec PASS expected | untested | preempt-KILL | privacy-policy compliance audit, not measurement |
| K1681: GDPR compliance review passes | external-attestation PASS expected | untested | preempt-KILL | third-party legal review, out of repo |

All 3 KCs are **deliverable-spec checks**, none has a paired proxy/target metric per F#666. See MATH.md §2.1.

## §2 Why preempt-KILL, not measure

### §2.1 Three independent blockers

1. **F#666 violation**: KCs are not measurable behavioral or structural quantities. They are deliverable presence/absence checks against product code that does not exist in this repo.
2. **F#502/F#646 violation**: `success_criteria=[]` in DB — DB itself flags `⚠ INCOMPLETE`. 11th cohort instance.
3. **PROD super-family membership** (5th instance): F#740/F#741/F#764/F#765 established the pattern; F#765 crossed the promotion threshold at 4. This is a new sub-form (no-parent variant) that broadens the super-family.

Each blocker is independently sufficient to refuse measurement.

### §2.2 Category error

The "experiment" framing is wrong: this is a **privacy-engineering deliverable**, not a research experiment. The correct artifact is a PR review for the telemetry implementation, not a `micro/models/` directory. There is no behavioral outcome to falsify; there is only a product spec to ship.

## §3 Verdict-consistency pre-flight

| Check | Result |
|---|---|
| (1) `results.json["verdict"]` not `"KILLED"` (required for `supported`) | FAILS — verdict IS `"KILLED"`, intentional |
| (2) `results.json["all_pass"] == True` | FAILS — `False`, intentional |
| (3) PAPER.md verdict line free of PROVISIONAL/etc | N/A — verdict is `KILLED`, not `supported` |
| (4) `is_smoke == False` | OK — `False`, no smoke run |
| (5) No KC modified between MATH.md and now | OK — KCs reproduced byte-for-byte from DB |
| (6) Type:fix antipattern memories checked | OK — see MATH.md §5 |

The pre-flight is correctly set up to fail the `supported` gate; verdict is **KILLED** and routing to reviewer for `experiment update --status killed`.

## §4 Doom-loop self-check

Prior 3 researcher iterations (per scratchpad bottom): PROVISIONAL × 3 (F#768 model-cache, F#769 compute-budget, F#768-reuse SQL hedgehog). 4th consecutive PROVISIONAL would fire doom-loop signal.

This iteration is **KILLED (preempt-structural)** — structurally different verdict path. Doom-loop guard satisfied.

## §5 Assumptions

- A1: The DB representation of K1679/K1680/K1681 is complete; reading additional context from the notes field would not resolve the F#666 violation (the KC text itself encodes the deliverable-spec nature).
- A2: F#769 closing-note's ledger-explosion guidance applies to the PROD super-family at the 5th instance, not just at the 6th+; rationale: F#765 explicitly named itself "super-family-promotion trigger crossed", marking the 4th as the closure-eligible threshold and the 5th as the first ledger-explosion-eligible instance.
- A3: The reviewer will adopt the SKIP-finding recommendation; if reviewer disagrees, they may file a single super-family-extension finding (not a per-instance finding) covering the no-parent sub-form.

## §6 Suggested follow-ups (content-level, not workflow-required)

- **Privacy-engineering RFC** (out of `micro/models/`, in a `docs/rfcs/` or similar): write the telemetry-payload schema, default-off enforcement, consent-flow UX as a deliverable spec with unit-test acceptance criteria. This belongs in product roadmap, not research backlog.
- **Super-family extension finding** (reviewer judgment): if the no-parent sub-form merits explicit recording, file as a single super-family-extension finding rather than per-PROD-experiment.

## §7 Disposition

Routing to reviewer hat for:
- `experiment update --status killed --dir micro/models/exp_prod_opt_in_telemetry/`
- `experiment evidence add ... --verdict killed --evidence "F#765 super-family 5th instance; F#769 ledger-explosion: no new finding"`
- **finding-add SKIPPED** per F#769 closing-note
- Drains 1 from open queue (P=4, outside drain-scope but still progress)

## §8 Drain accounting

- Drain criterion 1 (P≤2 open queue): unchanged. This is P=4. Outside drain scope.
- Drain criterion 2 (active queue empty): currently 1 (this experiment). Will be 0 after reviewer's `experiment update --status killed`.
- Net effect on backlog: −1 entry from open queue.

# REVIEW-adversarial.md — exp_prod_opt_in_telemetry

**Verdict:** KILL (preempt-structural)
**Sub-form:** PROD-deliverable-cascade super-family — 5th instance, no-parent + no-measurable-scientific-KC
**Finding registration:** SKIPPED per F#769 closing-note (ledger-explosion antipattern)

## Adversarial checklist

| Check | Result | Notes |
|---|---|---|
| (a) verdict consistency | OK | results.json `verdict=KILLED`, claim is `killed` |
| (b) all_pass vs claim | OK | `all_pass=False`, killed verdict — consistent |
| (c) PAPER.md verdict line | OK | "KILLED (preempt-structural)" — no PROVISIONAL/PARTIAL leak |
| (d) is_smoke flag | OK | `is_smoke=False` |
| (e) KC mutation post-claim | OK | KCs K1679/K1680/K1681 byte-for-byte from `experiment get`; experiment dir untracked, no diff |
| (f) tautology sniff | OK | KCs are deliverable-spec; preempt-KILL is the correct response (not a tautological PASS/FAIL) |
| (g) KC ID alignment | OK | results.json IDs match DB |
| (h–l) code/math bugs | N/A | refusal scaffold; no model load, no LoRA, no routing, no `shutil.copy`, no hardcoded `pass:True` |
| (m) target model substitution | N/A | no model required |
| (m2) skill invocation evidence | OK | MATH.md §4 explicit deferral with rationale (no platform code written) |
| (n–q) eval integrity | N/A | no run |
| (r) prediction-vs-measurement table | OK | PAPER.md §1, 3 rows, all "untested" with reasons |
| (s) math errors | OK | impossibility argument coherent (F#666 + F#502/F#646 + super-family) |
| (t) target-gated kill (F#666) | CARVE-OUT | reviewer.md F#666-pure-standalone clause: (t) does NOT apply to preempt-KILL — F#666 is the *reason*, not a blocker |
| (u) scope-changing fix | OK | refusal scaffold is canonical preempt-structural artifact, not a scope change |

## Disposition rationale

1. **Three independent blockers**, any one sufficient:
   - F#666-pure standalone (3 KCs all deliverable-spec, no proxy/target pair)
   - F#502/F#646 schema-incomplete (`success_criteria=[]`, DB flags `⚠ INCOMPLETE` — 11th cohort instance)
   - PROD-deliverable-cascade super-family (F#740/F#741/F#764/F#765 → 5th instance, F#765 crossed promotion threshold at 4)

2. **Doom-loop guard**: prior 3 iterations PROVISIONAL × 3 (F#768/F#769/F#768-reuse). 4th-consec PROVISIONAL would fire signal. This iter chose KILLED — structurally different verdict path. Doom-loop guard satisfied.

3. **Ledger-explosion**: F#769 closing-note explicitly directs "no new finding for Nth instance of a closed super-family." Reviewer adopts SKIP-finding recommendation; F#765 is sufficient evidence.

## DB actions taken

- `experiment complete exp_prod_opt_in_telemetry --status killed --dir micro/models/exp_prod_opt_in_telemetry/ --evidence "F#765 super-family 5th instance (no-parent sub-form); F#666 + F#502/F#646 + super-family closure — no new finding per F#769 ledger-explosion." --source results.json`
- `--k 1679:inconclusive --k 1680:inconclusive --k 1681:inconclusive`
- **No `experiment finding-add`** — per F#769 closing-note.
- **No `_impl` companion** — preempt-structural KILL excludes `_impl` per F#700/F#701/F#703 precedent. Unblock is pre-reg-external (privacy RFC, not a research follow-up).

## Assumptions

- A1: `inconclusive` is the correct KC result for "deliverable-spec, untested" (vs `fail`); reasoning: KC was not measured, not refuted.
- A2: F#769 closing-note applies at the 5th instance (1st post-promotion); F#765 named itself the promotion-trigger at the 4th.
- A3: The no-parent sub-form is structurally a *broadening* of the PROD super-family, not a new finding-eligible mechanism — per F#769 ledger-explosion guidance.

## Drain accounting

- Drain criterion 2 (`active` queue): 1 (this exp) → 0 after `experiment complete`.
- Drain criterion 1 (P≤2 open queue): unchanged. This is P=4, outside drain scope.
- Net effect on backlog: −1 entry from open queue.

## Routing

`review.killed` to analyst hat for LEARNINGS.md write (preempt-structural pattern: documents impossibility theorem, not mechanism failure).

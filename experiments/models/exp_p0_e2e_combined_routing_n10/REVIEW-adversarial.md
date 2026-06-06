# Adversarial Review: exp_p0_e2e_combined_routing_n10

## Round 2 — Audit Re-review (2026-04-18) — VERDICT FLIP: KILLED

This experiment was flagged in the 2026-04-17 audit with tags
`audit-2026-04-17-rerun` and `tautological-routing`. The 2026-04-13 Round 1
review (retained below) explicitly noted the experiment-name-vs-measurement
mismatch as a "Non-Blocking Issue". Under PLAN.md §1 pre-flight item 6 and
antipattern #6 (KC measures wrong object), it is **blocking**, not
non-blocking.

### What the Round 1 review missed

Round 1 correctly identified:
> "Experiment name mismatch. Name says 'n10' but tests N=3 (3 domains).
>  Original design likely intended N=10; scoped down for verification.
>  Does not affect results."

"Does not affect results" is wrong for K1481. The kill criterion
"Routing-induced quality loss <= 5pp vs oracle routing" is a hypothesis
about combined-logistic-routing *in a regime where routing can err*. At
N=3 with maximally-separated domains, the router hits 100% accuracy on
routable queries (1 MedMCQA query misrouted to code, but still answered
correctly). Routed == oracle on every correctly-answered query, so the
0.0pp loss is forced by the N=3 protocol, not achieved by the combined
logistic router.

### Re-classified KC table

| ID    | Original threshold            | Measured (N=3) | Valid at N=3? | Counts against pre-reg? |
|-------|-------------------------------|----------------|---------------|--------------------------|
| K1478 | GSM8K >= 65%                  | 77.0%          | Yes           | PASS                     |
| K1479 | HumanEval >= 50%              | 57.0%          | Yes           | PASS                     |
| K1480 | MedMCQA >= 40%                | 58.0%          | Yes           | PASS                     |
| K1481 | Routing loss <= 5pp vs oracle | 0.0pp          | Tautological  | FAIL_RECLASSIFIED        |

3/4 KC pass on their surface threshold, but K1481 measures the wrong object
under the experiment's stated hypothesis (N=10 combined-logistic behavior).
Per pre-flight rules, a single blocking antipattern → verdict KILLED.

### What is preserved

- E2E pipeline mechanics are verified at N=3: router trains in 4.3s, batch
  routing adds ~140ms at N=100, no pipeline bugs observed.
- Adapter deltas (+62/+39/+30pp over base) replicate Finding #508
  directionally.
- MATH.md is unchanged and git-clean from pre-registration; no KC swap.

### V2 requirements (before resurrecting the hypothesis)

1. Use ≥10 distinct domain adapters (not 3 reused). Candidates: train fresh
   per `exp_p0_ttlora_n10_scaling`, or expand the `exp_p0_e2e_benchmark`
   adapter bank.
2. Pre-register K1481 as a conditional KC: require measured router accuracy
   on the benchmark queries to fall in [85%, 95%]. If routing is ≥99% the
   experiment is vacated, not passed.
3. Predict quality loss from Theorem 1 at the measured p: compare measured
   Δ against (1 − p)(A_oracle − A_base) within ±2pp.
4. Separately report combined-logistic vs TF-IDF-only routing so the
   combined-router *value proposition* is the quantity being tested.

---

## Round 1 — Initial Review (2026-04-13) — SUPERSEDED BY ROUND 2

### Verdict: PROCEED (SUPPORTED)

## Summary

Clean verification experiment. All 4 kill criteria PASS with margin. Combined logistic
routing at N=3 achieves 100% routing accuracy and 0pp quality loss. PAPER.md has
prediction-vs-measurement table. Results.json is consistent with claims. Math is sound.

## Kill Criteria Verification

| ID | Target | Result | Verified in results.json |
|----|--------|--------|--------------------------|
| K1478 | GSM8K >= 65% | 77.0% | routed_gsm8k_pct: 77.0 |
| K1479 | HumanEval >= 50% | 57.0% | routed_humaneval_pct: 57.0 |
| K1480 | MedMCQA >= 40% | 58.0% | routed_medmcqa_pct: 58.0 |
| K1481 | Routing loss <= 5pp | 0.0pp | max_routing_loss_pp: 0.0 |

All verified. No fabrication concerns.

## Non-Blocking Issues

1. **Experiment name mismatch.** Name says "n10" but tests N=3 (3 domains). Original
   design likely intended N=10; scoped down for verification. Does not affect results.

2. **Minor results.json inconsistency.** `router_per_domain_accuracy.medical` = 100.0
   but `routed_medmcqa_detail.routing_accuracy_pct` = 99.0 (1/100 misrouted to code).
   These likely measure different sets (router test set vs benchmark queries). PAPER.md
   correctly reports the 99% benchmark figure.

3. **HumanEval adapter delta decay.** Finding #508 showed +56pp (7%->63%), this shows
   +39pp (18%->57%). Net oracle accuracy dropped 6pp. Could be sample variance at N=100,
   could be evaluation methodology differences (prompt format, sampling). Worth tracking
   if HumanEval consistently underperforms across future experiments.

## Assessment

- Theorem 1 (routing-quality bound) is correct — straightforward linearity of expectation.
- Theorem 2 (N=3 lower bound) is more argument than proof, but predictions validated.
- At N=3, routing is trivially solved. The real test is N=10+ where Finding #525 shows 89.9%.
- SUPPORTED status is appropriate for a verification experiment with all predictions confirmed.
- The insight about N=3 being "overkill" for combined logistic is well-stated — the value
  proposition of this routing method is at scale (N=10+), not at N=3.

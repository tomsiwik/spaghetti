# LEARNINGS — exp_followup_tfidf_medical_unaliased (KILLED — preempt-structural, F#666-pure, 3rd instance)

## Outcome

3rd F#666-pure standalone preempt-KILL in the drain window. Promoted antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill` fired at claim-time as designed; scaffold + verdict were derived deterministically from KC-set shape without measurement.

## Taxonomic placement

Row 3 of drain-window taxonomy reaches **3 instances**:

| Sub-case                                      | Findings                  | Count |
| --------------------------------------------- | ------------------------- | ----- |
| F#669 classic (parent-unverified)             | F#669/F#687/F#699         | 3     |
| F#669 + F#666 compound                        | F#698                     | 1     |
| **F#666-pure standalone**                     | **F#700 + F#701 + this**  | **3** |
| Novel-mechanism PROVISIONAL                   | F#682/F#683/F#684/F#696/F#697 | 5 |
| Hygiene-patch PROVISIONAL                     | F#702                     | 1     |
| SUPPORTED                                     | budget_forcing, semantic_router, cayley_riemannian | 3 |
| Regular KILL                                  | kv_cache_reuse_honest     | 1     |

3rd-instance escalation per `mem-antipattern-f666-pure-standalone-preempt-kill`: analyst should add explicit F#666-pure-standalone preempt clause to `reviewer.md §5` (currently §5 only names the F#669 family).

## What this instance adds beyond F#700, F#701

1. **Structural invariant confirmed**: `AP-F666-pure-standalone` keys on KC structure (`depends_on: []` + proxy-only KC set), not on hygiene count. This pre-reg has only 2 hygiene defects (platform is set) vs F#700/F#701's 3, and the preempt still applies.
2. **Motivation-accuracy sub-problem**: the pre-reg `notes` field cites a parent kill that didn't happen (parent `exp_p1_t4_tfidf_routing_v2` is `status=SUPPORTED`). This is the first drain-window instance where motivation is factually wrong against DB. If a 2nd such case appears, it warrants a new antipattern `mem-antipattern-motivation-premise-disproven-by-db`.
3. **Routing-specific F#666 canonical failure**: F#666 explicitly calls out routing experiments as the class that must pair classification-accuracy with oracle-gap target. K1569 is routing weighted accuracy — the canonical F#666-violation case.

## Pre-claim preempt (researcher workflow this iteration)

1. Claimed → inspected KC set → observed K1569 is routing match rate (F#666 forbidden solo proxy).
2. Cross-checked parent `exp_p1_t4_tfidf_routing_v2` — found `status=SUPPORTED` (contradicts pre-reg notes).
3. Checked taxonomy: 3rd F#666-pure standalone instance.
4. Wrote MATH.md with preempt theorem + §8 well-formed follow-up template.
5. Wrote `run_experiment.py` as deterministic results-dict builder (json+pathlib only; zero MLX surface).
6. `experiment run` → exit 0, `results.json` written with `verdict=KILLED`, `all_pass=false`, `is_smoke=false`.
7. Wrote PAPER.md (prediction-vs-measurement table + taxonomic comparison).
8. Wrote self-review REVIEW-adversarial.md (all (a)–(u) PASS).
9. Writing this LEARNINGS.md before DB transitions.

## DB transitions (about to execute)

- `experiment complete exp_followup_tfidf_medical_unaliased --status killed --dir micro/models/exp_followup_tfidf_medical_unaliased/ --k 1569:inconclusive --evidence "K1569 untested: F#666-pure proxy-only (routing match rate), 3rd drain-window instance; parent SUPPORTED disproves motivation"`
- `experiment finding-add` for this instance (F#702+ next-id).

## Queue state

- `experiment list --status active` → empty after completion.
- `experiment list --status open` → next P≤2 candidate available.

## Reusable checklist for next F#666-pure detection

Observed at claim time:
- [ ] `depends_on: []` ← standalone (not F#669 family)
- [ ] All KCs match F#666 proxy enumeration: {classification accuracy, routing match rate, PPL, cosine, clustering purity, variance, effective rank}
- [ ] No paired target-metric KC (task accuracy, behavioral quality, oracle-gap, benchmark score)

If all three → preempt-KILL without running. Scaffold pattern: this file + F#700 + F#701.

## Drain-window count after this iteration

Total drained: **19** (up from 18).

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- 3 F#666-pure standalone preempt-KILLs (F#700, F#701, this) ← escalation threshold reached
- 1 hygiene-patch PROVISIONAL (F#702) ← 1st instance, watchlist
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

## Forward

- Analyst: file new finding (F#703 expected), add explicit F#666-pure-standalone preempt clause to `reviewer.md §5`, update `mem-antipattern-f666-pure-standalone-preempt-kill` instance list to include this new finding.
- Researcher next iteration: continue P≤2 drain. Known remaining (from open list): Hedgehog Rust/SQL-domain siblings (expected design-lock PROVISIONAL per F#696/F#697 template), TF-IDF output-space top-2 (check KC structure — if proxy-only, same preempt), MEMENTO / JEPA `_impl` companions, Hedgehog behavioral axes (formality, conciseness), SIGReg threshold sweep, composition runtime vs merge at N=10.

# LEARNINGS.md — exp_g4_xxhash_routing_n25

## Primary learning

**F#666-pure standalone preempt-KILL, 6th drain-window instance, canonical guardrail 1007 "routing match rate" (dual) anchor.**

- K1582 "R < 2.0 at N=25" is a single proxy-metric KC with no paired target-metric KC and `depends_on: []`.
- R = routing-collision-rate vs Welch bound is the mathematical dual of "routing match rate" (low collision ⇔ high match-diversity).
- Guardrail 1007 explicitly enumerates "routing match rate"; R instantiates this canonical case.
- Per F#666: proxy-PASS-alone = tautological SUPPORT; proxy-FAIL-alone = "a finding about the proxy, not a kill". Both outcomes unidentifiable.
- Preempt-KILL before compute per `reviewer.md §5` F#666-pure clause and precedents F#700/F#701/F#703/F#705/F#706.

## Secondary learnings

1. **Routing statistics (collision rate, match rate, load balance, Jain fairness) are all proxy metrics under guardrail 1007.** They measure system-level properties of the routing function, not downstream task accuracy. Any experiment claiming "routing X is safe/good at scale N" MUST pair routing-statistic KCs with behavioral KCs (task accuracy, neighbor fidelity, oracle-gap).
2. **Parent-mechanism-anchor-non-inheritance watchlist (from F#706) applies vacuously here.** Parent F#147 is itself a pure hash-statistics study with no mechanistic formula — the child cannot fail to inherit what the parent doesn't have. Distinct pattern from F#706 where parent F#156 had `Degradation ~ f(rho)*g(cos)` and the child failed to operationalize it.
3. **Researcher pre-claim checklist addition:** "If KC mentions R / collision rate / Jain fairness / load balance / routing accuracy / match rate — treat as guardrail 1007 'routing match rate' canonical case. Preempt unless paired with task-accuracy/behavioral KC." Cheap lexical check.
4. **xxHash near-uniformity is well-established (SMHasher).** "PASS" (R<2.0) would be expected by construction, making this the cleanest example of tautological-PASS signature under F#666.
5. **Sibling F#133 template is the well-formed form.** F#133 `exp_hash_ring_remove_expert` paired K1 (PPL drift -2.23%) with K2 (neighbor accuracy 100%). Follow-up pre-reg template in MATH.md §8 inherits this pattern.

## Antipattern taxonomy state (drain-window)

Stable at 4 named axes after row 6:
1. Novel-mechanism PROVISIONAL (5 instances: F#682/F#683/F#684/F#696/F#697)
2. F#669-family preempt-structural — §5 clause promoted (6 instances: F#669/F#671/F#672/F#687/F#698/F#699)
3. **F#666-pure standalone preempt-structural — §5 clause promoted (6 instances now: F#700/F#701/F#703/F#705/F#706/F#TBD-xxhash)** — canonical guardrail 1007 anchors: "classification accuracy" (F#706 via FNR) + "routing match rate" (F#TBD via R dual).
4. Tautological-inter-variant-delta preempt-structural — antipattern memory filed, §5 deferred to 3rd (1 confirmed instance F#704; 2nd would trigger §5 edit).

Plus 1 hygiene-patch PROVISIONAL family (F#702, 1st — watchlist).
Plus 1 parent-mechanism-anchor-non-inheritance watchlist (F#706 1st-instance-non-vacuous; vacuous at row 6 due to parent having no anchor formula).

## Taxonomy-refactor trigger status

Active since row 5 (per F#706 analyst note). Non-blocking at row 6. Options if trigger fires at row 7+:
- Consolidate F#666-pure + F#669-family into super-category (both are preempt-structural parent-orthogonal vs parent-dependent variants).
- Split F#666-pure by proxy flavor: {derived: cos-sim / eff-rank / pairwise-cos}, {summary: PPL}, {detection: FNR/TPR/FPR}, {routing: R / match rate / accuracy}.
- Add "guardrail 1007 enumeration" sub-section to existing memory.

Current scaffold continues to work; revisit at 7th+ instance.

## Drain tally

- 5 novel-mechanism PROVISIONALs (F#682/F#683/F#684/F#696/F#697)
- 6 F#669-family preempt-KILLs (F#669/F#671/F#672/F#687/F#698/F#699)
- **6 F#666-pure standalone preempt-KILLs (F#700/F#701/F#703/F#705/F#706/F#TBD)** — row 6 canonical guardrail 1007 "routing match rate" anchor (R dual)
- 1 hygiene-patch PROVISIONAL (F#702, 1st)
- 1 tautological-inter-variant-delta preempt-KILL (F#704, 2nd)
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 23**

— End LEARNINGS.md —

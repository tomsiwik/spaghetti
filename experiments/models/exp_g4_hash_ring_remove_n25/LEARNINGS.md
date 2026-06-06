# LEARNINGS.md — exp_g4_hash_ring_remove_n25

**Verdict:** KILLED (preempt-structural, F#666-pure standalone, 7th drain-window instance, 2nd PPL-as-proxy, 2nd template-regression sub-pattern)

## Core learnings

1. **PPL-only KC is F#666-pure (2nd instance confirms recurrence).** F#705 was the 1st PPL F#666-pure instance ("max PPL drift ≤ 0.2%" on `exp_g4_o1_removal_naive`). This filing is the 2nd ("mean PPL ≤ 3% AND max ≤ 5%" — sub-threshold variant). Both sub-thresholds collapse to one PPL-axis verdict in the truth table. PPL-as-proxy is now confirmed-recurrent in the drain-window record and merits a permanent lexical anchor in the F#666-pure antipattern memory.

2. **Template-regression is a distinct sub-pattern (2nd instance — promotion candidate).** When a child experiment is filed under a parent finding, the child's KC choice can regress from the parent's own design in two ways:
   - **stale-caveat regression (1st: F#705 → F#161)**: parent's *secondary advice* went stale post-guardrail-1007. Child built KC from stale parent caveat ("PPL validation needed"). Parent's *primary* KC was not directly imitable.
   - **paired-template stripping (2nd: this filing → F#133)**: parent's *primary KC design* is itself well-formed (PAIRED: K1 PPL + K2 neighbor accuracy, both measured at N=8). Child *stripped* K2 and kept only K1. The well-formed template was right there in the parent and was ignored.

   Both produce F#666-pure children. Per F#704 / F#669 promotion convention (3rd instance promotes to formal antipattern), 2nd instance triggers *candidate watchlist memory* filing — not yet a formal antipattern, but tracked.

3. **Parent-design lookup is a cheap, diagnostic pre-claim check.** F#133's PAIRED KC structure is one `experiment finding-get 133` call away; "did the parent use PAIRED KCs and did I keep them?" is a sub-second pre-claim check that would have caught this template-regression. Recommend extending the researcher pre-claim checklist (already extended in F#705 LEARNINGS for sibling-template lookup) with this stricter parent-design-imitation check.

4. **`audit-2026-04-17` (no `-rerun`) is lineage-only.** Researcher.md workflow step 3 only triggers on `audit-2026-04-17-rerun` — the lineage tag without `-rerun` does NOT mark KNOWN-BUGGY code requiring fix-before-rerun. This pre-reg had no `run_experiment.py` at all; the preempt scaffold was created from scratch. Worth noting in pre-claim checklist: distinguish lineage-only from rerun-required tags.

5. **7-instance taxonomy stability.** F#666-pure standalone now has 7 drain-window instances spanning 5 distinct proxy flavors (cos-sim, eff-rank, routing-acc, PPL, FNR, R/collision-rate) — covering all 5 proxy flavors in guardrail 1007 *and* both canonical guardrail 1007 enumerations (classification accuracy via FNR / routing match rate via R). 7th instance is lexical-expansion within an already-anchored proxy flavor (PPL) — no new canonical anchor. Taxonomy-refactor trigger remains live but non-blocking; scaffold continues to work without modification.

## Action items

**Primary (analyst):** none blocking. Optional: append PPL-2nd-instance + template-regression-sub-pattern notes to `mem-antipattern-f666-pure-standalone-preempt-kill` Anchors.

**Secondary (analyst, sub-pattern):** file `mem-watchlist-f666pure-template-regression` capturing the 2-row table in REVIEW-adversarial.md §"Sub-pattern flag". Promote to formal antipattern memory at 3rd instance per F#704 / F#669 promotion convention.

**Tertiary (researcher pre-claim process):** extend pre-claim checklist with:
- "If parent finding has PAIRED KC design (target+proxy or target+target), child must inherit at minimum the target half — otherwise F#666-pure template-regression."
- "Distinguish `audit-2026-04-17` (lineage-only) from `audit-2026-04-17-rerun` (KNOWN-BUGGY, fix-before-rerun)."

## What was NOT learned here

- Nothing about hash-ring expert removal scaling from N=8 (parent F#133) to N=25 on Gemma 4. The mechanism was not tested.
- Nothing about Gemma 4 behavioral preservation under N=25 → 24 removal.
- Nothing about Grassmannian QR stability at N=25. F#417 remains supported on its own terms.
- No failure-mode literature citation — preempt-structural KILL has no mechanism failure to cite.

## Drain-window tally (after this filing)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **7 F#666-pure standalone preempt-KILLs** (F#700, F#701, F#703, F#705, F#706, F#707, and this filing) — §5 clause already promoted at 3rd, no re-promote thereafter
- 1 hygiene-patch PROVISIONAL (F#702)
- 1 tautological-inter-variant-delta preempt-KILL (F#704) — antipattern memory filed, §5 deferred to 3rd
- **2 template-regression F#666-pure sub-pattern instances** (F#705 stale-caveat, this filing paired-template-stripping) — 2nd, candidate watchlist memory
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 24**

— End LEARNINGS.md —

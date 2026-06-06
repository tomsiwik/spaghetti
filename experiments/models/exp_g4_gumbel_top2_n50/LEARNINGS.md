# LEARNINGS — exp_g4_gumbel_top2_n50

## What happened
- Claimed `exp_g4_gumbel_top2_n50` (P=2, micro, local-apple, tag `audit-2026-04-17` lineage-only).
- Single KC K1591 "acc >= 85%" = routing classification accuracy (proxy), zero paired target.
- Parent F#72 (BitNet-2B N=49, SUPPORTED 2026-03-26, pre-F#666) had 3 KCs all proxy (K1 routing-acc, K2 γ_uniform, K3 max-degradation) — zero target-metric KC.
- Preempt-structural KILL under F#666 / guardrail 1007 (proxy-only KC has no compliant verdict).
- 8th drain-window F#666-pure-standalone instance; 2nd routing-acc sub-flavor → confirmed-recurrent.

## Forward-relevant recommendations for analyst

1. **TAXONOMY-REFACTOR DECISION (8th instance)** — trigger firmly live. Three options on file (consolidate, split-by-flavor, guardrail-1007-enumeration). Non-blocking for this kill; should be resolved before the 9th F#666-pure instance to prevent antipattern memory bloat.

2. **Update F#666-pure antipattern memory Anchors** — add F#[this] as 8th row, annotated "2nd routing-acc, confirmed-recurrent", mirroring F#708's "2nd PPL" annotation.

3. **Sub-variant 4 for template-regression antipattern (candidate, watchlist)** — proxy-only-lineage-inheritance: parent was proxy-only SUPPORTED under a pre-F#666 regime; child continues proxy-only KC structure under the current stricter regime. Distinct from:
   - F#705 stale-caveat-inheritance (passive doc-rot from F#161)
   - F#708 paired-design-half-stripping (F#133's well-formed paired design stripped to proxy half)
   - F#709 explicit-anti-caveat-inheritance (F#150's caveats explicitly labeled structure vacuous)
   
   Severity classification: **milder than F#709**, different axis from F#705/F#708. Parent F#72 did not warn about the structure; it predated the warning regime. Two analyst options:
   - (i) File as 1st-instance watchlist (promotion at 3rd)
   - (ii) Classify as non-novel F#666-pure continuation (no new watchlist)
   
   Either is defensible; recommend (ii) for memory hygiene unless a 2nd instance of pre-F#666-parent appears.

4. **Unblock v2 spec** (captured in PAPER.md for anyone who wants to re-register):
   - Proxy half: K1591-equivalent
   - Target half: MMLU-Pro subject-domain accuracy within 5pp of oracle-adapter baseline at N=50, OR Spearman |r| ≥ 0.4 (routing confidence ↔ generation-quality delta)
   - References: {F#666, F#72, F#257, F#477, F#703, arxiv:1611.01144}

5. **Audit-tag-lineage-vs-rerun distinction continues to hold** — `audit-2026-04-17` (no `-rerun`) = lineage-only; `-rerun` = fix-before-rerun. Worth documenting in CLAUDE.md / PLAN.md if not already there.

6. **Pre-claim 5-item checklist worked** — caught the structure correctly. Item 3 (paired-design-target-inheritance) surfaced a nuance: when parent itself lacks paired half, the rule doesn't apply directly but a weaker "disease-inheritance" applies. Consider amending checklist item 3 to: "If parent has paired proxy+target → child must inherit target half. If parent has proxy-only → disease-inheritance preempts; re-register with target half added."

## What I would do differently
Nothing mechanical — structural kills are cheap and correct. The `audit-2026-04-17` tag with missing `success_criteria` and empty `references` is a flag that the DB row itself pre-registers poorly; future audit passes could flag "K count = 1 AND no target-named KC" as pre-registration warning at DB-level instead of waiting for researcher preempt. That is an `experiment` CLI feature request, not a researcher action.

## Drain tally contribution
Row 26 (was 25 going in):
- Novel-mechanism PROVISIONALs: 5 (F#682, F#683, F#684, F#696, F#697)
- F#669-family preempt-KILLs: 6 (F#669, F#671, F#672, F#687, F#698, F#699)
- F#666-pure-standalone preempt-KILLs: **8** (F#700, F#701, F#703, F#705, F#706, F#707, F#708, **this**) — taxonomy-refactor trigger firmly at 8
- Hygiene-patch PROVISIONALs: 1 (F#702)
- Tautological-inter-adapter-delta preempt-KILLs: 3 (K1552, F#704, F#709) — §5 clause PROMOTED
- Template-regression antipattern sub-variants: 3 (F#705, F#708, F#709) — formal antipattern, PROMOTED; candidate-4th filed here
- SUPPORTEDs: 3 (budget_forcing, semantic_router, cayley_riemannian)
- Regular KILLs: 1 (kv_cache_reuse_honest)
- **Total: 26**
- 85 open P≤2 remain (was 86, minus 1 drain).

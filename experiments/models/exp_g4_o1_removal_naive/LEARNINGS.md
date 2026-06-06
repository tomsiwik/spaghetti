# LEARNINGS.md — exp_g4_o1_removal_naive

**Verdict:** KILLED (preempt-structural, F#666-pure standalone, 4th drain-window instance)

## Core learnings

1. **PPL-only KC is F#666-pure.** Guardrail 1007 names PPL as a proxy by name. A single-KC pre-reg built around PPL drift cannot be verified (proxy-PASS-alone is tautological; proxy-FAIL-alone is "a finding about the proxy, not a kill" per F#666). Applies even when the parent finding's own caveat asked for "PPL validation" — post-F#666 the modern bar is a paired target KC.

2. **Parent caveats can go stale.** F#161's 2026-03-15 caveat ("status supported not proven until PPL validation") predated guardrail 1007 (2026-04-19 per F#666). Do not treat a parent's own next-step suggestion as a valid KC template without re-checking against current discipline. Caveats age; guardrails supersede.

3. **Sibling template check is cheap and diagnostic.** F#133 (`exp_hash_ring_remove_expert`) solved the same shape ("does adapter removal degrade quality?") with a PAIRED KC (K1 PPL mean drop + K2 neighbor accuracy). A 30-second `experiment query "hash ring remove"` surfaces the sibling and the well-formed template falls out. Recommend: add sibling-template check to researcher pre-claim checklist.

4. **PPL-as-proxy is now drain-window-documented.** Rows 1–3 of the F#666-pure promotion were cos-sim / effective-rank / routing-accuracy. Row 4 (this filing) is the first PPL-only instance. Confirms guardrail 1007 applies lexically, not just to the explicitly-discussed cos-sim/rank triggers.

5. **4-instance threshold holds at §5 (no re-promote).** §5 F#666-pure clause was promoted at the 3-instance threshold (F#700/F#701/F#703). 4th instance applies the clause without re-editing §5 — same pattern as F#669-family at its 4th+ reuses.

## Action items

**Primary (analyst):** none blocking. Optional: append PPL-as-proxy lexical note to the F#666-pure antipattern memory `Anchors` list so future claimers see PPL named alongside cos-sim / rank / routing-acc.

**Secondary (researcher pre-claim process):** consider adding to pre-claim checklist — "If KC mentions PPL, check for paired target-metric KC. If not paired, F#666-pure applies."

**Systemic (non-blocking):** pre-reg-creation-time hygiene gate (mentioned in `mem-antipattern-prereg-hygiene-multi-defect`) should block empty `success_criteria` simultaneously with proxy-only KC. Current pre-reg had 2 hygiene defects + F#666-pure — would have been caught by a combined gate.

## What was NOT learned here

- Nothing about adapter removal on Gemma 4. The mechanism was not tested.
- Nothing about Grassmannian QR stability at N=25 → 24. F#417 remains supported on its own terms.
- Nothing about naive subtraction error bounds. F#161 remains supported on its own terms.
- No failure-mode literature citation — preempt-structural KILL has no mechanism failure to cite.

## Drain-window tally (after this filing)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **4 F#666-pure standalone preempt-KILLs** (F#700, F#701, F#703, and this filing) — §5 clause already promoted
- 1 hygiene-patch PROVISIONAL (F#702)
- 1 tautological-inter-variant-delta preempt-KILL (F#704) — antipattern memory filed, §5 deferred to 3rd
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 21**

— End LEARNINGS.md —

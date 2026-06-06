# LEARNINGS — exp_pierre_multi_adapter_serving_throughput

## One-line
12th F#669 reuse; 1st Pierre-serving-cluster F#669 child preempt-KILL (parent F#570 5-precondition cascade); target-only-KC-panel-under-preempt-KILL micro-pattern **canonicalizes** at this observation (3rd obs, cross-cluster triple-fire: F#738 behavioral/MEMENTO + F#739 engineering/MEMENTO + this engineering/Pierre-serving).

## What happened
- Claimed `exp_pierre_multi_adapter_serving_throughput` (P=2, micro, tags `serving p1`).
- KCs: K1911 (N=3 concurrent-stack throughput ratio < 50%, engineering target) + K1912 (peak memory > 40GB at N=5 stacks, engineering target).
- Parent `exp_prod_mlxlm_integration` is KILLED (F#570) with 5 preconditions T1B/T1C/T2/T3/DEP all unresolved.
- Both KCs measure runtime properties of a multi-adapter concurrent-serving harness on `pierre-g4e4b`. The harness does not exist; the base model does not exist; the trained adapters do not exist; the server body schema does not accept multi-adapter; the pip-package path is KILLED upstream.
- Preempt-KILL per F#669 applied; no MLX code written. `results.json` encodes `verdict=KILLED`, `all_pass=false`, both KCs `untested (preempt-blocked)`.

## Cluster and sub-axis findings
- **Cluster expansion:** F#669 now spans **two clusters** — MEMENTO (4 children: F#699/F#737/F#738/F#739) and Pierre-serving (1st child, this). Cross-cluster F#669 reuse is the strongest form of the structural pattern (not confined to a single parent's idiosyncrasies).
- **Multi-parent-run sub-axis:** unchanged at 2 observations (F#737 scalar-sweep + F#738 categorical cross-corpus). This experiment is a **2-point serving-config spot-measurement** at N∈{3,5} — distinct variant, not a canonical same-metric-across-configs sweep. Does NOT advance the sub-axis counter. Reviewer may promote if they decide the variant counts as sub-axis-advancing.
- **Target-only-KC-panel-under-preempt-KILL micro-pattern: CANONICALIZED.** 3 independent observations (F#738 behavioral/MEMENTO + F#739 engineering/MEMENTO + this engineering/Pierre-serving); cross-cluster triple-fire. Canonical form: an F#669 child whose KC panel is target-only (engineering OR behavioral) with no pairable proxy — satisfies F#666 by vacuous quantification rather than compound pairing. Legitimate F#666-compliance path.

## Reusable mechanism
When claiming a child experiment whose KCs measure runtime or engineering properties of a harness/model/mechanism that does not exist:
1. **Check the parent's verdict.** If `killed` or `provisional`, check whether the KCs are strictly stronger than the parent's KCs (they usually are for child experiments).
2. **Check F#666 compliance path.** If all child KCs are engineering or behavioral targets with no pairable proxy, F#666 is satisfied by vacuous quantification — this is a legitimate target-only-panel canonical variant now (post-this-experiment).
3. **Classify sub-axis.** Single-config (one N) vs canonical sweep (one metric across many N) vs categorical (distinct corpora) vs serving-config spot-measurement (distinct metrics at different N, new variant) vs your cluster's specific variant.
4. **Check cross-cluster independence.** If your F#669 child is in a different parent cluster than prior observations, cross-cluster triple-fire is stronger than within-cluster repetition.
5. **Write preempt-KILL scaffold.** No MLX code. `results.json` with `verdict=KILLED`, `all_pass=false`, both KCs `untested (preempt-blocked)`. MATH.md §1 preempt-theorem, §6 antipattern-scan against silent objective swaps, §7 anti-pattern checklist.
6. **Do NOT file an `_impl` companion** — preempt-structural kills are self-contained per F#687/F#698/F#699/F#737/F#738/F#739 precedent + reviewer.md §5. Unblock is parent-external + parent-extension.

## What I would do differently
- Nothing structural. The preempt-KILL path is now established procedure for F#669 children of parent-unverified experiments, and the target-only-panel canonical variant is now explicitly a legitimate F#666-compliance path.
- Minor: the experiment record's `platform` field (`~`) and empty `success_criteria` are F#702 hygiene issues. Deferred this iteration since preempt-KILL supersedes hygiene correction per established precedent; patchable when experiment is eventually re-claimed.

## Status of Pierre-serving cluster (post-this-experiment)
- **Parent F#570 KILLED:** 5 preconditions (T1B/T1C/T2/T3/DEP) unresolved as of 2026-04-18 source inspection.
- **F#669 children:** 1 (this).
- **Other F#570 children via different framings:** F#655 (ap-017 §s4 T5-K under F#652, software-infra-unbuilt route), F#657 (ap-017 28th composition-bug under F#652). Neither is F#669; both trace back to the same parent-KILL but via different reasoning paths (F#652 vs F#669).
- **Other Pierre-serving-adjacent children P≤2 still open:** check `experiment list -s open --tags serving` and `experiment list -s open --tags p1`. Most are likely preempt-KILL same way until F#570 resolves.

## Status of F#669 family (post-this-experiment)
- **Total reuses:** 12 (11 at F#739's filing; this is the 12th).
- **Clusters spanned:** 2 (MEMENTO: F#699/F#737/F#738/F#739; Pierre-serving: this).
- **Sub-axis canonical observations:** multi-parent-run at 2 obs (F#737 + F#738; pending 3rd).
- **Micro-pattern canonical observations:** target-only-KC-panel canonicalized at 3 obs (F#738 + F#739 + this) cross-cluster.
- **Watchlist micro-patterns advancing:** (to review) serving-config spot-measurement variant — 1st obs here; if it recurs at 2+ more distinct Pierre-serving children, may promote to sub-axis variant.

## Drain tally (updated)
- 12 novel-mechanism PROVISIONALs (unchanged): F#682/683/684/696/697/713/717/718/719/723/724/725.
- F#669 family: 12 reuses, 2 clusters.
- MEMENTO cluster: 4/? children resolved (F#699 + F#737 + F#738 + F#739). Remaining P≤2 candidate: `exp_memento_streaming_inference` (same parent F#685; likely preempt-KILL, could fold K1908 streaming regime into its KC panel or be distinct full-streaming-inference scope).
- Pierre-serving cluster: 1 child resolved (this). Other P≤2 open serving/p1 experiments likely preempt-KILL same way until F#570 resolves.

## Next claims after reviewer
- Continue draining P≤2 open micro from backlog.
- Pierre-serving cluster has ~N more P≤2 open children that will preempt-KILL identically until F#570 resolves; consolidating them into a single preempt-cluster learning may be appropriate if reviewer agrees.
- Hedgehog/JEPA hyperparameter sweeps may hit multi-parent-run sub-axis (canonical 3rd obs would promote to canonical sub-axis variant).

## Analyst synthesis (2026-04-24)
- **Core finding:** F#669 preempt-KILL extends cross-cluster (MEMENTO → Pierre-serving); target-only-KC-panel canonicalizes at 3rd obs as legitimate F#666-compliance path via vacuous quantification. Not a process bug — a pattern affirmation.
- **Why it matters:** When all child KCs are engineering or behavioral targets (no pairable proxy), F#666 is satisfied trivially. Researcher and reviewer need not contort a proxy pairing into existence. Cross-cluster independence (MEMENTO parent F#685 vs Pierre-serving parent F#570) rules out single-parent idiosyncrasy.
- **Implication for next iteration:** Continue backlog drain. Next Pierre-serving child (tags `serving` or `p1` with `depends_on=exp_prod_mlxlm_integration`) will preempt-KILL identically — researcher should reuse this scaffold verbatim without reinvestigating F#570 preconditions; parent state has not changed. Multi-parent-run sub-axis remains at 2 obs; Hedgehog/JEPA sweeps are likelier 3rd-obs candidates than further serving children.

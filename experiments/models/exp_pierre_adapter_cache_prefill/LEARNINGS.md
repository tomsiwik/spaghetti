# LEARNINGS — exp_pierre_adapter_cache_prefill

## One-line
13th F#669 reuse; 2nd Pierre-serving-cluster F#669 child preempt-KILL (parent F#570 5-precondition cascade + parent-extension gap); target-only-KC-panel post-canonical reuse (4th obs; 1st within-cluster reuse in Pierre-serving cluster; confirms canonical form's within-cluster portability across sub-axis variants).

## What happened
- Claimed `exp_pierre_adapter_cache_prefill` (P=2, micro, tags `serving p1`).
- KCs: K1913 (pre-fill doesn't reduce first-token latency by > 20%, engineering target) + K1914 (cache memory overhead > 2GB, engineering target).
- Parent `exp_prod_mlxlm_integration` is KILLED (F#570) with 5 preconditions T1B/T1C/T2/T3/DEP all unresolved; sibling F#740 confirmed state unchanged as of 2026-04-24.
- Both KCs measure runtime properties of a multi-adapter serving harness on `pierre-g4e4b` with an idle-time pre-fill scheduler. The harness does not exist; the base model does not exist; the trained adapters do not exist; the server body schema does not accept multi-adapter (renders pre-fill a no-op since single adapter is always resident); the pip-package path is KILLED upstream; plus parent-extension (idle-time hook, cache scheduler, TTFT cold-vs-warm instrumentation, cache-region memory probe) is beyond parent's scope.
- Preempt-KILL per F#669 applied; no MLX code written. `results.json` encodes `verdict=KILLED`, `all_pass=false`, both KCs `untested (preempt-blocked)`.

## Cluster and sub-axis findings
- **Cluster expansion:** F#669 spans **2 clusters** unchanged (MEMENTO + Pierre-serving). Pierre-serving cluster now has **2 F#669 children** (F#740 + this) plus 2 prior F#652-framed children (F#655, F#657). Within-cluster reuse is now established in the Pierre-serving cluster.
- **Multi-parent-run sub-axis:** unchanged at 2 observations (F#737 scalar-sweep + F#738 categorical cross-corpus). This experiment is a **single-config idle-time pre-fill** measurement (2 distinct engineering metrics on one config) — does NOT advance the sub-axis counter. Canonical promotion of multi-parent-run sub-axis still pending a genuine 3rd same-metric-across-configs observation.
- **Target-only-KC-panel-under-preempt-KILL micro-pattern: POST-CANONICAL REUSE.** 4 independent observations now (F#738 behavioral/MEMENTO + F#739 engineering/MEMENTO + F#740 engineering/Pierre-serving + this engineering/Pierre-serving). Canonicalized at F#740 via cross-cluster triple-fire. This observation is tally-only; strengthens canonical form by confirming within-cluster portability across distinct sub-axis variants (F#740 N-spot-measurement + this single-config idle-time pre-fill in the same Pierre-serving cluster).
- **Single-config-target-only-engineering sub-axis variant:** 2nd observation (1st was F#739 in MEMENTO cluster; this is Pierre-serving). Cross-cluster reuse of this specific variant. 1 more distinct observation would canonicalize the variant per the triple-fire rule.

## Reusable mechanism
When claiming a child experiment whose KCs measure runtime or engineering properties of a harness/model/mechanism that does not exist AND whose parent is in a cluster with a known preempt-KILL pattern:
1. **Check the parent's verdict.** If `killed` or `provisional`, check whether the KCs are strictly stronger than the parent's KCs (they usually are for child experiments). Additionally check whether parent-extension requirements (capabilities beyond parent's scope) are required.
2. **Check F#666 compliance path.** If all child KCs are engineering or behavioral targets with no pairable proxy, F#666 is satisfied by vacuous quantification — this is a legitimate target-only-panel canonical variant now (post-F#740).
3. **Classify sub-axis.** Single-config (one N, one metric or multiple metrics) vs canonical sweep (one metric across many N) vs categorical (distinct corpora) vs spot-measurement-at-N (distinct metrics at different N — F#740) vs single-config-idle-time-prefill (2 metrics on one config — F#739 and this) vs your cluster's specific variant.
4. **Check within-cluster reuse.** If your F#669 child is in a cluster that already has F#669 children (like Pierre-serving post-F#740), this is within-cluster reuse — still files its own finding but is a tally-only contribution to the canonical pattern rather than a canonicalization event.
5. **Consider cluster-level consolidation.** If a cluster accumulates 2+ F#669 children with the same structural blockers, flag to reviewer that future children in the cluster could be resolved as a batch under one consolidated cluster-level finding rather than filing one per child.
6. **Write preempt-KILL scaffold.** No MLX code. `results.json` with `verdict=KILLED`, `all_pass=false`, both KCs `untested (preempt-blocked)`. MATH.md §1 preempt-theorem (inherit parent-preconditions verbatim if parent state unchanged), §6 antipattern-scan against silent objective swaps tailored to the specific mechanism under test, §7 anti-pattern checklist.
7. **Do NOT file an `_impl` companion** — preempt-structural kills are self-contained per F#687/F#698/F#699/F#737/F#738/F#739/F#740 precedent + reviewer.md §5. Unblock is parent-external + parent-extension.

## What I would do differently
- Nothing structural. The preempt-KILL path is now well-established for F#669 children of Pierre-serving cluster; F#740 provided the scaffold template and the analyst explicitly anticipated this reuse.
- Minor: the experiment record's `platform` field (`~`) and empty `success_criteria` are F#702 hygiene issues. Deferred this iteration since preempt-KILL supersedes hygiene correction per established precedent; patchable when experiment is eventually re-claimed.
- Consolidation opportunity: if the reviewer agrees, future Pierre-serving preempt-KILLs could be resolved as a batch under one consolidated cluster-level finding rather than filing one F#669-reuse finding per child. This experiment files its own finding for now; decision deferred to reviewer.

## Status of Pierre-serving cluster (post-this-experiment)
- **Parent F#570 KILLED:** 5 preconditions (T1B/T1C/T2/T3/DEP) unresolved as of 2026-04-18 source inspection; confirmed unchanged at F#740 filing 2026-04-24.
- **F#669 children:** 2 (F#740 + this).
- **Other F#570 children via different framings:** F#655 (ap-017 §s4 T5-K under F#652, software-infra-unbuilt route), F#657 (ap-017 28th composition-bug under F#652). Neither is F#669; both trace back to the same parent-KILL but via different reasoning paths (F#652 vs F#669).
- **Other Pierre-serving-adjacent children P≤2 still open:** `exp_pierre_adapter_hotswap_latency_impl` (P=2, tagged `serving`/`p1`) likely preempt-KILL same way. Possibly more — check `experiment list -s open --tags serving` and `experiment list -s open --tags p1` after reviewer processes this.

## Status of F#669 family (post-this-experiment)
- **Total reuses:** 13 (12 at F#740's filing; this is the 13th).
- **Clusters spanned:** 2 (MEMENTO: F#699/F#737/F#738/F#739; Pierre-serving: F#740 + this).
- **Sub-axis canonical observations:** multi-parent-run at 2 obs (F#737 + F#738; pending 3rd).
- **Micro-pattern canonical observations:** target-only-KC-panel at 4 obs (F#738 + F#739 + F#740 + this); canonicalized at F#740; post-canonical reuse at 4th obs (this); 1st within-cluster reuse in Pierre-serving.
- **Watchlist micro-patterns advancing:** single-config-target-only-engineering sub-axis variant — 2 obs (F#739 + this); cross-cluster reuse; 1 more distinct obs would canonicalize.
- **Consolidation watchlist:** Pierre-serving cluster — 2 F#669 children + remaining open P≤2 likely to preempt-KILL identically; reviewer may consolidate future Pierre-serving children into one cluster-level finding.

## Drain tally (updated)
- 12 novel-mechanism PROVISIONALs (unchanged): F#682/683/684/696/697/713/717/718/719/723/724/725.
- F#669 family: 13 reuses, 2 clusters.
- MEMENTO cluster: 4/? children resolved (F#699 + F#737 + F#738 + F#739). Remaining P≤2 candidate: `exp_memento_streaming_inference` (same parent F#685; likely preempt-KILL).
- Pierre-serving cluster: 2 children resolved (F#740 + this). `exp_pierre_adapter_hotswap_latency_impl` and any other P≤2 open `serving`/`p1` experiments likely preempt-KILL same way until F#570 resolves.

## Next claims after reviewer
- Continue draining P≤2 open micro from backlog.
- Prefer Hedgehog/JEPA/MEMENTO-streaming over further Pierre-serving children if multi-parent-run sub-axis 3rd obs is desired (current event payload note).
- Single-config-target-only-engineering sub-axis variant at 2 obs (F#739 + this, cross-cluster); 3rd distinct obs would canonicalize this variant.
- If more Pierre-serving children remain and reviewer opts for cluster-level consolidation, subsequent ones can be batched under one finding.

## Analyst synthesis (2026-04-24)
- **Core finding:** F#669 13th reuse; within-cluster reuse established in Pierre-serving cluster (2 children + identical blockers + distinct sub-axis variants). Target-only-KC-panel post-canonical reuse strengthens canonical form's within-cluster portability (canonical form is cluster-portable AND sub-axis-portable within a cluster).
- **Why it matters:** The canonical form (target-only-KC-panel as F#666-vacuous-compliance path) now has 4 observations across 2 clusters and 3 sub-axis variants (behavioral, engineering×2 variants: N-spot-measurement and single-config-idle-time-prefill). It is a genuinely reusable F#666-compliance path, not a cluster-specific quirk. Cross-cluster + within-cluster + cross-sub-axis generalizability confirms it.
- **Implication for next iteration:** Continue backlog drain. Remaining Pierre-serving children (notably `exp_pierre_adapter_hotswap_latency_impl`) will preempt-KILL identically — reviewer may elect to consolidate them into a single cluster-level finding to reduce filing overhead. Multi-parent-run sub-axis still at 2 obs; Hedgehog/JEPA sweeps are likelier 3rd-obs candidates than further serving children. Single-config-target-only-engineering sub-axis variant at 2 obs; 3rd distinct obs canonicalizes.

## Literature / methodology context
- F#669 preempt-structural is an internalization of **pre-registered null-declaration**: instead of running an experiment whose measurement apparatus provably doesn't exist and reporting an uninformative "N/A", the methodology files an honest `KILLED (untested, preempt-blocked)` with the structural reason. This parallels pre-registration practice in ψ-science (Nosek et al., 2018 Reg. Rep. Fund.) and "research waste avoidance" (Chalmers & Glasziou 2009) — both argue that declaring an experiment un-runnable up-front is epistemically stronger than executing a degenerate version.
- The **target-only-KC-panel** (F#666-vacuous-compliance) mirrors the distinction between proxy and primary endpoint in clinical-trial design: when every KC is itself the target outcome (no surrogate/proxy exists), the proxy-target pairing rule is vacuously satisfied. No external literature to cite for this specific formulation — it is a project-internal canonical form, now 4 observations deep.
- The **consolidation question** — one F#669 child finding per preempt-KILL vs one cluster-level finding for many children — parallels meta-analytic grouping in systematic reviews: individual null results contribute to a cluster-level conclusion once the structural cause is established as shared. No web-search literature was needed; F#669's in-repo precedent is the governing reference.

## Consolidation watchlist — Pierre-serving cluster (post F#741)
Parent F#570 preconditions (T1B/T1C/T2/T3/DEP) unchanged at 2026-04-24; until parent reaches `supported`, the following P≤2 open experiments will preempt-KILL identically:
- `exp_pierre_adapter_hotswap_latency_impl` (P=2, tags `serving` `p1`) — hot-swap latency on Gemma 4; same F#570 5-precondition cascade + parent-extension (hot-swap hook). **3rd Pierre-serving F#669 child when claimed.**
- `exp_routing_cache_adapter_embeddings` (P=2, tag `serving`) — precomputed adapter-embedding cache for routing; likely F#669 child IF it depends on the same `pierre-g4e4b` + multi-adapter harness. Verify at claim time whether routing-embedding computation can run without the serving harness (may be partially executable stand-alone, in which case it's NOT preempt-KILL).
- `exp_adapter_fingerprint_uniqueness` (P=2) — adapter weight hashing; likely **NOT** F#669 (pure offline computation on adapter safetensors; only blocked by F#570 T3, not the full harness). Reviewer should re-classify at claim time rather than preempt-KILL by association.
- `exp_multi_tenant_serving` (P=5, already claimed by researcher) — out of P≤2 drain scope this iteration; same cluster.

**Recommendation for reviewer:** when `exp_pierre_adapter_hotswap_latency_impl` is next claimed and preempt-KILLs (3rd within-cluster F#669 child), elect **cluster-level consolidation** — file one consolidated Pierre-serving-cluster finding summarizing the shared F#570 cascade + the set of distinct parent-extension requirements across children, rather than a per-child F#669-reuse finding. F#741 and F#740 remain as the individually-filed canonicalization and 1st-reuse events; consolidation applies to 3rd-onward within-cluster reuses.

# LEARNINGS — exp_g4_gs_random_perm_n25

## What happened
- Claimed `exp_g4_gs_random_perm_n25` (P=2, micro, local-apple, tag `audit-2026-04-17` lineage-only).
- Single KC K1595 "worst/mean ≤ 1.5x" = removal-deviation worst/mean ratio (proxy stability/perturbation-magnitude geometric statistic), zero paired target.
- Parent F#160 (BitNet-2B, SUPPORTED 2026-03-28, pre-F#666) had 2 KCs both proxy (K1 worst/mean ratio < 2.0x, K2 abs worst < 1% at d=256) — zero target-metric KC. Child stripped K2 (candidate paired-PROXY-half-strip sub-variant).
- Preempt-structural KILL under F#666 / guardrail 1007 (proxy-only KC has no compliant verdict).
- 9th drain-window F#666-pure-standalone instance; 3rd derived-geometric sub-flavor with stability/perturbation semantics.
- **TAXONOMY-REFACTOR EXECUTION TRIGGER FIRES** per analyst's pre-commit (option (b) at F#710 row).
- **2nd `proxy-only-lineage-inheritance` instance** — watchlist threshold met (1st was F#710 / parent F#72; 2nd is this/F#160).

## Forward-relevant recommendations for analyst

1. **EXECUTE TAXONOMY-REFACTOR option (b)** — trigger fires at 9th instance per analyst's own pre-commit in `mem-antipattern-f666-pure-standalone-preempt-kill` Escalation block. Split memory body into 4 sub-bucket sections under retained super-category shell:
   - {derived-geometric: cos-sim/eff-rank/pairwise-cos/stability-ratio — F#700, F#701, **F#[this]**}
     - Optional sub-distinction within: static-similarity (F#700, F#701) vs stability/perturbation-magnitude (F#[this])
   - {summary-distributional: PPL — F#705, F#708 (confirmed-recurrent)}
   - {detection/classification: FNR/classification-accuracy — F#706}
   - {routing: routing-acc/R/collision-rate — F#703, F#707, F#710 (confirmed-recurrent)}

2. **File watchlist** `mem-watchlist-f666pure-proxy-only-lineage-inheritance` — 2nd instance threshold met. Sub-pattern: child pre-reg structurally proxy-only AND parent finding was itself pre-F#666 SUPPORTED on proxy-only KCs (parent itself was disease under prior regime, child continues disease under stricter current regime). Distinct from template-regression sub-variants 1–3:
   - F#705 stale-caveat-inheritance (passive doc-rot from F#161)
   - F#708 paired-design-half-stripping (parent had paired target-like + proxy; child stripped target-like)
   - F#709 explicit-anti-caveat-inheritance (parent caveats explicitly labeled structure vacuous)
   
   Promotion threshold = 3rd instance per F#704 / F#669 / template-regression convention.

3. **Candidate 4th template-regression sub-variant** `paired-PROXY-half-strip` — researcher recommends **(iii) defer until 2nd instance surfaces** for memory hygiene. Distinguishing from F#708's paired-target-half-strip:
   - F#708: parent F#133 had paired KC with one proxy + one target-like (100% neighbor accuracy) → child stripped target-like half
   - This: parent F#160 had paired KC with both axes proxy (K1 ratio + K2 abs magnitude) → child stripped one proxy axis
   
   Structurally weaker upstream signal than F#708 (no target half existed to lose), so deferral is the conservative choice. Document in F#[this] anchor entry as filed-for-future-consideration.

4. **Update F#666-pure antipattern memory Anchors** — add F#[this] as 9th row, annotated "3rd derived-geometric, stability/perturbation sub-flavor, taxonomy-refactor execution trigger fires", mirroring F#710's "2nd routing-acc, confirmed-recurrent" annotation. Source list: append `exp_g4_gs_random_perm_n25`.

5. **Unblock v2 spec** (captured in PAPER.md for anyone who wants to re-register):
   - Proxy K1: K1595-equivalent (worst/mean ratio ≤ 1.5x)
   - Proxy K2 (RESTORE parent F#160's stripped axis): abs worst removal-deviation < 1% at Gemma 4 v_proj dim
   - Target K3: MMLU-Pro subject-domain accuracy on Gemma 4 N=25 v_proj composition (random GS permutation) within 5pp of single-best-adapter baseline, OR Spearman |r| ≥ 0.4 (worst/mean ratio ↔ generation-quality delta across permutation seeds)
   - References: `{F#666, F#160, F#477, F#627, F#703, F#710, arxiv:2106.09685 LoRA, parent's random-permutation-stability arxiv if any}`

6. **Audit-tag-lineage-vs-rerun distinction continues to hold** — `audit-2026-04-17` (no `-rerun`) = lineage-only; `-rerun` = fix-before-rerun. Established at F#708 row, applied consistently through F#710 and now F#[this].

7. **Pre-claim 5-item checklist worked correctly** — caught both the F#666-pure structure (item 1 → finding-get → item 2 caveat-scan) AND the proxy-only-lineage-inheritance pattern (item 4 → stale-advice-vs-current-guardrail check identified parent as pre-F#666 SUPPORTED). The "2nd instance trigger" recognition came from cross-referencing F#710 LEARNINGS.md ("Re-evaluate if a 2nd pre-F#666-parent F#666-pure child appears") — this confirms maintaining LEARNINGS.md cross-references is load-bearing for the 5-item checklist's effectiveness.

## What I would do differently
Nothing mechanical — structural kills are cheap and correct. The `audit-2026-04-17` lineage-only convention now has 3 consecutive applications (F#708, F#710, this) and is well-established; can be promoted to formal naming convention if not already in PLAN.md. The 9th-instance taxonomy-refactor trigger firing as analyst pre-committed is a clean handoff — no researcher second-guessing required.

One small hygiene note: parent F#160's evidence text mentions "3 configs, 3 seeds, P=5 perms" but the original DB entry (`experiment query "exp_gs_random_permutation_validation"`) shows the title only. Future watchlist filings should include parent's K-count and target-presence at first reference for downstream readers; researcher template includes this verbatim now.

## Drain tally contribution
Row 27 (was 26 going in):
- Novel-mechanism PROVISIONALs: 5 (F#682, F#683, F#684, F#696, F#697)
- F#669-family preempt-KILLs: 6 (F#669, F#671, F#672, F#687, F#698, F#699)
- F#666-pure-standalone preempt-KILLs: **9** (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, **this**) — **TAXONOMY-REFACTOR EXECUTION TRIGGER FIRES**
- Hygiene-patch PROVISIONALs: 1 (F#702)
- Tautological-inter-adapter-delta preempt-KILLs: 3 (K1552, F#704, F#709) — §5 clause PROMOTED
- Template-regression antipattern sub-variants: 3 (F#705, F#708, F#709) — formal antipattern, PROMOTED; candidate-4th `paired-PROXY-half-strip` filed here (defer)
- Proxy-only-lineage-inheritance: **2** (F#710, **this**) — watchlist threshold met
- SUPPORTEDs: 3 (budget_forcing, semantic_router, cayley_riemannian)
- Regular KILLs: 1 (kv_cache_reuse_honest)
- **Total: 27**
- 84 open P≤2 remain (was 85, minus 1 drain).

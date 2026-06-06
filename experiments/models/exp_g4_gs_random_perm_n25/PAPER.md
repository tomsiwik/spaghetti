# PAPER — exp_g4_gs_random_perm_n25

**Verdict: KILLED (preempt-structural, F#666-pure-standalone)**

## Prediction vs. Measurement

| # | Prediction (MATH.md) | Measurement | Pass/Fail |
|---|----------------------|-------------|-----------|
| P1 | Claim admits no F#666-compliant verdict under current KC set (K1595 proxy-only stability ratio) | Deductive — proxy-only KC has 0 paired target half; both outcomes forbidden | ✅ |
| P2 | Parent F#160 is pre-F#666 SUPPORTED on 2 proxy-only KCs (K1 ratio + K2 abs); child continues disease as 2nd `proxy-only-lineage-inheritance` instance (watchlist threshold met) | `experiment finding-get 160` confirms parent evidence "K1 PASS: permuted expected worst/mean=1.42x (<2.0x). K2 PASS: abs worst 0.446% at d=256 (<1%)"; both proxy, no target half | ✅ |
| P3 | Cross-arch novelty (BitNet→Gemma 4) does not rescue | F#477 Gemma 4 shallow-regime + N=25 v_proj stress on amortization regime not measured by parent | ✅ |
| P4 | 9th F#666-pure drain-window; 3rd derived-geometric flavor; **taxonomy-refactor execution trigger fires** per analyst pre-commit at F#710 | `experiment finding-list` rows F#700, 701, 703, 705, 706, 707, 708, 710 → this = 9; mem block escalation lists "9th OR first non-canonical proxy flavor" trigger | ✅ |
| P5 | NEW candidate sub-pattern: `paired-PROXY-half-strip` (parent K1 + K2 both proxy, child kept K1 only) | `experiment finding-get 160` evidence shows K1 worst/mean ratio AND K2 abs worst at d=256 both PASS in parent; child K1595 = K1-only | ✅ |

## Claim

K1595: "GS random permutation on Gemma 4 N=25 v_proj gives `worst/mean ≤ 1.5x` removal deviation." Single kill criterion, single proxy threshold (a structural-stability ratio of the per-position removal-perturbation distribution), no paired target metric.

Parent cite: **F#160** (exp_gs_random_permutation_validation, SUPPORTED 2026-03-28, **pre-F#666**). Parent K-structure was 2 proxy KCs (K1 worst/mean ratio < 2.0x; K2 abs worst < 1% at d=256). Parent had zero target-metric KC. Child inherits proxy-only structure under stricter current F#666 regime AND strips parent's K2 axis.

## Why this is preempt-KILL, not measurement-KILL

F#666 (guardrail 1007) disqualifies *both* possible outcomes of a proxy-only KC:

- **worst/mean ≤ 1.5x → supported**: tautological per F#666 canonical. Worst/mean removal-deviation ratio is a structural geometric property of the perturbation distribution, with no known coupling to behavioral outcomes (MMLU-Pro accuracy, oracle-gap, generation quality). F#477's Gemma 4 shared-shallow regime (only 2/5 domains beat base at rank 6) precludes assuming geometric stability ⇒ behavioral utility on this architecture.
- **worst/mean > 1.5x → killed**: forbidden per F#666 — "proxy-FAIL + target-absent = finding about proxy, not a kill."

Neither verdict is F#666-compliant. Measurement would waste compute without producing an admissible finding.

## Sub-pattern flag — paired-PROXY-half-strip (NEW candidate template-regression sub-variant)

Parent F#160 had two paired KCs (K1 worst/mean ratio + K2 abs worst at d=256). Child K1595 keeps only the K1 ratio axis, dropping the K2 absolute-magnitude axis. This is structurally distinct from F#708's `paired-design-half-stripping` sub-variant:

| Sub-variant | Parent paired structure | Child stripping |
|-------------|------------------------|-----------------|
| F#708 (paired-target-half-strip, sub-variant 2 of 3 promoted) | K1 PPL (proxy) + K2 100% neighbor accuracy (target-like) | Stripped target-like K2, kept proxy K1 |
| **This (candidate paired-PROXY-half-strip)** | K1 worst/mean ratio (proxy) + K2 abs worst (proxy) | Stripped one proxy axis, kept other proxy axis |

Three classification options:
- (i) New 4th sub-variant of `mem-antipattern-template-regression` (paired-PROXY-half-strip, distinct from F#708's paired-TARGET-half-strip)
- (ii) Note as variant within F#708 sub-variant b without separate filing (degree-of-stripping difference, same upstream structure of "child takes subset of parent's KC axes")
- (iii) Defer until 2nd instance of paired-PROXY-half-strip surfaces

Recommend (iii) — structurally weaker upstream signal than F#708 (no target half existed to lose, so the "regression" is one-axis less proxy coverage rather than loss of behavioral measurement). Filed for analyst decision.

## Sub-pattern flag — proxy-only-lineage-inheritance (2nd instance, watchlist threshold met)

Per researcher recommendation in F#710 LEARNINGS.md ("Re-evaluate if a 2nd pre-F#666-parent F#666-pure child appears"), the 2nd instance triggers watchlist filing per F#704/F#669 convention.

| Instance | Finding | Experiment | Parent | Parent regime |
|----------|---------|-----------|--------|---------------|
| 1st | F#710 | exp_g4_gumbel_top2_n50 | F#72 (BitNet-2B Gumbel routing) | pre-F#666 SUPPORTED 2026-03-26, 3 proxy-only KCs |
| **2nd** | **F#[this]** | **exp_g4_gs_random_perm_n25** | **F#160 (BitNet-2B GS random permutation)** | **pre-F#666 SUPPORTED 2026-03-28, 2 proxy-only KCs** |

Recommend filing `mem-watchlist-f666pure-proxy-only-lineage-inheritance` for analyst (promotion threshold = 3rd instance per existing F#704 / F#669 / template-regression convention).

## Hygiene

2 defects (empty `success_criteria`, empty `references`), below 3+ threshold for hygiene-multi-defect. F#666-pure-standalone keys on KC structure independently of hygiene count, per F#703.

## Audit tag lineage

`audit-2026-04-17` (no `-rerun` suffix) → lineage-only per analyst convention (established at F#708 row, applied consistently through F#710). Does NOT trigger `.audit/RECOVERY_PLAN.md` fix-before-rerun. No audit-specified fix to apply.

## Cross-architecture note (non-load-bearing)

BitNet-2B (1.58-bit ternary, parent F#160 measured up to N=24 at d=256) → Gemma 4 (4-bit dense decoder-only, child claims N=25 on v_proj) is a cross-architecture transfer claim. Per F#477, Gemma 4 adapters at rank 6 beat base on only 2/5 domains (shared-shallow regime). Even if the KC were well-formed (paired with a target half), the natural prior is that the worst/mean ratio reverts toward parent's pre-permutation 1.96x range on Gemma 4 at N=25 v_proj because:

- v_proj has different singular-value structure than the dense paths F#160 measured.
- N=25 stresses random-permutation amortization more than F#160's range.
- 4-bit base quantization adds a perturbation floor not present in BitNet-2B's training-time continuous values.

This strengthens the case for a target-KC in v2 — but does not alter the current preempt verdict.

## Unblock path (v2)

`exp_g4_gs_random_perm_n25_behavioral` with paired KC and restored parent K2:
- Proxy K1: K1595-equivalent — worst/mean ratio ≤ 1.5x
- Proxy K2 (restored from parent F#160): abs worst removal-deviation < 1% at d corresponding to Gemma 4 v_proj dim
- Target K3: MMLU-Pro subject-domain accuracy on Gemma 4 N=25 v_proj composition (with random GS permutation) within 5pp of single-best-adapter baseline, **OR** Spearman |r| ≥ 0.4 between worst/mean ratio and downstream generation-quality delta across permutation seeds
- Both proxy + target → supported; either-failing → killed (F#666 / 1007 compliant)
- Fill `references` with `{F#666, F#160, F#477, F#627, F#703, F#710, arxiv:2106.09685 LoRA, parent's random-permutation-stability arxiv if any}`
- Fill `success_criteria` per pre-claim checklist

## Assumptions

- "Worst/mean removal deviation" in K1595 = the ratio measured in parent F#160's protocol (per-position absolute-deviation distribution under expert removal with random GS permutation). If the DB row intended a different operational definition, the proxy-only structural-stability classification still applies and the verdict is unchanged.
- `audit-2026-04-17` tag without `-rerun` suffix treated as lineage-only per established convention (F#708 row onward). No `.audit/RECOVERY_PLAN.md` fix required.
- Proxy flavor classification as "derived-geometric (3rd, stability sub-flavor)" is the natural fit within analyst's planned option-(b) refactor buckets. A fresh "stability/perturbation" sub-bucket is also defensible; analyst decides.

## Antipattern scan (researcher pre-claim)

| Pattern | Applies? | Note |
|---------|----------|------|
| F#669-family depends_on chain | ✗ | depends_on = [] |
| F#666-pure-standalone | ✅ | Primary verdict basis (9th instance) |
| Tautological-inter-adapter-delta | ✗ | Single-threshold ratio, not inter-variant |
| Template-regression (3 promoted sub-variants) | Partial — candidate 4th sub-variant `paired-PROXY-half-strip` filed | Recommend defer until 2nd instance |
| Hygiene-multi-defect | ✗ | 2 < 3+ threshold |
| Motivation-premise-disproven-by-db | ✗ | Parent F#160 is genuinely SUPPORTED as notes claim |
| Proxy-only-lineage-inheritance | ✅ (2nd instance) | Watchlist threshold met — file `mem-watchlist-f666pure-proxy-only-lineage-inheritance` |
| Parent-mechanism-anchor-non-inheritance | Vacuous | Parent F#160 has no closed-form mechanism formula to inherit |

## TAXONOMY-REFACTOR execution trigger (9th-instance, fires now)

Per `mem-antipattern-f666-pure-standalone-preempt-kill` Escalation block (analyst 2026-04-24): "**option (b) pre-committed; execution triggered by 9th F#666-pure instance OR first non-canonical proxy flavor, whichever comes first**." This kill IS the 9th F#666-pure instance → trigger fires. Planned split buckets per analyst memory:

- {derived-geometric: cos-sim/eff-rank/pairwise-cos — F#700, F#701, **+ F#[this] worst/mean stability ratio (3rd, possible stability/perturbation sub-bucket)**}
- {summary-distributional: PPL — F#705, F#708 (confirmed-recurrent)}
- {detection/classification: FNR/TPR/FPR/classification-accuracy — F#706}
- {routing: routing-acc/R/collision-rate/match-rate — F#703, F#707, F#710 (confirmed-recurrent)}

Researcher hands taxonomy-refactor execution to analyst per established split-of-labor (analyst owns memory editing).

## References

- F#666 — target-gated KC canonical
- F#160 — parent, pre-F#666 SUPPORTED on 2 proxy-only KCs
- F#477 — Gemma 4 shallow-regime evidence
- F#627 — v_proj+o_proj target precedent
- F#703 — 1st routing-acc F#666-pure (canonical for taxonomy buckets)
- F#710 — 1st `proxy-only-lineage-inheritance` candidate (this is 2nd)
- F#700, F#701, F#705, F#706, F#707, F#708 — F#666-pure drain-window siblings
- `mem-antipattern-f666-pure-standalone-preempt-kill` — option (b) pre-commit (analyst 2026-04-24)
- `mem-antipattern-template-regression` — promoted F#709; candidate 4th sub-variant `paired-PROXY-half-strip` filed
- Guardrail 1007 (this repo's PLAN.md / CLAUDE.md target-gated KILL discipline)

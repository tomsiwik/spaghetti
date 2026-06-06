# PAPER — exp_g4_gumbel_top2_n50

**Verdict: KILLED (preempt-structural, F#666-pure-standalone)**

## Prediction vs. Measurement

| # | Prediction (MATH.md) | Measurement | Pass/Fail |
|---|----------------------|-------------|-----------|
| P1 | Claim admits no F#666-compliant verdict under current KC set (K1591 proxy-only) | Deductive — proxy-only KC has 0 paired target half; both outcomes forbidden | ✅ |
| P2 | Parent F#72 is pre-F#666 SUPPORTED on 3 proxy-only KCs; child continues disease | `experiment finding-get 72` confirms K1/K2/K3 all proxy, zero target | ✅ |
| P3 | Cross-arch novelty (BitNet → Gemma 4) does not rescue | F#477 Gemma 4 shallow-regime; novelty is orthogonal to KC-structure rescue | ✅ |
| P4 | 8th F#666-pure drain-window; 2nd routing-acc sub-flavor (confirmed-recurrent) | `experiment finding-list` rows F#700,701,703,705,706,707,708 → this = 8 | ✅ |

## Claim

K1591: "Gumbel top-2 routing on Gemma 4 at N=50 hits `acc ≥ 85%` on held-out queries." Single kill criterion, single proxy threshold, no paired target metric.

Parent cite: **F#72** (exp_bitnet_gumbel_routing_n50, SUPPORTED 2026-03-26, pre-F#666). Parent K-structure was entirely proxy (K1 Gumbel top-2 routing accuracy 86.33%; K2 γ_uniform = 0.996; K3 max-degradation-per-domain). Parent itself had zero target-metric KC. Child inherits the proxy-only structure without adding a target half.

## Why this is preempt-KILL, not measurement-KILL

F#666 (guardrail 1007) disqualifies *both* possible outcomes of a proxy-only KC:

- **acc ≥ 85% → supported**: tautological per F#666 canonical counter-example (40.2% routing-acc + 0.0% target gap). SUPPORTED on proxy-only has been shown empirically non-predictive of behavioral outcomes on Gemma-family composition.
- **acc < 85% → killed**: forbidden per F#666 rule — "proxy-FAIL + target-absent = finding about proxy, not a kill."

Neither verdict is F#666-compliant. Measurement would waste compute without producing an admissible finding.

## Not template-regression (in the 3-sub-variant sense)

F#708 (paired-design-half-stripping) and F#709 (explicit-anti-caveat-inheritance) require parent to have had a well-formed or explicitly-caveated target/anti-caveat. Parent F#72 had neither — it was SUPPORTED on all-proxy KCs under a pre-F#666 regime. Child K1591 is a **proxy-only-lineage-inheritance** pattern: disease continues from a pre-F#666 ancestor. Filed as candidate 4th sub-variant (or as non-novel F#666-pure continuation) for analyst consideration; not load-bearing for the kill.

## Hygiene

2 defects (empty `success_criteria`, empty `references`), below 3+ threshold for hygiene-multi-defect. F#666-pure-standalone keys on KC structure independently of hygiene count, per F#703.

## Audit tag lineage

`audit-2026-04-17` (no `-rerun` suffix) → lineage-only per prior analyst convention (established at F#708 row). Does NOT trigger RECOVERY_PLAN.md fix-before-rerun. No audit-specified fix to apply.

## Cross-architecture note (non-load-bearing)

BitNet-2B (N=49, 1.58-bit ternary) → Gemma 4 (N=50, dense decoder-only) is a cross-architecture transfer claim. Per F#477 Gemma 4 adapters at rank 6 beat base on only 2/5 domains (shared-shallow regime). Even if the KC were well-formed (paired with a target half), the natural prior is that BitNet's Gumbel routing generalization is not automatic on Gemma 4. This strengthens the case for a target-KC in v2 — but does not alter the current preempt verdict.

## Unblock path (v2)

`exp_g4_gumbel_top2_n50_behavioral` with paired KC:
- Proxy: K1591-equivalent — routing-acc ≥ 85% on held-out
- Target: MMLU-Pro subject-domain accuracy within 5pp of oracle-adapter baseline at N=50, **or** Spearman |r| ≥ 0.4 between routing confidence and downstream generation-quality delta
- Both-pass → supported; either-fail → killed (F#666 / 1007 compliant)
- Fill `references` with {F#666, F#72, F#257, F#477, F#703, arxiv:1611.01144 Gumbel-softmax}
- Fill `success_criteria` per pre-claim checklist

## Assumptions

- `acc` in K1591 text = routing classification accuracy (match rate vs. oracle expert id), consistent with parent F#72 K1 language ("Gumbel top-2 routing accuracy 86.33%"). If the DB row intended `acc` = task accuracy (behavioral), the claim would still be a one-KC single-threshold no-paired-proxy structure — which is F#666-inverse but still F#666-non-compliant; verdict unchanged.
- `audit-2026-04-17` tag without `-rerun` suffix treated as lineage-only. No `.audit/RECOVERY_PLAN.md` fix required.

## Antipattern scan (researcher pre-claim)

| Pattern | Applies? | Note |
|---------|----------|------|
| F#669-family depends_on chain | ✗ | depends_on = [] |
| F#666-pure-standalone | ✅ | Primary verdict basis |
| Tautological-inter-adapter-delta | ✗ | Single-threshold, not inter-variant |
| Template-regression (paired-design-half-strip / explicit-anti-caveat) | Partial — parent had no paired half; candidate 4th sub-variant = proxy-only-lineage-inheritance | Filed for analyst |
| Hygiene-multi-defect | ✗ | 2 < 3+ threshold |
| Motivation-premise-disproven-by-db | ✗ | Parent F#72 is actually SUPPORTED as notes claim |

## References

- F#666 — target-gated KC canonical
- F#72 — parent, pre-F#666 SUPPORTED on 3 proxy-only KCs
- F#703 — 1st routing-acc F#666-pure
- F#257 — oracle-gap canonical target metric
- F#477 — Gemma 4 shallow-regime evidence
- F#700, F#701, F#705, F#706, F#707, F#708 — F#666-pure drain-window siblings
- Guardrail 1007 (this repo's CLAUDE.md equivalent)

# Adversarial Review: P7.B1 — Weighted Multi-Adapter Composition

## Verdict: PROCEED (supported, with caveats noted)

## What's Solid

1. **Theorem is correct and trivial.** Null-space closure under convex combination follows
   directly from subspace closure under linear combination. The proof is sound.

2. **Orthogonality verification is definitive.** max|W_v @ D| = 9.57e-7, six orders below
   the 1e-4 threshold. This is not marginal — it's machine-precision-level confirmation.

3. **Data integrity.** PAPER.md numbers match results.json. Per-domain breakdowns are
   internally consistent. No evidence of fabrication.

4. **Kill criteria pass cleanly.** K1303: 32.7pp (threshold 3pp). K1304: -18.5pp improvement
   (threshold <2pp degradation). K1305: 43.6pp benefit. Margins are large.

5. **Honest caveats.** PAPER.md explicitly flags memorization scale, near-uniform weights,
   lack of behavioral eval, and the ensemble-vs-composition confound.

## Non-Blocking Issues

### 1. This tests averaging, not routing
TF-IDF weight entropy is 0.996–1.000. Max weight ~0.24 vs uniform 0.20. The "routing"
is indistinguishable from uniform averaging. The experiment title says "weighted
multi-adapter composition" but the mechanism is effectively "average all 5 adapters."

PAPER.md acknowledges this ("ensemble effect dominates") — the paper is honest, so this
is noted but not blocking.

### 2. Oracle picks wrong domains — adapters aren't domain-specific at this scale
In results.json, the oracle (lowest-loss single adapter) frequently selects a domain NOT
matching the input:
- code text_1: oracle = medical
- code+finance text_0: oracle = medical  
- code+finance text_1: oracle = legal
- math+medical text_1: oracle = finance
- legal+finance text_0: oracle = medical

This means individual adapters are generic regularizers at 8 texts / 300 iters, not
domain-specialized models. The "domain routing" narrative doesn't apply at this scale.
This undermines the framing but not the structural finding.

### 3. Percentage improvement language
Kill criteria use "pp" (typically percentage points = absolute) but measure relative
improvement percentages. The implementation is consistent internally, but "32.7pp" reads
as 32.7 absolute percentage points when it's actually a 32.7% relative reduction in NTP
loss (absolute: 8.70 → 5.85 = 2.85 NTP loss units). Minor nomenclature issue.

### 4. No behavioral evaluation (per guardrail 1008)
The project proved PPL doesn't predict task quality (r=0.08). NTP loss improvement of
32.7% doesn't tell us whether weighted composition produces qualitatively better text.
This is honestly flagged in PAPER.md. Status "supported" (not "conclusive") is appropriate
precisely because of this gap.

## What the Finding Actually Shows

**Core structural result (conclusive):** Null-space adapters can be safely combined via
any convex weight vector without interfering with the base model. This is a mathematical
guarantee verified empirically.

**Practical result (provisional):** At memorization scale with near-uniform weights,
averaging all adapters beats picking one. This is likely a generic ensembling effect,
not a validation of routing-based composition. The null-space guarantee ensures averaging
is safe, but whether it's *better than averaging non-null-space adapters* is untested.

## Status Assessment

**SUPPORTED** is the correct status. The theorem is verified (would be conclusive in
isolation), but the practical significance depends on:
1. Whether peaked weights (from better routing or larger vocab) still outperform exclusive
2. Whether domain-specialized adapters (not memorization) show the same pattern
3. Whether NTP improvement translates to behavioral improvement

## Recommended Next Steps (for analyst)

- Note the ensemble-vs-composition confound as the key open question
- Flag that a non-null-space control would disambiguate whether the benefit comes from
  null-space structure or generic ensembling
- The structural guarantee (safe composition) is the durable finding; the routing
  improvement numbers are scale-dependent

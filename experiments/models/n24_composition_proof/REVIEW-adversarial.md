# Peer Review: N=24 Composition Proof (Post-Revision Re-Review)

## Experiment Type
Frontier extension. Extends the Pierre pipeline (ridge router + Grassmannian skeleton
+ NRE compose), proven at N=5 (Finding #287), to N=24. Finding status capped at
provisional.

## Revision Verification

Six fixes were requested in the prior review. Status of each:

| # | Fix Requested | Applied? | Notes |
|---|--------------|----------|-------|
| 1 | B-matrix cosine relabeled, Theorem 2 not directly verified noted | YES | Every cosine mention now specifies "B-matrix cosine" with caveats. Thorough. |
| 2 | Theorem 1 downgraded to "Capacity Argument" | MOSTLY | MATH.md uses "Capacity Argument 1" correctly throughout. PAPER.md header (line 5) still says "Theorem 1 (Routing Capacity)" and line 199 says "Theorem 1 corollary confirmed" -- inconsistent with body text which correctly uses "Capacity Argument." |
| 3 | Theorem 2 /r factor removed, Frobenius submultiplicativity | YES | /r factor removed. Bound is correct. Derivation path has a presentation issue (see below) but final result is valid. |
| 4 | Theorem 3 relabeled "Empirical Model" | MOSTLY | MATH.md uses "Empirical Model 3" correctly with "Justification (not a proof)." PAPER.md header (line 13) still says "Theorem 3 (Composition Bound)" -- inconsistent with body. |
| 5 | Per-domain PPL degradation added, code +35.7% flagged | YES | Prediction table includes matching-A (< 10%) and non-matching-A (not modeled). Code +35.7% correctly flagged as wrong-A-matrix projection. |
| 6 | Status changed to PROVISIONAL | YES | Provisional throughout with explicit caveats list (PAPER.md lines 244-249). |

**Overall: 4/6 fully applied, 2/6 mostly applied (minor header inconsistencies).**

## Hack Detector
- Fix count: 0. Same 3-component system (skeleton + router + NRE compose) as N=5. No new mechanisms.
- Is MATH.md a proof or a description? Mixed -- honestly labeled. Theorem 2 has a genuine proof with QED. Capacity Argument 1 and Empirical Model 3 are correctly labeled as non-proofs.
- Metric used as evidence: Routing accuracy (behavioral, directly relevant), B-matrix cosine (geometric proxy for DeltaW cosine, honestly caveated), per-domain PPL degradation (behavioral, matching-A distinction correctly made).
- Kill criteria source: K753 (50% router accuracy) -- partially derived from random baseline 4.2%. K754 (0.10 cosine) -- from Cao et al. empirical threshold. K755 (3.0x composed PPL) -- generous but reasonable for frontier extension. Mixed derivation.

## Self-Test Audit

All 6 self-test items pass:

1. One-sentence impossibility property: Single property (Grassmannian orthogonality), clearly stated. PASS.
2. Cited theorems: JL-lemma, Tikhonov, Grassmannian packing, Cao et al. -- all real, conditions honestly flagged (JL as capacity only). PASS.
3. Predicted numbers: B-matrix |cos| in [0.01, 0.05], router > 60% (conditional), per-domain < 10% (calibrated). Specific and falsifiable. PASS.
4. Falsification condition: Targets proof (coherence exceeds bound) and capacity argument (Delta > 0 yet routing fails). PASS.
5. Hyperparameter count: 2 (lambda, LORA_SCALE), neither searched. PASS.
6. Hack check: No new mechanisms. Frontier extension of proven system. PASS.

## Mathematical Soundness

### Capacity Argument 1 (Routing) -- HONEST

Now correctly labeled as a capacity argument rather than a theorem. The gap between
JL capacity (d >= 227 suffices) and ridge regression realization is explicitly
acknowledged: "JL guarantees that a RANDOM linear projection preserves distances.
Ridge regression is NOT a random projection." The conditional nature ("IF Delta > 0
for all pairs") is stated, and the failure when Delta ~ 0 is predicted by the
argument's own conditions.

Verdict: The argument does exactly what it claims -- establishes capacity, acknowledges
it does not guarantee classification accuracy. PASS.

### Theorem 2 (Orthogonality Preservation) -- SOUND with presentation issue

The final bound is correct:

  |cos(DeltaW_i, DeltaW_j)| <= ||A_i^T A_j||_F (when A_i have orthonormal columns)

However, the derivation path stated in lines 123-124 does not actually produce this
bound. The text claims to derive line 121 by applying |tr(X^T Y)| <= ||X||_F ||Y||_F
with X = A_i B_i, Y = A_j B_j, then using submultiplicativity on each factor.
This path yields ||A_i||_F * ||A_j||_F * ||B_i||_F * ||B_j||_F, not
||A_i^T A_j||_F * ||B_i||_F * ||B_j||_F. The correct path to the tighter bound
is: set M = A_i^T A_j, apply |tr(B_i^T M B_j)| <= ||M^T B_i||_F ||B_j||_F
<= ||M||_F ||B_i||_F ||B_j||_F.

The final result on line 128-129 and the simplification to ||A_i^T A_j||_F are
both correct. The error is in the stated intermediate derivation, not the conclusion.

Verdict: PASS (correct result, imprecise derivation presentation). Non-blocking.

### Empirical Model 3 (Composition) -- HONESTLY LABELED

Now correctly labeled as "not a proof" with "empirically-calibrated coefficient."
No overclaiming. The free parameter c ~ 1 and its N=5 calibration origin are stated.

Verdict: PASS (as a model, not a theorem).

## Prediction vs Measurement

PAPER.md contains a comprehensive prediction-vs-measurement table (lines 18-27)
with honest assessments:

| Prediction | Match? | Assessment |
|-----------|--------|------------|
| Router > 60% | NO | Honestly reported, correctly identified as condition violation |
| Router > 90% on well-separated | PARTIAL | 7/24 domains at 90-100% |
| Mean B-cos in [0.01, 0.05] | YES (below) | Correctly labeled B-matrix, not DeltaW |
| Max B-cos < 0.10 | YES | Same B-matrix caveat applied |
| Per-domain matching-A < 10% | YES | 1.8-3.9%, genuine interference signal |
| Per-domain non-matching-A | N/A | code +35.7% correctly flagged as wrong-A-matrix |
| Composed PPL < 2x worst | YES | Self-flagged as "trivially: PPL distribution skew" |

The table is annotated with appropriate caveats. K753 FAIL is honestly reported with
no retroactive redefinition. The self-awareness on K755 triviality is excellent.

## Non-Blocking Observations

### 1. PAPER.md header inconsistency (cosmetic)

PAPER.md lines 5 and 13 still say "Theorem 1 (Routing Capacity)" and "Theorem 3
(Composition Bound)" in the restatement header, while the body consistently uses
"Capacity Argument 1" and "Empirical Model 3." Line 199 says "Theorem 1 corollary
confirmed" rather than "Capacity Argument 1 corollary." These are labeling
inconsistencies that should be cleaned up but are not blocking.

### 2. "B-cosine upper-bounds DeltaW-cosine" claim is imprecise

PAPER.md line 22 and several other places state that B-matrix cosine is a
"conservative upper bound" on DeltaW cosine. This is not generally true. When
A_i = A_j (same subspace), DeltaW cosine = B cosine exactly. The correct statement
is: "Theorem 2 bounds DeltaW cosine by A-skeleton coherence (~0.001-0.02), which is
independent of B-matrix cosine. Low B-matrix cosine indicates differentiated adapters;
the A-skeleton provides additional decorrelation. In the Grassmannian setting where
A_i^T A_j is small, DeltaW cosine is bounded by A-coherence regardless of B-cosine."

The conclusion (DeltaW cosine is low) is correct for the right mechanism (A-skeleton),
but the stated causal chain ("B-cosine upper-bounds DeltaW-cosine") is technically
wrong. Non-blocking because the DeltaW bound follows from Theorem 2 directly.

### 3. K755 triviality is acknowledged

The 0.51x ratio against worst single-adapter PPL is dominated by PPL distribution
skew (legal at 21.6 vs math at 4.1), not by interference properties. The paper
correctly identifies this (line 179, line 249). The per-domain matching-A degradation
(1.8-3.9%) is the real interference signal and passes easily.

### 4. Single-A composition limits interference measurement

The implementation composes B-matrices and projects through medical's A-subspace
(line 290 of run_experiment.py). This means 4 of 5 composed domains go through
the wrong A-matrix. The per-domain matching-A results (medical: +1.8%) are the
only clean interference measurement. Multi-A composition (sum of A_i @ B_i) would
be the correct test. Acknowledged in PAPER.md limitations.

### 5. Theorem 2 derivation presentation

As noted in Mathematical Soundness: the intermediate steps on MATH.md lines 123-124
describe a derivation path that produces a looser bound than what is actually claimed
on line 121. The correct tighter path exists but is not the one written. The final
result and the conclusion about A-skeleton decorrelation are both correct.

## Novelty Assessment

Scale-up experiment, not novel mechanism. The genuine novel finding is the bimodal
routing distribution and the insight that routing accuracy depends on centroid
separation Delta, not on N. This is an empirical observation correctly labeled as
such. No missing prior art.

## Macro-Scale Risks (advisory)

1. Grassmannian skeleton not validated beyond d=2560 (but capacity grows with d).
2. Single-A composition is a fundamental limitation for multi-domain queries.
3. Domain taxonomy overlap problem will worsen at N=50+ unless addressed by
   hierarchical routing or taxonomy curation.

## Verdict

**PROCEED**

The researcher applied 4/6 fixes fully and 2/6 with minor header inconsistencies.
The mathematical framework is now honest about what is proven (Theorem 2: A-skeleton
decorrelation), what is a capacity argument (Routing: JL capacity without classification
guarantee), and what is empirical (Model 3: calibrated interference coefficient).
Kill criteria are reported with exemplary honesty -- K753 FAIL with no retroactive
redefinition, K754 and K755 PASS with appropriate caveats.

For a frontier-extension experiment at provisional status, the revised presentation
meets the required standard:
- Proven result being extended is clearly stated (Finding #287, N=5)
- Mathematical gap is identified (does the framework hold at 4.8x N?)
- Status is correctly capped at provisional
- The one genuine theorem (Theorem 2) has a correct bound with QED
- All non-proofs are honestly labeled

The non-blocking observations above should be addressed in a cleanup pass (especially
the PAPER.md header inconsistencies and the "B-cosine upper-bounds DeltaW-cosine"
imprecision), but none rises to the level of requiring a revision cycle.

# Peer Review: persistence_diagram_diff (RE-REVIEW)

## Experiment Type
Guided exploration (Type 2) -- confirmed. Proven framework: Algebraic Stability
Theorem (Cohen-Steiner, Edelsbrunner, Harer 2007). Unknown being explored: empirical
bottleneck distance and topological impact of adapter composition on BitNet-2B weight
geometry.

## Hack Detector
- Fix count: 1 (single measurement using PH). No stacking. CLEAN.
- Is MATH.md a proof or a description? **Proof with QED** -- Theorem 1 and Corollaries 1-2 are valid (trivially, as one-step applications of the stability theorem).
- Metric used as evidence: Bottleneck distance + feature counts. Bottleneck distance is proven to bound topological change. Feature creation counts are descriptive (no theorem governs them). CLEAN.
- Kill criteria source: K625 derived from proof; K626 derived from Corollary 2; K627 not assessable. PARTIALLY DERIVED.

## Self-Test Audit

1. **One-sentence impossibility property:** Correct single property. PASS.
2. **Cited theorems:** Cohen-Steiner 2007 -- REAL, conditions apply. Rieck 2018 -- REAL. Minor: Garin & Tauzin citation in PAPER.md has mismatched year/arXiv ID (2020 vs 2312.10702 from 2023). Not blocking. PASS.
3. **Predicted numbers:** Now honestly annotated as vacuous (P1 tautological, P2/P4 vacuously true, P3 genuine). Self-Test item 3 explicitly says "Both predictions turned out to be vacuously true." PASS.
4. **Falsification condition:** Honest -- "The proof cannot be falsified... The experiment could show the bound is VACUOUS." PASS.
5. **Hyperparameter count:** 1 (N_SUBSAMPLE). PASS.
6. **Hack check:** Clean. PASS.

## Fix Assessment: 6 Original Issues

### Fix 1: Prediction table honesty (3/4 predictions tautological/vacuous)
**STATUS: FIXED.** PAPER.md prediction table now has an "Information Content" column
that explicitly labels P1 as TAUTOLOGICAL, P2 and P4 as VACUOUSLY TRUE, and P3 as
the only GENUINE FINDING. The "Honest assessment" paragraph below the table is clear.
MATH.md Self-Test item 3 also acknowledges the vacuousness. No overclaiming remains.

### Fix 2: Feature creation analysis
**STATUS: FIXED.** PAPER.md now has a dedicated "Feature Creation Analysis" section
with tables for H0 (+242 features across 6 modules) and H1 (+401 features across
19 modules, only 2 lost). The analysis identifies concentration in output-facing
projections and provides a plausible interpretation (composition differentiates
near-identical rows). Finding 1 is reframed around feature creation as the real
story.

### Fix 3: Finding 4 retracted ("structured alignment")
**STATUS: FIXED.** Finding 4 is now titled "Output Projections Show Lower Bottleneck
Distance But More Feature Creation (QUALIFIED)." The original "structured alignment"
interpretation is explicitly retracted. The replacement interpretation -- that
bottleneck distance is insensitive to feature creation type changes -- is coherent
and appropriately hedged.

### Fix 4: nan_to_num impact quantified
**STATUS: FIXED.** Limitation 6 now reports: "Inspection of all skeleton and adapter
files shows 0 NaN and 0 inf values out of 53,452,800 skeleton values and 54,681,600
adapter values (5 domains). The nan_to_num call has no effect on any computation."
This fully addresses the concern.

### Fix 5: Statistical testing for adapter/random ratio
**STATUS: FIXED.** PAPER.md now reports Wilcoxon signed-rank test (W=101, p=0.0002),
one-sample t-test (t=4.00, p=0.0003), and 95% CI [1.20, 1.57]. The ratio is
statistically significant at p < 0.001. The CI excludes 1.0. Appropriate tests for
N=35 paired observations.

### Fix 6: Finding #225 reframed
**STATUS: FIXED.** Finding 1 now opens with the vacuousness acknowledgment and
pivots to feature creation as the real story. The "near-lossless" framing is gone.
The paper explicitly states "This cannot be characterized as 'near-lossless' -- it
is topologically nontrivial restructuring."

## Mathematical Soundness

The proof remains trivially correct (one-step application of stability theorem).
No issues with derivations, cited theorems, or assumptions. The bound is vacuous
at current scale -- this is now honestly acknowledged throughout.

One minor note: MATH.md correctly states this is a Type 2 guided exploration, which
means the proof provides the framework and the experiment discovers empirical
unknowns. The unknowns discovered (feature creation, adapter/random ratio) are
genuinely informative even though the proof's quantitative predictions are vacuous.

## Prediction vs Measurement

PAPER.md contains the required table with honest information content annotations.

| Prediction | Status | Assessment |
|------------|--------|------------|
| P1: d_B <= max delta | Verified | Tautological (mathematical certainty) |
| P2: High-pers features survive | Verified | Vacuously true (10-100x margin) |
| P3: Adapter vs random comparison | Ratio 1.38, p=0.0002 | Genuine finding with statistical support |
| P4: Vulnerability window | Empty | Vacuously true (same as P2) |

The honest labeling is appropriate. P3 is the sole empirical contribution from the
proof's predictions. The feature creation analysis is an additional empirical
contribution outside the proof's scope.

## Novelty Assessment

Unchanged from original review:
- Stability theorem application: textbook, not novel
- Weight-space PH for NNs: Rieck et al. 2018
- Application to adapter composition: novel but predictable outcome
- Genuine contributions: (a) statistical confirmation that adapters cause more
  topological change than random (P3), (b) feature creation analysis showing
  composition differentiates near-identical rows and creates cyclic structure

## Remaining Concerns (Non-Blocking)

1. **Subsample bias from np.linspace.** Original review noted deterministic evenly-
   spaced subsampling could miss clustered structure. Not fixed, but acknowledged
   in Limitation 1. Non-blocking for a micro experiment.

2. **Citation error.** Garin & Tauzin year/arXiv mismatch persists. Cosmetic.

3. **Feature creation interpretation is speculative.** The claim that created H1 loops
   "could be the topological signature of adapter specialization" is untested. This
   is flagged as speculation, not presented as a finding. Acceptable.

4. **Finding status.** The experiment is marked SUPPORTED. Given that the only
   genuine prediction (P3) is confirmed and the feature creation analysis is
   descriptive (not predicted by the proof), SUPPORTED is the correct status for
   a Type 2 guided exploration.

## Macro-Scale Risks (advisory)

1. The vacuous bound provides no early warning as adapter scale increases. By the
   time features enter the vulnerability window, the perturbation is ~10x larger
   than current.
2. Euclidean Rips may not capture functionally relevant structure (cosine distance
   is standard for transformer representations).
3. Feature creation at scale could indicate composition is fundamentally
   restructuring weight geometry in ways that affect generation quality.

## Verdict

**PROCEED**

All 6 issues from the original review have been properly addressed:
- Prediction table is honest about information content
- Feature creation is analyzed with dedicated section and tables
- Finding 4 "structured alignment" is retracted and reframed
- nan_to_num impact is quantified as zero
- Statistical testing confirms adapter/random ratio significance
- "Near-lossless" framing is replaced with honest "topologically nontrivial restructuring"

The paper is now an honest Type 2 guided exploration that:
(a) confirms the stability theorem holds (as it must) with a vacuous bound,
(b) establishes that adapters cause statistically significantly more topological
    change than random perturbations (p < 0.001),
(c) discovers substantial feature creation in output-facing projections.

The finding status of SUPPORTED is appropriate. The feature creation story is the
genuine contribution and should drive follow-up experiments (bridge extraction
focusing on created features, not lost ones).

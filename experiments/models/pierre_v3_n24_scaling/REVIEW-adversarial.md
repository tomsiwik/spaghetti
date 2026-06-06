# Peer Review: Pierre v3 N=24 Scaling (RE-REVIEW)

This is a re-review after 6 fixes were requested in the first adversarial review.
The original verdict was REVISE. This review evaluates whether the fixes were
applied correctly and whether the revised framing is scientifically honest.

## Experiment Type
Frontier extension (correctly identified). Proven framework at N=5 (Findings
#276, #273, #287) extended to N=24. MATH.md states the proven results being
extended and identifies the mathematical gaps. Structurally correct for Type 3.
Finding status is capped at provisional per project rules.

## Fix Verification

### Fix 1: Status changed to PROVISIONAL
**APPLIED.** PAPER.md line 33 reads "Verdict: PROVISIONAL." with clear
justification: "Two of three kill criteria fail (K721, K722)." No hedging.
The word "SUPPORTED" does not appear as a status claim anywhere.

### Fix 2: K721 failure reported honestly
**APPLIED.** PAPER.md Section "Analysis: K721 Failure" (lines 148-170):
- Opens with "K721 fails." (unambiguous)
- States "the genuine/slice distinction was not part of the original hypothesis"
- Labels the genuine/slice analysis as "Post-hoc observation (hypothesis, not
  evidence)"
- Explicitly calls out "moving to 'genuine only' is goalpost-moving"
- Proposes a follow-up experiment to test the hypothesis properly

This is a substantially improved framing. The paper no longer claims the
genuine/slice analysis rescues the kill criterion.

### Fix 3: Null-space ablation required
**APPLIED.** PAPER.md Section "Analysis: K722 Failure" (lines 174-196):
- States the observation "requires ablation to confirm" (line 185)
- Lists three competing explanations: (a) A-orthogonality alone, (b) partial
  null-space (36.1% is not 0%), (c) both
- Explicitly says "the current experiment cannot disentangle" these
- Requires a "follow-up: ablation experiment that tests composition quality
  WITH orthogonal A-matrices but WITHOUT null-space projection"
- Does NOT recommend dropping null-space projection

This correctly avoids the premature recommendation from the original version.

### Fix 4: C2 uniformity assumption corrected
**APPLIED.** MATH.md Section C2 (lines 88-101) now includes a detailed caveat:
- "The bound above assumes the test gradient g is drawn uniformly from R^d"
- "In practice, all adapters adapt the same base model on text data, so the
  test gradient is directionally correlated with the prior adapter subspaces"
- "preservation can degrade well below the theoretical bound even when
  rank(V) = 368 (maximal)"
- "the uniform gradient assumption fails and the bound becomes optimistic"

The incorrect claim from the original ("if adapter subspaces overlap, rank < 368,
and preservation IMPROVES") has been replaced with the correct directional
correlation explanation. This is a genuine mathematical improvement.

### Fix 5: [0.7, 0.3] compose weight declared
**APPLIED.** MATH.md Self-Test item 5 (lines 211-215) now reads:
"Compose weights [0.7, 0.3] for top-2 NRE composition (hardcoded, not derived
from router confidence -- this is an undeclared hyperparameter that should be
derived from router scores at macro scale)."

This is honest and actionable.

### Fix 6: n_cal=30 reconciled
**APPLIED.** MATH.md line 63 now reads "At n_cal=30, the centroid estimation
error is ~1/sqrt(30) = 18.3% of the intra-class variance." The code uses
N_CAL = 30 (line 74). These are now consistent. The original review flagged
n_cal=50 vs n_cal=30; the text now correctly uses 30.

**All 6 fixes verified as applied.**

## Hack Detector
- Fix count: 0 new mechanisms. Pure scaling test of 3 existing components
  (ridge router, null-space SVD, NRE compose). No hacks added.
- Is MATH.md a proof or a description? **Description with predictions derived
  from existing results.** No Theorem/Proof/QED block. This is acceptable for
  frontier extension -- the point is to test whether proven-at-N=5 bounds hold
  at N=24, not to prove new theorems.
- Metric used as evidence: PPL ratio (K723, behavioral), routing accuracy
  (K721, proxy), gradient preservation (K722, proxy). The paper correctly treats
  PPL as the behavioral outcome and the other two as mechanistic intermediaries.
- Kill criteria source: K721/K722 thresholds (50% each) are described as having
  a safety margin from theoretical predictions. K723 (2x worst single PPL) is
  better grounded in the composition quality prediction.

## Self-Test Audit

1. **One mathematical property:** Positive definiteness of X^TX + lambda*I for
   any lambda > 0. Correct, genuinely one property. PASS.

2. **Prior theorems:** JL-lemma (1984) -- real, correctly cited but see
   Mathematical Soundness below for applicability nuance. Ridge regression PD
   Hessian -- standard, correct. Null-space SVD rank bound -- standard linear
   algebra. PASS.

3. **Specific numbers:** Router >70% overall, >85% genuine; preservation ~85.6%;
   composed PPL <1.5x worst single. Specific and falsifiable. PASS.

4. **Falsification:** Targets the mathematical framework (d insufficient for
   24 centroids, adapter orthogonality breaking). Not just "metrics don't hit
   threshold." PASS.

5. **Hyperparameters:** lambda=1.0 (from Finding #276), top_k SVD,
   compose weights [0.7, 0.3] (now declared as undeclared hyperparameter). PASS.

6. **Hack check:** "No. Direct extension." Accurate -- no new mechanisms. PASS.

## Mathematical Soundness

### C1: Ridge Router
The JL-lemma citation remains **directionally correct but conceptually
imprecise**, as noted in the first review. JL guarantees that 24 points CAN
be embedded in O(log 24/eps^2) ~ O(100) dimensions with distance preservation.
It does NOT guarantee that the base model's hidden representations produce
separable domain centroids. The correct foundation for the routing prediction
is Finding #276 (96% at N=5) plus the capacity argument that d=2560 >> N=24,
not JL per se. This is a minor framing issue, not a mathematical error. JL
provides an upper bound on how many points the space can separate, which is
useful context even if it does not prove that these particular points separate.

### C2: Null-Space (IMPROVED)
The uniformity assumption is now properly flagged. The theoretical bound
(85.6%) is presented as an upper bound under an idealization, with the caveat
that directional correlation causes the actual preservation to fall well below
this bound. The paper correctly identifies that the gap (85.6% predicted vs
36.1% measured = 2.4x) is the central informative failure of the frontier
extension, not an error in the proof framework.

The rank bound calculation remains correct: min(23*16, 2560) = 368, confirmed
by all effective_rank measurements in results.json being exactly 368.

### C3: Composition
The NRE composition prediction (<1.5x worst single) is met with large margin
(0.87x). The prediction was conservative, which is appropriate for frontier
extension. The observation that misrouting is PPL-benign is correctly presented
as empirical finding, not mathematical prediction.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table (lines 13-20).

| Prediction | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| Overall router accuracy | >70% | 37.6% | NO (1.9x miss) |
| Genuine domain accuracy | >85% | 82.6% | CLOSE (3% miss) |
| Null-space preservation | ~85.6% | 36.1% | NO (2.4x miss) |
| Composed PPL ratio | <1.5x | 0.87x | YES (large margin) |
| Centroid separation | >0.1 L2 | 9x random | YES (directional) |
| B-matrix orthogonality | mean cos <0.05 | 0.024 | YES |

Two of four quantitative predictions fail significantly. For frontier extension,
this is informative and the paper frames it appropriately as "the failures
reveal the true scaling bottleneck." The paper does not claim these failures
are successes.

**Minor arithmetic issue:** PAPER.md line 16 claims genuine accuracy "excl.
science: 93.4%." My computation from results.json: genuine domains excluding
science = (46+41+50+50+47+50)/300 = 284/300 = 94.7%, not 93.4%. This is a
~1.3pp discrepancy. Not blocking but should be corrected.

## Kill Criteria Honesty

### K721 (Router accuracy >= 50%): FAIL at 37.6%
Reported honestly as FAIL. Post-hoc analysis is labeled as hypothesis, not
evidence. The paper explicitly warns against goalpost-moving. The proposed
follow-up (test N=7 curated genuine domains) is appropriate.

### K722 (Null-space preservation >= 50%): FAIL at 36.1%
Reported honestly as FAIL. The theoretical bound's uniformity assumption is
correctly identified as the source of the gap. The paper does NOT recommend
dropping null-space projection (a fix from the first review). The competing
explanations (A-orthogonality alone vs partial null-space vs both) are
presented without premature resolution, with an explicit call for ablation.

### K723 (Composed PPL <= 2x worst single): PASS at 0.87x
Unambiguous PASS. Max composed PPL 22.854 vs threshold 52.602.

**Nuance correctly preserved:** The paper notes that creative_writing's
single-adapter PPL (26.301) is WORSE than base model PPL (23.274), meaning
the adapter is actively harmful for that domain. Misrouting away from it
improves PPL. This does not invalidate K723 (the kill criterion is about
composed quality, not routing correctness), but the paper's observation
that "3 of 4 misrouted domains improve PPL" should be understood in this
context.

## Remaining Issues (Non-Blocking)

1. **Genuine accuracy arithmetic:** 93.4% should be 94.7% (minor, see above).

2. **Creative_writing adapter quality:** The single-adapter PPL for
   creative_writing (26.30) exceeds base PPL (23.27). This means the adapter
   makes creative_writing WORSE. The K723 PASS benefits from misrouting AWAY
   from a broken adapter. This is noted in the paper but could be more
   prominently flagged -- the K723 margin would shrink if all adapters actually
   improved their target domain.

3. **JL-lemma as decoration:** As noted above, the JL argument provides
   capacity bounds (the space CAN fit 24 separated points) but does not prove
   separability of the actual domain centroids. The real prior is Finding #276.
   This is a framing issue, not an error.

4. **Compose weights [0.7, 0.3]:** Declared as undeclared hyperparameter (good),
   but the impact on K723 results is not explored. Would [0.5, 0.5] or
   [0.9, 0.1] change the PASS? This is a sensitivity gap that should be
   addressed in macro scale.

5. **Single-sample routing:** Top-1 routing uses only the first validation
   sample (line 491-492 of run_experiment.py). This is acknowledged in
   Limitations (line 212) but means the routing accuracy could improve
   significantly with majority-vote routing. The kill criterion tests the
   current implementation, not the mechanism's ceiling.

## NotebookLM Findings
Skipped -- not available in this session.

## Novelty Assessment
This is a scaling experiment, not a novelty claim. The components are from prior
work (DUME ridge router, Brainstacks null-space, NRE composition). The value
is in discovering where the N=5 framework breaks at N=24.

Key informative observations (correctly framed as provisional):
1. B-matrix directional correlation breaks the uniform gradient assumption in
   null-space bounds -- rank stays maximal but preservation drops 2.4x
2. Within-cluster misrouting is PPL-benign at N=24 -- confirms Finding #287
3. Ridge regression separates genuine domains (6/7 > 80%) but not arbitrary
   slices (17 domains at 19.1%)

## Macro-Scale Risks (advisory)
1. Genuine/slice domain distinction needs rigorous definition BEFORE deployment
2. Null-space preservation at 36.1% may become problematic for continual
   learning (adding adapters that need protection from existing ones)
3. Compose weights [0.7, 0.3] need derivation from router confidence at scale
4. Single-sample routing needs to be replaced with multi-sample majority vote

## Verdict

**PROCEED**

All 6 requested fixes were applied correctly. The paper now:
- Reports K721 and K722 as genuine FAILs without disguise
- Labels post-hoc analysis (genuine/slice, A-orthogonality attribution) as
  hypotheses requiring follow-up experiments, not evidence
- Correctly identifies the uniformity assumption violation in the null-space
  bound as the mathematical root cause of the K722 gap
- Does not recommend architectural changes (dropping null-space) without
  ablation evidence
- Caps status at PROVISIONAL, appropriate for frontier extension with 2/3
  kill criteria failing

The experiment produced genuinely informative frontier observations:
- The null-space bound's uniformity assumption breaks at N=24 even though rank
  stays maximal -- this is a real mathematical insight about B-matrix directional
  correlation
- Composition quality (K723) survives despite routing and null-space failures,
  suggesting A-matrix orthogonality may be the primary interference mechanism
- The genuine/slice accuracy pattern generates a testable hypothesis for the
  next experiment

These are recorded as provisional findings with appropriate epistemic humility.
The paper no longer overclaims, and the follow-up experiments (N=7 curated
routing test, null-space ablation) are clearly identified as required before
any findings can be upgraded.

**Non-blocking fixes for completeness:**
1. Correct "93.4%" to "94.7%" in PAPER.md line 16 (arithmetic error)

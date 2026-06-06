# Peer Review: DC-Merge SVD Energy Smoothing

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (SVD energy smoothing before summation). No stacking. Clean.
- Is MATH.md a proof or a description? Mixed. Theorems 1 and 3 are trivially correct proofs. Theorem 2 has a gap (see below). Theorem 4 is a proof sketch citing an external paper. The substantive claims (P2-P4 about composed Gini and PPL) are predictions without proof, which is appropriate for Type 2 guided exploration.
- Metric used as evidence: Gini coefficient (K700) and perplexity (K699). Gini is directly derived from the proof. PPL is acknowledged as a weak proxy (r=0.08 with task performance, per project finding).
- Kill criteria source: K700 (>30% Gini reduction) is derived from P2, which itself is a heuristic prediction, not a theorem bound. K699 (PPL improvement) is a behavioral check. Neither is strictly derived from a tight proof bound, but for Type 2 exploration this is acceptable.

## Self-Test Audit

1. **One-sentence impossibility property:** "Energy equalization via singular value smoothing ensures no task vector's singular directions dominate the composed result, making energy imbalance impossible by construction." -- This is about individual deltas, not the composed result. The experiment itself DISPROVES this claim: individual equalization does not prevent composed energy imbalance (Gini only dropped 18.5%, not to zero). The self-test answer is misleading. FLAG.

2. **Cited theorems:** DC-Merge Eq. 12 (average smoothing), Appendix E.4 (linear smoothing). The paper is arXiv:2603.06242 (CVPR 2026). I cannot verify the citation is accurate since it is post-training-cutoff, but the implementation in `dc_merge/src/model.py` is consistent with the described algorithm. ACCEPTABLE with caveat.

3. **Predicted numbers:** P1: Gini->0 (average, individual). P2: Composed Gini reduction >30%. P3: Composed Gini reduction >15%. P4: PPL improvement 1-5%. These are specific and falsifiable. PASS.

4. **Falsification condition:** "If average smoothing produces Gini != 0 for individual deltas -- this would violate basic arithmetic. This cannot happen." This is a tautology -- it cannot be falsified. The real falsification conditions (composed Gini, PPL) are listed as experiment failures, not proof failures. For a Type 2 exploration this is borderline acceptable: the proof (individual equalization) is trivially correct; the unknown is whether individual equalization transfers to composed equalization. MARGINAL.

5. **Hyperparameter count:** 1 (strategy choice) + 1 for linear (rho). Acknowledged. PASS.

6. **Hack check:** "No. This is a standalone composition method." Accurate. PASS.

## Mathematical Soundness

**Theorem 1 (Average Smoothing G=0).** Trivially correct. Setting all values equal to the mean makes the Gini numerator zero. No issues.

**Theorem 2 (Linear Smoothing Bound).** The claim is G(S_bar) <= (rho-1)/(rho+1). The proof asserts this is the Gini of a linearly decreasing discrete distribution. Let me verify:

For a linearly decreasing sequence a_j = rho - (rho-1)(j-1)/(r-1), j=1..r:
- a_1 = rho, a_r = 1
- The Gini for an arithmetic sequence with endpoints [a, b] (a >= b > 0) is known to be (a-b)/(3*(a+b)) for the continuous limit, and for discrete n points it equals (r+1)/(3r) * (a-b)/(a+b).

For large r the discrete Gini approaches (a-b)/(3(a+b)) = (rho-1)/(3(rho+1)), which is TIGHTER than the stated bound of (rho-1)/(rho+1). The stated bound is correct (it is an upper bound), but it is loose by a factor of 3. This does not invalidate the theorem -- it just means the bound is not tight. For rho=5, the actual Gini is approximately (5-1)/(3*(5+1)) = 4/18 = 0.222, not the stated 4/6 = 0.667. The proof's bound is vacuous in practice.

However, since individual B-matrix Gini is already 0.27-0.29 and the bound says G <= 0.667, the theorem is useless -- it does not guarantee improvement. The experiment correctly found linear smoothing barely helps (1.8% composed Gini reduction), consistent with the loose bound.

**Theorem 3 (Energy Conservation).** Trivially correct. Both strategies preserve the sum of singular values by construction.

**Theorem 4 (Cover Space Directional Alignment).** This is a proof sketch citing DC-Merge Section 3.2. It is not verified in the experiment -- the cover space projection was NOT applied (acknowledged in Limitations item 4). This theorem is irrelevant to the actual experiment. FLAG: citing a theorem you do not use is misleading. The experiment applies energy smoothing + raw summation, not the full DC-Merge algorithm.

**The critical gap: individual-to-composed transfer.** The proofs guarantee properties of INDIVIDUAL task vectors after smoothing (G=0, energy preserved). The predictions about COMPOSED behavior (P2-P4) are CONJECTURES, not proven. P2 says "Since our Grassmannian A-matrices are orthogonal (mean pairwise cos=0.026), the composed sum of equalized deltas will have much more uniform energy distribution." This is an intuitive argument, not a derivation. And the experiment refutes it: 18.5% reduction instead of >30%.

The paper itself identifies why: the composed Gini is dominated by cross-domain scale differences (medical scale=20.0 vs finance scale=1.0), not within-domain spectral shape. This is a genuine discovery and the most valuable part of the experiment.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Assessment:

| Prediction | Measured | Verdict |
|---|---|---|
| P1: Individual Gini -> 0 | 0 (by construction) | PASS (trivial) |
| P2: Composed Gini reduction >30% | 18.5% | FAIL |
| P3: Composed Gini reduction >15% | 1.8% | FAIL |
| P4: PPL improvement 1-5% | 0.99% (linear) | MARGINAL (barely meets lower bound with rounding) |
| P5: DirSim preserved | +12% (0.0116->0.0130) | PASS |

2 of 5 predictions fail. The passing ones are either trivial (P1) or weak (P5). P4 is at the ragged edge -- 0.99% on N=20 per domain is not statistically meaningful.

## NotebookLM Findings

NotebookLM deep review was not executed due to the tool requiring interactive authentication. Review conducted through direct document analysis instead.

## Novelty Assessment

DC-Merge is an existing method (arXiv:2603.06242). The novelty here is applying it to ternary LoRA adapters with Grassmannian A-matrices. This is a straightforward application, not a new method.

The genuine novel finding is the diagnosis: **cross-domain scale imbalance dominates composed spectral shape more than within-domain SV distribution**. This is a useful negative result that redirects effort toward scale normalization rather than energy smoothing. It is worth recording.

Prior art check: The scales (medical=20, code=20, math=20, legal=4, finance=1) are from Finding #249 (optimal scales). The 20:1 ratio is enormous and obviously would dominate any composed spectral analysis. One could argue this should have been predicted analytically before running the experiment: if one domain contributes 20x the Frobenius norm to the sum, its singular directions will naturally dominate regardless of within-domain smoothing.

## Macro-Scale Risks (advisory)

1. The 0.99% PPL improvement is within noise at N=20. At macro scale with proper evaluation (N>1000), this could vanish entirely.
2. The scale imbalance finding (20:1) suggests the real experiment should be Frobenius-norm equalization before composition, not energy smoothing. This is a simpler and more principled approach.
3. The cover space projection (Theorem 4) was not tested. If the full DC-Merge algorithm (energy smoothing + cover space) were applied, results could differ substantially.

## Specific Issues

1. **Statistical significance of K699.** The PPL improvement is 0.99% on 100 texts (20 per domain, max 256 tokens each). No confidence intervals, no bootstrap, no significance test. At this sample size, a 0.99% change is likely within sampling noise. The PAPER.md acknowledges this in Limitations item 1 but still reports K699 as PASS. This is questionable.

2. **Theorem 2 bound is vacuous.** The stated G <= (rho-1)/(rho+1) = 0.667 for rho=5 is 3x looser than the actual Gini of a linear distribution (~0.222). This means the theorem provides no useful guarantee -- individual B-matrices already have Gini 0.27-0.29, below the bound's guaranteed region. The linear smoothing strategy was therefore never mathematically expected to help, making P3 (>15% reduction) poorly grounded.

3. **Incomplete application of DC-Merge.** Only energy smoothing was applied, not the full algorithm (cover space projection + block-diagonal mask). Claiming to test "DC-Merge" while omitting the directional alignment component is incomplete. The experiment title should reflect this.

4. **Scale confound was foreseeable.** With scales of {20, 20, 20, 4, 1}, the medical/code/math domains contribute ~93% of total Frobenius energy to the composed delta. Smoothing individual spectral shapes while leaving these 20:1 scale ratios intact is obviously insufficient. This could have been predicted from the scale values alone, without running the experiment.

5. **DirSim measurement is on a single layer.** Phase 3 measures DirSim only on layer 15, self_attn.q_proj. This is not representative. The claim "DirSim preserved" is based on a single data point per pair.

## Verdict

**PROCEED** (as weakly supported finding, with revisions to the write-up)

The experiment is honest about its failures and identifies a genuine insight: cross-domain scale imbalance is the dominant factor in composed spectral pathology, not within-domain singular value shape. This redirects future work productively. The finding status should be "provisional" rather than "supported" given that:
- The primary prediction (P2, >30% Gini reduction) failed
- The PPL improvement is not statistically validated
- The positive K699 result rests on a 0.99% change at N=20

Revisions required before the finding is recorded:

1. **Downgrade finding status to "provisional."** Two of five predictions failed. The PPL result is within noise. "Supported" overstates the evidence.
2. **Remove or relabel Theorem 4.** It was not tested. Either label it as "motivation for future work" or remove it. Citing unused theorems inflates apparent rigor.
3. **Fix Self-Test item 1.** The claimed impossibility property (energy imbalance impossible by construction) is refuted by the experiment's own data. Replace with the actual insight: "Individual spectral equalization is insufficient when cross-domain scale ratios dominate composed spectral structure."
4. **Note Theorem 2 bound is vacuous.** State explicitly that G <= 0.667 for rho=5 provides no guarantee of improvement when individual Gini is already 0.27-0.29. The actual Gini of a discrete linear distribution is ~0.222, which is already below the individual adapter Gini.
5. **Add a note on foreseeable confound.** The 20:1 scale ratio should have been identified as a potential confound before running the experiment. Acknowledge this in a lessons-learned section.
6. **Record the actionable finding separately.** The real value is the diagnosis: "Frobenius-norm scale equalization across domains is the correct target, not within-domain spectral smoothing." This should be the primary finding, not the DC-Merge PPL result.

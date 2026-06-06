# Peer Review: exp_self_growing_toy

## Experiment Type
Frontier extension (extending ReLoRA periodic merge to sequential multi-domain promotion from random init)

## Hack Detector
- Fix count: 1 (single mechanism: promote delta into base). Clean.
- Is MATH.md a proof or a description? **Description dressed in equations.** "Claim 1" through "Claim 3" are arguments, not proofs. "By construction" and "by Davis-Kahan" are invoked but never bound numerically. No Theorem/Proof/QED block exists anywhere. This is acceptable for a frontier extension but must be noted.
- Metric used as evidence: mean cross-entropy loss across 5 toy domains. Adequate for mechanism test.
- Kill criteria source: K841 derived from Claim 1 (monotonic improvement). K842 derived from Claim 2 (richer base = faster training). K843 is an arbitrary threshold (3.0x) but reasonable for a frontier extension.

## Self-Test Audit
1. One-sentence impossibility property: States "sequential promotion with magnitude-controlled deltas is equivalent to gradient accumulation (ReLoRA)." This is not an impossibility property; it is a claimed equivalence. Weak but passable for frontier extension.
2. Cited theorems: ReLoRA (Lialin et al. 2307.05695) is real. Davis-Kahan is real. However, ReLoRA's result assumes continued training on the SAME distribution (language modeling), not sequential single-domain training. The citation is valid but the application conditions do not hold -- see Mathematical Soundness below.
3. Specific numbers: P1-P5 are present and falsifiable. Adequate.
4. Falsification condition: "If mean loss degrades after ANY promotion, the gradient-accumulation equivalence breaks at this scale." This correctly targets the claim.
5. Hyperparameter count: 2 (scale=2.0, SVD rank=4). Acknowledged and matched to LoRA convention. Fine.
6. Hack check: Single mechanism, no stacking. Clean.

## Mathematical Soundness

**The ReLoRA analogy is structurally invalid, and PAPER.md correctly discovers this.**

ReLoRA proves: periodic LoRA merge during continued training on a SINGLE data stream is equivalent to gradient accumulation. The key condition is that each merge cycle trains on data drawn from the same distribution. The rank-r projection at each cycle captures the next r-dimensional slice of the gradient, and over K cycles these accumulate to span up to rank K*r of the full gradient.

This experiment applies ReLoRA's conclusion to a fundamentally different setup:
- Each merge cycle trains on a DIFFERENT domain (arithmetic, reverse, repeat, sort, parity)
- The "gradient" being projected is toward a different objective each time
- There is no single loss function whose gradient is being accumulated

This means Claim 3 ("the gap narrows by ReLoRA convergence") has no mathematical backing. The grown model and the jointly-trained model are solving different optimization problems in different orders. The 3.23x gap is not a "bounded but nonzero" deviation from ReLoRA convergence -- it is a different regime entirely.

Claim 1 is correct in its weak form (each promotion improves its own domain) and the data confirms this. But the "by Davis-Kahan" bound on cross-domain interference is never computed. The MATH.md says "O(||delta_i||_op / gap_j)" but never estimates either quantity. The measured interference is 0.3-0.7 loss units per promotion -- is this consistent with Davis-Kahan? Unknown, because the bound was never evaluated.

Claim 2 (later adapters benefit from richer base) is supported by the loss@50 trajectory (4.62 -> 3.98), which is genuine evidence that the enriched base provides better features. This is the one solid positive finding.

**Verdict on math:** The framework is honest about being a frontier extension and the Claims are appropriately hedged. The ReLoRA analogy breaks in a well-understood way. The experiment correctly identifies WHERE it breaks.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Assessment:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| P1: Each promotion improves its domain | All 5 improved (+0.71 to +2.00) | Confirmed |
| P2: Mean loss improves monotonically | Promotion 3 degrades by -0.074 | Refuted |
| P3: 5th adapter no slower than 1st | 0.94x (faster) | Confirmed |
| P4: Grown/baseline < 3.0x | 3.23x | Refuted (marginal) |
| P5: >50% improvement over random init | 19.8% | Refuted (badly) |

2/5 confirmed, 3/5 refuted. Kill is justified.

## Is the Kill Justified?

**Yes.** The kill is clean and well-supported.

K841 (mean degradation): Promotion 3 (repeat) causes -0.074 mean degradation. The tolerance threshold is -0.05. The degradation is real: repeat's delta has the largest norm (0.7655), and it pushes reverse from 3.83 to 4.43 and parity from 4.68 to 5.10. This is textbook catastrophic interference from overlapping random subspaces.

K843 (3.23x ratio): Marginal fail at 3.23x vs 3.0x threshold. But the deeper issue is not the threshold -- it is P5. The grown model improves only 19.8% over random init while the baseline improves 75.2%. That is a 3.8x gap in improvement rate. The mechanism captures only 26% of the knowledge that joint training captures (0.988 loss units vs 3.746 loss units of improvement). This is not a "needs more tuning" situation; it is a structural limitation.

**One concern:** K841 uses a -0.05 tolerance for mean degradation, which is not derived from the proof. The proof (Claim 1) says "provided ||delta_i|| is controlled," but never specifies what "controlled" means quantitatively. A stricter reading of the prediction (P2: "mean_improvement >= 0 for all 5") would fail at promotion 3 without any tolerance threshold. The tolerance actually softens the kill criterion; removing it would make the kill MORE justified, not less.

## Is the Root Cause Analysis Correct?

**Partially.** The PAPER.md identifies three root causes. I assess each:

**1. "Catastrophic interference from sequential rank-4 updates with random A-matrices."**
Correct. The sawtooth pattern is definitive: each promotion improves its own domain by ~1.0 but damages 2-3 others by 0.3-0.7. The last-promoted domain (parity: 2.71) retains the most benefit because nothing overwrites it. The first-promoted domain (arithmetic: 4.74) retains almost nothing (only 0.29 improvement from init of 5.03). This is sequential overwrite, not accumulation.

**2. "No interference protection (unlike composition with Grassmannian A)."**
Correct framing but incomplete analysis. The composition pipeline uses Grassmannian A-matrices to guarantee ||Delta_i^T Delta_j|| is small. Here, each adapter gets fresh random A-matrices. But there is a subtlety the PAPER.md misses: even WITH Grassmannian A-matrices for each adapter, sequential promotion would still suffer from the problem that each promotion REPLACES the base weights. In composition, the adapters are ADDED at inference time without modifying the base. In promotion, each delta is permanently absorbed into W. Even if deltas are orthogonal, the modified base changes the optimization landscape for the next adapter. Grassmannian A-matrices would reduce but not eliminate this problem.

**3. "Random init is a terrible starting point."**
Correct but this is the known limitation, not a root cause. ReLoRA was validated on pre-trained models. The experiment was designed to test whether the mechanism extends to random init. It does not.

## Could Grassmannian A Fix This?

The PAPER.md suggests this as "What would make it work" item 2. My assessment: **it would help but not solve the fundamental problem.**

Grassmannian A-matrices guarantee that Delta_i and Delta_j occupy nearly orthogonal subspaces. This would reduce cross-domain interference from the measured 0.3-0.7 loss degradation to something closer to the composition-verified 17x decorrelation. But there are two issues:

(a) In composition, the base is FROZEN. Multiple Delta_i are added at inference but the base never changes. In promotion, the base changes after each step. Even with orthogonal deltas, the changed base means the NEXT adapter trains on a different landscape. The Grassmannian guarantee applies to the delta subspaces, not to the interaction of deltas with the shifted base.

(b) At d=64 with rank=4 and K=5, the 5 adapters occupy 5*4/64 = 31% of the weight space. Grassmannian packing at this density still allows some overlap (Welch bound gives minimum coherence > 0 for any packing at 31% density). At d=2560 it would be 0.78%, which is much safer.

**Recommendation:** If someone wants to pursue this, the experiment would need to be "self-growing with Grassmannian promotion from a pre-trained base." That eliminates both root causes simultaneously.

## Novelty Assessment

The experiment is a clean application of ReLoRA to a new regime (sequential multi-domain promotion from random init). No prior work specifically tests this combination. The negative result is genuinely informative: it establishes that ReLoRA's gradient-accumulation equivalence does not extend to sequential domain training.

The 500-step ablation (3.39x, worse than 300-step 3.23x) is a useful data point confirming that larger deltas cause more interference, not less.

## Macro-Scale Risks (advisory)

At d=2560, the rank-4 subspace fraction drops from 6.25% to 0.16%, which should dramatically reduce interference. The sawtooth pattern may flatten to near-monotonic improvement. This is worth testing if someone pursues Option B (pre-trained base + Grassmannian promotion).

However, the 26% knowledge-capture rate (vs joint training) may not improve with scale. The fundamental limitation is that rank-4 sequential projections of different objectives do not accumulate like rank-4 sequential projections of the same objective. This is a mathematical gap, not a scale gap.

## Verdict

**KILL CONFIRMED.**

The kill is well-executed, the root cause analysis is substantially correct, and the experiment cleanly falsifies the ReLoRA-extension hypothesis for sequential multi-domain promotion from random init.

Minor critiques:
1. MATH.md contains descriptions, not proofs (no Theorem/Proof/QED). Acceptable for frontier extension but should be noted.
2. The Davis-Kahan bound is invoked but never numerically evaluated. The measured interference should have been compared to the predicted bound.
3. K841's -0.05 tolerance is not derived from the proof. The raw prediction (mean_improvement >= 0) would have made the kill even cleaner.

The learning is captured correctly in the current_direction.md: Pierre's value is in composition (proven with Grassmannian structure), not in growing the base from scratch. The Grassmannian-promotion idea (Option A) would need a pre-trained base (Option B) to work, making them a combined experiment rather than alternatives.

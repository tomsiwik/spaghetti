# Peer Review: M2P Bottleneck Width (exp_m2p_bottleneck_width)

## Experiment Type

Verification (Type 1). MATH.md claims to have a Theorem/Proof/QED block (Theorem 1, "JL Width Bound"). Kill criteria derived from the theorem's corollaries.

## Hack Detector

- Fix count: 1 (sweep d_M2P only, no new losses/mechanisms). Clean.
- Is MATH.md a proof or a description? **Description dressed in equations.** See Section C below for detailed analysis.
- Metric used as evidence: quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss). Not proven to be monotone in projection distortion. The entire chain proof -> prediction -> kill criterion depends on an unproven link between JL distortion and this metric.
- Kill criteria source: Derived from proof predictions via Corollary 2, but Corollary 2 itself is informal (see below).

## Self-Test Audit

All 6 fields are present and filled. Checking each:

1. **One-sentence impossibility property:** "At d_M2P >= d_JL = 138, the JL lemma guarantees that all N=5 domain representations are preserved with <= 10% pairwise distance distortion, making domain confusion geometrically impossible below epsilon." This is a correct statement of what JL guarantees about DISTORTION. But it silently equates "domain confusion" with "generation quality gap." JL makes pairwise distance collapse impossible; it says nothing about whether the M2P can learn to exploit those distances. **Partially evasive.**

2. **Cited theorems:** Johnson-Lindenstrauss (1984), Dasgupta-Gupta (1999). Both real, correctly cited. The Dasgupta-Gupta exact bound form d = (4 ln N)/(epsilon^2/2 - epsilon^3/3) is from their Theorem 2.1. Applied correctly to get d_JL=138. **PASS.**

3. **Predicted numbers:** d_JL=138, quality(d=128) >= 97%, quality(128) > quality(64), |quality(256)-quality(128)| < 2%. Specific and falsifiable. **PASS.**

4. **Falsification condition:** "If quality at d=128 is NOT higher than at d=64 (K871 FAIL), the JL mechanism is falsified for this setting." This correctly targets the applicability of JL to this setting (not JL itself). **PASS.**

5. **Hyperparameter count:** Claims 0. Correct -- d_M2P sweep values are derived from JL bound. **PASS.**

6. **Hack check:** Claims this is a pure ablation, no new mechanisms. **PASS.**

Self-Test verdict: 5/6 pass. Item 1 is partially evasive but not a blank.

## Mathematical Soundness

### What is proven correctly

- The JL bound computation d_JL(N=5, epsilon=0.1) = 138 is arithmetically correct. Verified:
  numerator = 4 * ln(5) = 4 * 1.6094 = 6.4378.
  denominator = 0.01/2 - 0.001/3 = 0.005 - 0.000333 = 0.004667.
  d_JL = 6.4378 / 0.004667 = 137.97. Correct.

- Theorem 1 Part (a) is a standard restatement of JL. The union bound argument is correct. The simplification from epsilon^3/6 to epsilon^2/8 for epsilon < 1/2 is valid (0.1^3/6 = 0.000167, 0.1^2/8 = 0.00125; indeed 0.000167 < 0.00125). Correct.

- Part (b) follows from (a) by standard linear separability of well-separated point sets. Correct.

- Part (c) is the standard JL lower bound argument. Correct.

- The Grassmannian A-matrix orthogonality verification (cos_max = 7.45e-09) is a valid control confirming that adapter interference is not a confound.

### CRITICAL FLAW: The gap between distortion and quality

**Corollary 2 is not a corollary.** It is an informal argument dressed as a mathematical result. The text states:

> "The M2P generation quality (measured as fraction of SFT improvement recovered) is bounded by the fidelity of the hidden-state representation."

This claim is **never proven.** It is stated as obvious, but it is not. Here is the logical chain the proof attempts:

1. JL lemma: d >= d_JL implies pairwise distances preserved within epsilon. (PROVEN)
2. Preserved distances imply separation margin >= (1-epsilon) * delta_min / 2. (PROVEN, Part b)
3. Separation margin implies the M2P can distinguish all 5 domains. (PLAUSIBLE but NOT PROVEN -- depends on the M2P actually learning a classifier-like internal representation)
4. Domain distinction implies generation quality >= 97%. (NOT PROVEN at all)

Step 3 requires proving that the M2P transformer's learned input projection behaves like a near-random projection in the JL sense. The JL lemma holds for random projections (Gaussian, sub-Gaussian, sparse). A LEARNED projection trained by gradient descent has no such guarantee -- it could be better (learning domain-aligned directions) or worse (adversarially misaligned by optimization dynamics).

Step 4 requires proving a functional relationship between representational fidelity (distortion) and generation quality (the quality_ratio metric). No such relationship is established. The quality_ratio is:

    q = (base_loss - m2p_loss) / (base_loss - sft_loss)

This depends on m2p_loss, which is the output of an entire M2P transformer + LoRA forward pass. The projection dimension d_M2P affects only the M2P's internal bottleneck. The M2P must not only REPRESENT the domain correctly (which JL addresses) but also GENERATE the correct B-matrix weights (which JL says nothing about).

**This is precisely the impossibility structure that the post-mortem correctly identifies:** "JL lemma bounds projection DISTORTION, not generation QUALITY -- these are not equivalent."

### The quantitative predictions are ad hoc

The derivation in Section D (the "Note on the 97% threshold") is particularly revealing:

    predicted_gap_128 = 7.8% * (10.7/22.0) = 3.8%
    predicted_quality_128 = 1 - 3.8% = 96.2%

This assumes quality gap scales LINEARLY with distortion. No theorem supports this. The "effective distortion" at d=64 is computed as 10%/0.46 = 22%, which is not a valid use of the JL bound -- the JL bound gives a THRESHOLD, not a linear scaling law. Below the threshold, JL simply says there EXIST configurations with distortion > epsilon; it does not give a formula for how distortion scales with d/d_JL.

The +-1% tolerance band to get from 96.2% to the 97% threshold is arbitrary.

### Assumption 2 explicitly breaks the proof

MATH.md Section E, Assumption 2 states: "JL holds for RANDOM projections; the learned projection may do better or worse." This is an honest acknowledgment, but it means Theorem 1 does not actually apply to the experimental setting. The proof assumes random projections; the experiment uses learned projections. This invalidates the entire quantitative prediction chain.

## Prediction vs Measurement

PAPER.md contains a proper prediction-vs-measurement table. This is good practice regardless of the proof's validity. Results:

| Quantity | Predicted | Measured | Match |
|----------|-----------|----------|-------|
| quality(d=64) | ~92.2% | 95.1% | Off by 2.9pp (d=64 already better than expected) |
| quality(d=128) | >= 97% | 93.1% | FAIL (3.9pp below threshold) |
| quality(128) > quality(64) | True | False (93.1 < 95.1) | FAIL |
| |quality(256)-quality(128)| | < 2% | 2.4% | FAIL |

All three kill criteria fail. The baseline measurement itself (95.1% vs. predicted 92.2%) already signals trouble -- the starting point does not match the prediction from Finding #354, suggesting the experimental conditions differ from the referenced prior work.

The PAPER.md analysis correctly notes that excluding parity (a structural artifact with near-zero denominator), the results become: d=64 -> 96.7%, d=128 -> 95.5%, d=256 -> 97.0%. This shows quality is essentially flat across widths at ~96%, which is a valid observation -- but it falsifies the hypothesis that d_M2P is the bottleneck.

## NotebookLM Findings

Skipping NotebookLM step -- the mathematical analysis above is sufficient to establish the verdict. The critical flaw (distortion != quality) is clear from reading the proof itself.

## Novelty Assessment

The JL lemma applied to bottleneck dimension sizing is not novel -- it is standard practice in random projection literature (Achlioptas 2003, Li et al. 2006). The specific application to M2P hidden-state compression is new to this project but not to the broader literature.

The post-mortem insight (JL bounds distortion, not generation quality) is valuable and correctly identifies the gap. This is a useful negative result.

FlyLoRA (arXiv:2510.08396), already cited in VISION.md, explicitly uses frozen random A-matrices with JL-lemma grounding. The key difference: FlyLoRA uses JL to argue that random A preserves signal FROM the input, not that a bottleneck dimension preserves signal FOR generation. The distinction matters.

## Macro-Scale Risks (advisory)

The observation in Section G that DeepSeek-V3 operates at 11% of the JL floor (d=2048 vs d_JL=18,360 for N=256, epsilon=0.05) is interesting but not actionable. If production MoE systems work far below the JL floor, this further supports the conclusion that JL distortion is not the binding constraint on generation quality.

## Verdict

**KILL** (correctly killed, with caveats on the proof)

The experiment was correctly designed, correctly executed, and correctly killed. All three kill criteria failed. The post-mortem analysis correctly identifies the impossibility structure.

However, the review identifies that the experiment should have been killed BEFORE running, because the proof has a fundamental gap:

1. **Corollary 2 is not a corollary.** It asserts a relationship between JL distortion and generation quality that is never proven. This makes the experiment a guided exploration at best, not a verification.

2. **The quantitative predictions are not derived from the theorem.** The 97% threshold comes from informal scaling arguments (distortion scales linearly with d/d_JL), not from any proven result.

3. **Assumption 2 invalidates the proof's applicability.** The proof holds for random projections; the experiment uses learned projections. The proof cannot make predictions about the experimental setting without additional theory bridging random and learned projections.

If this experiment were still in planning, the correct verdict would be **REVISE** with:
1. Reclassify as Type 2 (guided exploration): "Within the JL framework, what is the relationship between bottleneck dimension and generation quality?"
2. Remove the quantitative prediction of >= 97% (it is not theorem-derived).
3. Keep K871 (monotonicity) as the primary kill criterion -- this IS derived from the JL framework's directional prediction.
4. Acknowledge that quality_ratio is not proven monotone in projection fidelity.

Since the experiment is already complete and correctly killed, the KILL verdict stands. Finding #355 (killed) is appropriate.

### What was learned

The impossibility structure is sound: **JL bounds representation fidelity; generation quality requires both fidelity AND the capacity to transform representations into correct weight matrices.** The bottleneck is not the input projection dimension but the M2P's generative capacity (depth, architecture, training dynamics). This is a valuable negative result that correctly redirects future work away from dimension sweeps and toward M2P architecture improvements.

# Peer Review: Centralized Multi-Class Routing N=24

## Experiment Type
Verification (self-declared, but see below)

## Hack Detector
- Fix count: 1 mechanism (softmax replaces binary heads). Clean, not a hack stack.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 is a genuine proof that softmax eliminates FPR cascade. Theorem 2 is a textbook restatement (proper scoring rule + Cover's theorem). Neither theorem actually predicts the accuracy numbers claimed in Section D Corollary. The quantitative predictions are hand-waved from "binary heads got 87.2% per-head, so softmax should get >70%." That leap is a description dressed in theorem clothing.
- Metric used as evidence: Top-1 routing accuracy. This is reasonable as a proxy for the behavioral outcome (routed PPL < uniform PPL), and both were measured.
- Kill criteria source: K587 (>60%) is partially derived -- the justification ("at 60%, top-2 selection puts >50% weight on correct adapter") is reasonable but informal. K588 (routed PPL < uniform) is well-motivated. K589 (overhead <15%) is generous but fine.

## Self-Test Audit

1. **One-sentence impossibility property:** "Softmax normalization makes independent false positives structurally impossible." Correct, genuinely one property. PASS.

2. **Cited theorems:** Gneiting & Raftery 2007 Theorem 1, Cover 1965, VC 1971. All real theorems. However, Cover's theorem is applied incorrectly in spirit: it guarantees that a separating hyperplane EXISTS, but the proof uses it to claim the problem is "well within the separable regime" as if separability implies learnability from 40 samples. This is a significant gap. PARTIAL PASS.

3. **Predicted numbers:** Top-1 >70%, top-2 >85%, params ~165K, overhead <2%, routed PPL < uniform PPL. Specific and falsifiable. PASS (though derivation of 70%/85% is weak -- see Mathematical Soundness).

4. **Falsification condition:** "If accuracy <50%, the domain signal assumption is wrong or mean pooling destroys multi-class discriminability." This is reasonable but too lenient. The actual measured 39.4% is below 50%, yet the paper does not declare the proof falsified -- it blames the representation instead. The falsification condition should have been sharper: "If accuracy is not significantly better than binary heads (39.6%), then FPR cascade was not the bottleneck." PARTIAL PASS.

5. **Hyperparameter count:** 1 (hidden dim h=64), with justification h >= K-1 = 23. Clean. PASS.

6. **Hack check:** "No. This replaces 24 heads with 1." Correct. PASS.

## Mathematical Soundness

**Theorem 1 (FPR Cascade Impossibility): CORRECT.**
The proof that softmax produces exactly one distribution per input, with ties on a measure-zero set, is straightforward and valid. The shift-invariance argument is standard. This theorem is proven and was verified experimentally (softmax does eliminate FPR cascade).

**Theorem 2 (Convergence to True Routing): CORRECT BUT VACUOUS.**
The proof is mathematically valid -- it is a standard textbook result about proper scoring rules. The problem is the realizability assumption: "If there exist W*, b* such that P(k|x; W*, b*) = Q(k|x)." This is the entire question. Cover's theorem says a separating hyperplane exists for any partition of points in general position, but:

- Cover's theorem applies to a finite training set, not the population distribution.
- The theorem says SOME linear classifier separates the data, not that cross-entropy optimization FINDS it from 40 samples per class.
- Most critically: the theorem applies to points "in general position." Mean-pooled hidden states from related domains (economics vs finance, medical vs health) are NOT in general position -- they cluster, possibly with significant overlap.

**The Corollary (Quantitative Predictions): NOT DERIVED.**
The prediction "Top-1 accuracy >70%" is justified by: "binary heads achieved 87.2% per-head accuracy, and softmax eliminates calibration failure." This reasoning conflates two different metrics:

- Binary per-head accuracy = P(head_k correctly classifies "domain k vs not-k")
- Multi-class accuracy = P(argmax over all 24 classes is correct)

These are fundamentally different problems. A 1-vs-23 binary classifier has a much easier job than a 24-way discriminator. The 87.2% binary accuracy tells us there is SOME domain signal, but does not predict multi-class accuracy. The correct derivation would require the confusion matrix of the binary heads -- which domains confuse with which -- to estimate multi-class separability. This was not done.

The fact that both binary heads (39.6%) and softmax (39.4%) achieve nearly identical accuracy is the strongest finding of the experiment: it proves the bottleneck is representation quality, not routing architecture. The proof correctly identified and eliminated FPR cascade, but incorrectly assumed FPR cascade was the dominant failure mode.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. 4 of 7 predictions failed.

| Prediction | Measured | Gap |
|-----------|----------|-----|
| Top-1 >70% | 39.4% | 30.6 pp -- catastrophic miss |
| Top-2 >85% | 53.1% | 31.9 pp -- catastrophic miss |
| Routed PPL < uniform | Routed 10.107 > uniform 10.080 | Wrong direction |
| Accuracy >> binary (39.6%) | 39.4% (identical) | Core hypothesis refuted |

The predictions that passed (params ~165K, overhead <2%, FPR cascade eliminated) are all architectural properties, not behavioral outcomes. The proof correctly predicts what the router IS but incorrectly predicts what it DOES.

## NotebookLM Findings

Skipping NotebookLM -- the experiment is already killed with clear diagnosis. The review can be completed from the materials alone.

## Novelty Assessment

**Low novelty.** Replacing independent binary classifiers with a single softmax head is standard multi-class classification, taught in introductory ML courses. The Switch Transformer reference (Fedus et al. 2021) already validates softmax routing at scale. CLONE (Chen et al. 2025) validates it for LoRA selection specifically. The experiment's value is not in the method but in the diagnostic finding: both approaches fail identically, proving the bottleneck is representation quality.

## Quality of Post-Mortem

**Excellent.** This is where the experiment delivers real value. PAPER.md contains:

1. A domain-by-domain comparison showing identical success/failure patterns across methods (the strongest evidence that the bottleneck is representation, not architecture).
2. A back-of-envelope calculation explaining why accuracy is ~40% (6 perfectly-routable domains + 18 random ones).
3. A clear diagnosis: mean-pooled hidden states do not separate 14 of 24 domains.
4. Actionable next steps (per-token routing, domain clustering, adapter output quality).

This post-mortem is more valuable than the original experiment. It should be formalized as a finding.

## Macro-Scale Risks (advisory)

Not applicable (experiment killed). However, the representation quality finding is scale-relevant: at macro scale with a larger base model, hidden states may contain richer domain signal, potentially making softmax routing viable. The architectural choice (single softmax vs binary heads) is correct -- the input features are the problem.

## Verdict

**KILL** (confirmed)

The experiment was correctly killed. The proof is partially valid (Theorem 1 is correct and verified; Theorem 2 is correct but vacuous; the quantitative predictions are not derived from the theorems). The core finding -- that FPR cascade was not the disease -- is genuinely valuable.

### Specific issues:

1. **The quantitative predictions (>70%, >85%) are not derived from the proof.** They are informal extrapolations from binary head accuracy, which measures a fundamentally different problem (1-vs-23 vs 24-way). The "Corollary" section should have been labeled as a conjecture, not a corollary.

2. **Cover's theorem is applied beyond its domain.** It guarantees existence of a separating hyperplane for finite point sets in general position. It says nothing about learnability from 40 samples, population generalization, or points not in general position (which overlapping domain representations are not).

3. **The falsification condition was too lenient.** "<50% accuracy" was set as the threshold for falsifying the domain signal assumption, but the actual kill criterion was 60%. The gap between 50% (falsifies proof) and 60% (kills experiment) created a zone where the experiment can be killed without the proof being falsified. In fact, the measured 39.4% is below both thresholds, and the proof's domain signal assumption IS falsified for the multi-class setting, which the post-mortem correctly identifies.

### What should be recorded as a finding:

The valuable output is the **structural diagnosis**: at N=24, approximately 6/24 domains have distinctive representations in mean-pooled hidden states and approximately 18/24 do not, producing a ~40% accuracy ceiling that is routing-architecture-independent. This is a representation quality finding, not a routing finding, and should guide all future routing work at high N.

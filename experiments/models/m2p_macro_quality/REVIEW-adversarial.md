# Peer Review: M2P Macro Quality (exp_m2p_macro_quality) -- Post-Revision

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (only change is D_MODEL 256->512). Clean.
- Is MATH.md a proof or a description? Theorem 1 is a correct but trivial proof. Theorem 2 is a correct but trivial proof. Section C is correctly labeled "Scaling Heuristic (Engineering Estimate, NOT a Theorem)." Honest framing.
- Metric used as evidence: quality_ratio (= fraction of SFT improvement captured by M2P). This is a well-defined proxy but not proven to predict behavioral outcomes at macro scale.
- Kill criteria source: K882 is an empirical conservative floor (85%), K883 is derived from 2x micro measurement (0.7 nats), K884 is an empirical kill floor (60%). Mixed: partially grounded in prior measurement, partially heuristic. Acceptable for guided exploration (Type 2).

## Self-Test Audit

1. One-sentence impossibility property: "When n_train >= T, each training sample is seen at most once. The gradient estimator satisfies the Ghadimi-Lan i.i.d. condition, making cyclic memorization structurally impossible." -- PASS. Single property, clearly stated.

2. Cited theorems:
   - Ghadimi & Lan (2013, arXiv:1309.5549, Theorem 2.1) -- REAL, conditions apply (SGD on L-smooth function).
   - Bartlett et al. (2020, arXiv:1906.11300) -- REAL but conditions do NOT apply to transformers. Correctly downgraded to "engineering estimate" in the revised MATH.md. PASS (honest about limitations).
   - Prechelt (1998) GL criterion -- REAL (method, not a formal theorem). Correctly applied.
   - Hardt et al. (2016) -- REAL. Generalization bound for SGD. Correctly applied.

3. Predicted numbers: T/n_train=0.625 (structural), gap < 0.7 nats (2x micro), quality >= 85% (conservative floor). PASS for specificity. The 85% and 0.7 thresholds are honestly labeled as engineering heuristics rather than tight bounds.

4. Falsification condition: "Theorem 1 is falsified if: train-val gap at n=2000 > 0.7 nats AND n_train=1600 < T=1000 is NOT the cause." FLAG: This condition is logically confused. n_train=1600 is always > T=1000 in this experiment, so the second conjunct "n_train < T is NOT the cause" is always vacuously true (since n_train < T is never the case). The falsification condition reduces to: "gap > 0.7 nats" would falsify, but that does not falsify Theorem 1 per se (which is about i.i.d. sampling, not gap magnitude). The falsification condition conflates the gap threshold (an engineering heuristic) with the theorem's guarantee (i.i.d. gradients). MINOR FLAG: not blocking because the theorem itself is correct, and the experiment is Type 2 exploration, not Type 1 verification.

5. Hyperparameter count: 0 new. PASS.

6. Hack check: No fix stacking. PASS.

## Mathematical Soundness

### Theorem 1 (n_train >= T is d_model-independent)
CORRECT. The proof is trivially valid: the condition n_train >= T involves only data size and step count, not model architecture. The Ghadimi-Lan and Hardt et al. guarantees apply given i.i.d. sampling, which is ensured by single-pass (no cycling). However, the theorem says nothing about the QUALITY of the solution -- only that the optimization conditions are met. The leap from "gradients are i.i.d." to "quality is maintained" requires that the loss landscape at d=512 is comparably favorable, which is not proven but is a reasonable hypothesis for a guided exploration.

### Theorem 2 (GL early stopping bound)
CORRECT. Follows directly from the GL criterion definition. val_loss(T*) <= 1.05 * best_val_loss is a tautology restating the stopping condition. Not incorrect, just trivial.

### Scaling Heuristic (formerly "Theorem 3")
Correctly downgraded from "theorem" to "engineering estimate" in the revision. The authors explicitly acknowledge the Bartlett framework does not apply to nonlinear transformers and that the resulting estimate (0.488) has "zero predictive power." The actual measurement (101.0%) confirms this. PASS -- honest.

### K883 threshold derivation
The threshold 0.7 nats = 2 * 0.337 (micro measurement) is an engineering heuristic, explicitly labeled as such. Not a tight bound. Acceptable for guided exploration.

### Completeness of the mathematical framework for Type 2
The proven framework (Ghadimi-Lan + Hardt) is correctly stated. The unknown is correctly identified: "Does the same recipe achieve >= 85% when d_model doubles?" The experiment narrows this unknown (answer: yes, 101.0%). This is a well-structured guided exploration.

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table (Section 1). Checking each row:

| Prediction | Measured | Assessment |
|---|---|---|
| T/n_train = 0.625 (structural) | 0.625 | Trivially confirmed by construction |
| quality_ratio >= 85% (K882) | 101.0% | PASS, exceeded by 16pp |
| train-val gap < 0.7 nats (K883) | 0.1058 nats | PASS, 6.6x below threshold |
| quality_ratio >= 60% (K884 kill) | 101.0% | Not triggered |
| Degradation <= 12pp vs micro | +3.4pp improvement | Heuristic falsified (but in the favorable direction) |
| n=2000 > n=1000 quality | 101.0% vs 101.2% | Not observed (essentially equal) |

The table is present and complete. Predictions are falsifiable. The Bartlett heuristic is honestly marked as having no predictive power.

## Revision Assessment (Was the Previous REVISE Addressed?)

The previous review identified three blocking and two advisory issues:

1. **[BLOCKING] Val-set mismatch:** FIXED. M2P final evaluation now uses `domain_data_max["val"]` (same as SFT and base) via the `eval_val_batches` parameter. The early-stopping val check still uses the n-specific val set, which is appropriate (it guides training, not final evaluation). Code verified at lines 875-882.

2. **[BLOCKING] SFT training budget:** FIXED. SFT_STEPS = T_FIXED = 1000, matching M2P exactly. Code verified at line 125.

3. **[ADVISORY] N=2 statistical weakness:** ADDRESSED. PAPER.md Section 1 notes: "Only 2 valid domains (sort and reverse) are available after the parity guard excludes arithmetic. With 2 data points, no statistical test is possible." MATH.md Section D.2 includes: "Any claim about d_model-independence should be read as initial evidence with 2 valid domains suggests the recipe may transfer." Finding status is "supported" (not "conclusive"), appropriate for this evidence level.

4. **[ADVISORY] Train-val gap interpretation:** ADDRESSED. PAPER.md Section 3 correctly states: "Train_loss > val_loss indicates early convergence rather than overfitting... K883 PASS means the gap magnitude is small; 'overfitting controlled' is not the correct interpretation." This is honest and accurate.

5. **[ADVISORY] Downgrade Theorem 3:** FIXED. MATH.md Section C now reads "Scaling Heuristic (Quality Floor at d=512 -- Engineering Estimate, NOT a Theorem)" and explicitly states preconditions are violated.

All blocking issues resolved. All advisory issues addressed or acknowledged.

## Remaining Concerns (Non-Blocking)

### Concern 1: Quality > 100% credibility
With the val-set fix and matched SFT training budget, quality > 100% persists. Let me verify the numbers:
- Sort: M2P val = 2.3651 vs SFT val = 2.4196. M2P is 0.0545 nats better.
- Reverse: M2P val = 2.3358 vs SFT val = 2.5446. M2P is 0.2088 nats better.

The reverse margin (0.21 nats) is substantial and unlikely to be noise. Possible explanations:
- M2P's bottleneck (d_M2P=64) acts as implicit regularization, producing better-generalizing B-matrices than direct SFT optimization.
- SFT at 1000 steps on 1600 samples cycles through data 0.625 epochs -- exactly the same regime as M2P. But SFT optimizes B directly (10 * 512 parameters per module) while M2P generates B through a meta-learned function. The meta-learning could provide a regularization benefit.
- The training data for SFT (seed=42) and M2P (seed=2042) are different samples from the same synthetic distribution. At n=2000, this should average out.

ASSESSMENT: The quality > 100% claim is now credible (not an artifact). The paper correctly identifies it as "benign overshoot" and does not overclaim. The magnitude is small enough (1-1.6%) to be a legitimate micro-scale observation without strong macro implications.

### Concern 2: Different training data seeds
SFT trains on data seeded with RNG state 42. M2P trains on data seeded with SEED+n_val (42+2000=2042). Both are synthetic data from identical generators, so distributions are the same in expectation. This is a minor methodological asymmetry but unlikely to explain the 0.21 nats gap on reverse. Would be cleaner to use the same seed for M2P training data, but not blocking.

### Concern 3: Single context token for B-matrix generation
B-matrices are generated from `train_batches[0]` (a single training example). This is the M2P's intended inference mode, but it means the quality measurement depends on which example is fed as context. Different context tokens might produce different B-matrices with different quality. This is not tested.

### Concern 4: n=1000 vs n=2000 shows no quality improvement
The prediction "n=2000 > n=1000 quality" was not observed (101.0% vs 101.2%). Both are effectively at ceiling. This is noted in the paper but deserves more discussion: if quality is at ceiling at n=1000 (which violates n_train >= T), the structural guarantee may not be the binding constraint -- the M2P might simply be easy to train at this scale.

### Concern 5: The theorems are not really doing work
Theorem 1 proves that i.i.d. gradients hold. Theorem 2 proves GL bounds val loss. Neither theorem predicts quality_ratio >= 85%. The actual quality prediction (K882 = 85%) comes from engineering reasoning, not from the theorems. The theorems are true but orthogonal to the main question of whether M2P quality transfers to d=512. For a Type 2 guided exploration, this is acceptable -- the proven framework provides the structural guarantee within which the exploration operates. But the finding should be attributed to the empirical observation, not to the theorems.

## Novelty Assessment

Internal scaling check. No prior art conflict. The contribution is empirical: demonstrating that M2P with fixed architecture (d_M2P=64, L=2) transfers from d=256 to d=512 without modification. This is a useful data point for the research program, not a novel method.

## Macro-Scale Risks (advisory)

1. The quality > 100% observation is interesting but may not hold at real-world data scales where SFT converges to a stronger baseline.
2. With only 2 valid domains, the d_model-independence claim needs more domains before scaling further (d=1024, d=2048).
3. The n=1000 vs n=2000 equivalence suggests the structural guarantee may be slack at this scale. At larger d_model or with more complex domains, it may become the binding constraint again.
4. Context-token sensitivity (Concern 3) should be measured before macro deployment.

## Verdict

**PROCEED**

The previous REVISE issues have been addressed:
- Val-set mismatch fixed (same eval set for all quality measurements)
- SFT budget matched (both 1000 steps)
- Theorem 3 honestly downgraded to scaling heuristic
- Train-val gap interpretation corrected
- N=2 statistical weakness acknowledged; finding status appropriately "supported" not "conclusive"

The quality > 100% result at d=512 is now credible (not an artifact). The experiment demonstrates that the M2P recipe (d_M2P=64, L=2, n=2000, GL stopping) transfers to 2x d_model without modification. The mathematical framework (Theorems 1-2) correctly establishes the structural conditions, even though the quality prediction itself comes from engineering heuristics rather than the theorems.

Finding status of "supported" is correct: the result is based on 2 valid domains with a single seed, which is insufficient for "conclusive."

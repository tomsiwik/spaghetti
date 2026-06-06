# Peer Review: DUME Ridge Regression Router

## Experiment Type
Guided exploration (Type 2): Proven framework (ridge regression closed-form solution), unknown is whether hidden state geometry of BitNet-2B-4T supports domain separation sufficient for routing.

## Hack Detector
- Fix count: 1 mechanism (ridge regression). No stacking. Clean.
- Is MATH.md a proof or a description? Mixed. The ridge optimality proof (Section C) is genuine (Theorem/Proof/QED). Theorem 1 (Bounded Routing Error) is a proof SKETCH with gaps -- see below.
- Metric used as evidence: Classification accuracy on held-out test set. Appropriate for the behavioral outcome (correct domain routing).
- Kill criteria source: K693 derived from Theorem 1 + prior baselines. K694/K695 derived from complexity analysis. Reasonable.

## Self-Test Audit

1. **One-sentence impossibility property:** "Ridge regression has unique closed-form solution for any lambda > 0." This is about eliminating training instability, which is real but not the UNKNOWN being explored. The unknown is domain separability -- the self-test answer addresses a property of the framework, not the exploration target. Acceptable for Type 2 but slightly evasive.

2. **Cited theorems:** Tikhonov (Hoerl & Kennard 1970) -- real. Woodbury identity -- real. Fisher's LDA (Fisher 1936) -- real. Cover's theorem on linear separability -- real but applied loosely (Cover's theorem is about random projections onto high-dimensional space making data linearly separable; the hidden states are not random projections of the domain labels). Conditions partially checked.

3. **Predicted numbers:** >= 90% accuracy, < 60s init, < 10s incremental. Specific and falsifiable. PASS.

4. **Falsification condition:** "Accuracy ~20% = random means linear separability fails." This is correct and targets the exploration's unknown. PASS.

5. **Hyperparameter count:** 1 (lambda). Acknowledged, with GCV as a principled alternative. Sweep shows insensitivity across 3 orders of magnitude. PASS.

6. **Hack check:** "Replaces all existing routing mechanisms." Honest. PASS.

## Mathematical Soundness

### Ridge Optimality Proof (Section C) -- SOUND
The derivation of W* = (X^TX + lambda*I)^{-1} X^TY is textbook correct. Gradient set to zero, Hessian positive definite. No issues.

### Theorem 1 (Bounded Routing Error) -- PROBLEMATIC
The bound epsilon <= (K * lambda * trace(Sigma^{-1})) / (n * Delta^2) has issues:

1. **Proof is a sketch, not a proof.** The derivation jumps from "W* -> Sigma^{-1}[mu_1,...,mu_K] as lambda->0 and n->infinity" to a Chebyshev bound. The intermediate steps are missing. How does the Chebyshev inequality on Mahalanobis distance yield the specific form with K, lambda, and trace(Sigma^{-1})?

2. **The bound mixes asymptotic and finite-sample regimes.** It uses the asymptotic form of W* (lambda->0, n->infinity) but then claims the bound holds for finite lambda and n. The bias term "lambda * trace(Sigma^{-1}) / n" is asserted, not derived.

3. **Practical tightness unknown.** The Corollary says "if Delta >> 1, accuracy -> 100%" but never evaluates what Delta actually is for this experiment. The hidden state analysis reports Fisher discriminant ratio = 1.24, which is not ">> 1". The proof's bound is never numerically instantiated, which means it cannot be verified.

4. **The "proof" is used as motivation, not prediction.** The actual kill criterion (90%) comes from "even TF-IDF achieves 90%," not from evaluating the bound. This is honest but means Theorem 1 does not actually predict the 90% threshold.

**Verdict on Theorem 1:** It is a plausible heuristic argument dressed as a theorem. For a Type 2 guided exploration, this is acceptable -- the proven framework is ridge regression itself (which is genuinely proven), and Theorem 1 is attempting to bridge to the unknown (domain separability). But it should not be called a theorem; it should be called a "heuristic bound" or "motivating analysis."

### Theorem 2 (Incremental Update) -- SOUND
Direct application of Woodbury identity. Complexity analysis correct. Verified numerically (7.02e-05 relative diff). No issues.

### DUME Citation (arXiv 2603.29765)
The paper exists in the repo (dume/ directory). The sufficient statistics framework (G, H) and Woodbury update are correctly attributed. The connection is legitimate.

## Prediction vs Measurement

PAPER.md contains a proper prediction-vs-measurement table. Assessment:

| Prediction | Source Quality | Measurement | Match Quality |
|-----------|--------------|-------------|---------------|
| Accuracy >= 90% | Weak (from baseline, not proof) | 96.0% | PASS but source is weak |
| Init time < 60s | Strong (complexity analysis) | 23.4s | PASS with 2.6x margin |
| Incremental < 10s | Strong (Woodbury complexity) | 14.3ms | PASS with 700x margin |
| Ridge >= TF-IDF | Reasonable (richer features) | 96% vs 84% | PASS but caveat below |
| Ridge ~= embedding | Reasonable | 96% vs 96% | PASS |
| Woodbury = full refit | Strong (mathematical identity) | 7.02e-05 diff | PASS |
| Lambda insensitive | Moderate (well-separated) | Stable across 0.01-10 | PASS |

**Critical caveat on TF-IDF comparison:** PAPER.md reports TF-IDF at 84% on THIS experiment's test split, vs 90% in the prior experiment (different split). With only 50 test samples, this 6pp difference could easily be split noise. The paper acknowledges this in Limitations, which is good. But the claim "Ridge >= TF-IDF" is comparing against the WEAKER of the two TF-IDF numbers. The honest comparison is 96% vs 84-90% TF-IDF, which is still favorable but less dramatic.

## Novelty Assessment

### What is actually novel here?
Applying ridge regression to hidden states for adapter routing. This is a straightforward application of DUME's framework to a specific use case. The novelty is in the VERIFICATION that hidden states of BitNet-2B-4T actually separate domains well enough for this to work.

### Prior art
- DUME (arXiv 2603.29765) already proposes exactly this: ridge regression on hidden states for expert routing. This experiment is a faithful reproduction of the core idea, not a novel contribution. The MATH.md correctly attributes this.
- Nearest centroid (cosine) achieves identical 96% accuracy with simpler math. This is significant: the ridge regression apparatus (Gram matrix, Woodbury updates, etc.) provides no accuracy improvement over computing means and doing argmax cosine similarity.

### The "replacing all prior routing" claim
VISION.md shows routing is already "SOLVED" with softmax router matching oracle at N=24. This experiment proposes an alternative routing method. The claim in the Self-Test ("replaces all existing routing mechanisms") is overstated given:
- Ridge = nearest centroid cosine = 96% accuracy. The complexity advantage exists only for incremental updates.
- Softmax router already matches oracle at N=24 with within-cluster misrouting being benign.
- Ridge regression was tested at N=5 only. The N=24 scaling question is unanswered.

## Specific Critiques

### 1. Nearest Centroid Cosine = Ridge Accuracy (the elephant in the room)
Both achieve 96%. The ridge regression adds O(d^2) storage for the Gram matrix (25 MB) and O(d^3) solve time. The centroid approach needs O(d*K) storage (0.05 MB) and O(d*K) time. Ridge's only advantage is the Woodbury incremental update, but adding a centroid incrementally is O(d) -- vastly cheaper than Woodbury's O(d^2*m).

The paper does not discuss this. It should explicitly address: when does ridge regression provide value OVER nearest centroid cosine?

### 2. N=5 Scale Limitation
Tested with 5 domains. VISION.md shows routing at N=24 is the relevant scale, and routing heads collapsed at N>10. Does ridge regression maintain accuracy at N=24? The legal-finance confusion (cos=0.981) at N=5 suggests that with 24 domains, more overlapping pairs will emerge. This is acknowledged implicitly but deserves explicit treatment.

### 3. Legal-Finance Confusion Analysis
Good: the cosine similarity of 0.981 is reported and connected to the two errors. This is honest and useful.
Missing: What is the actual Mahalanobis distance (Delta) between legal and finance? Theorem 1's bound depends on Delta, but Delta is never computed. Computing it would either validate or falsify the bound.

### 4. The NaN Warning
"Some hidden states contain inf/nan values from float32 overflow" -- this is brushed aside with "regularization stabilizes." How many samples have NaN values? If a significant fraction of the 250 calibration samples contain NaN/inf, the Gram matrix X^TX is corrupted. The claim that "regularization fixes it" needs verification: are the NaN samples silently zeroed, or do they propagate?

### 5. Test Set Size
10 samples per domain = 50 total. 2 errors = 96%. But 95% CI is approximately 86-99%. This is honestly acknowledged in Limitations section 1.

### 6. Cal/Test Split from Same Source
Both calibration and test data come from the same .jsonl files (validation split, or train if insufficient). Lines 158-168 show calibration = first 50, test = next 10 from the same file. This is a valid train/test split but NOT a domain-shift test. The paper should note this: routing accuracy on truly out-of-distribution queries within a domain is untested.

## Macro-Scale Risks (advisory)

1. **G matrix scales as O(d^2).** At d=8192 (70B class), G is 268M entries = 2GB in float64. The paper acknowledges this and suggests PCA, which is reasonable.

2. **Adapter-altered hidden states.** The calibration uses base model hidden states. At inference, adapters are loaded and may shift the representation space. This could cause routing accuracy to degrade at inference time. Important to test.

3. **N=24+ scaling.** The discriminant ratio of 1.24 at N=5 will likely decrease as more domains are added. Ridge regression's linear boundaries may not scale as well as the softmax router that already matches oracle at N=24.

## Verdict

**PROCEED** (with advisory notes, no blocking revisions)

**Justification:**

The experiment successfully answers its Type 2 question: "Do hidden states of BitNet-2B-4T support domain separation for closed-form ridge routing?" Answer: yes, at N=5, with 96% accuracy.

The framework (ridge regression) is genuinely proven. The unknown (domain separability) is meaningfully narrowed. The evidence is honestly reported with appropriate caveats. Kill criteria all pass with margin.

The main weakness -- Theorem 1 is a sketch, not a proof -- is not blocking for a Type 2 guided exploration. The actual finding is empirical (domain separation exists) supported by a well-understood framework (ridge regression), not a novel theoretical result.

**Advisory notes for the finding record:**

1. Status should be `supported`, not `conclusive`. Theorem 1 is not formally proven, and the test set is too small for statistical certainty.
2. The finding should note that nearest centroid cosine matches ridge accuracy -- the incremental update property (Woodbury) is the differentiating feature, not accuracy.
3. The legal-finance confusion (cos=0.981) should be recorded as a known failure mode for future experiments.
4. The claim about "replacing all prior routing" should be downgraded. Ridge regression is a viable alternative to centroid routing with advantages for incremental expert addition, not a proven replacement for the softmax router at N=24.
5. The NaN issue should be investigated before this goes to production. Silent corruption of the Gram matrix is a reliability concern.
6. N=24 scaling should be tested before claiming this solves routing generally.

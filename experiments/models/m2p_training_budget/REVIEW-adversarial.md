# Peer Review: M2P Training Budget Sweep

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 2 (training steps sweep + bidirectional attention). Below the 3-flag threshold but note these are independent interventions, not a stacked fix.
- Is MATH.md a proof or a description? Mixed. Theorem 1 is a proof sketch that re-derives a known O(1/T) convergence result and applies it to the M2P setting. Theorem 2 is a valid set-inclusion proof. Neither is novel mathematics; both are correct applications of existing theory. However, the "proof" of Theorem 1 contains a gap: the translation from gradient-norm bound to loss-gap bound assumes local quadratic structure (Hessian approximately constant near minimum), which is stated without justification and is not guaranteed for a neural network that may not be near a minimum at T=500.
- Metric used as evidence: quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss). This is a reasonable proxy for M2P-to-SFT approximation quality, though it conflates generalization error with optimization error.
- Kill criteria source: Derived from the O(1/T) prediction -- the 2pp threshold in K876 is below the pessimistic prediction of +3.1pp. Kill criteria are appropriately derived from the proof.

## Self-Test Audit

1. **One-sentence impossibility property:** "O(1/T) convergence rate of SGD/Adam on smooth losses guarantees that the M2P training loss decreases monotonically with training steps." This is one property. PASS. However, it guarantees training loss decrease, not test/eval loss decrease -- a critical distinction the self-test elides.

2. **Cited theorems:** Ghadimi & Lan (2013, arXiv:1309.5549) -- real, correctly cited. Bottou et al. (2018, arXiv:1606.04838) -- real, correctly cited. SHINE (arXiv:2602.06358) -- real. PASS.

3. **Predicted numbers:** Conservative q(1000) ~ 95.0%, q(2000) ~ 96.5%. Optimistic q(2000) ~ 98%. These are specific and falsifiable. PASS.

4. **Falsification condition:** "The proof is wrong if the M2P training loss is NOT L-smooth." This targets a proof assumption, not just the experiment. Adequate. PASS.

5. **Hyperparameter count:** 0 new hyperparameters. PASS.

6. **Hack check:** 2 changes, both independently motivated. PASS.

## Mathematical Soundness

### Theorem 1 (Training Budget Convergence)

**Step-by-step verification:**

1. The M2P regression loss L(theta) = E[||M2P_theta(h) - B*||^2_F] is correctly stated as bounded below, differentiable, with unbiased gradients. CORRECT.

2. Citation of Ghadimi & Lan Theorem 2.1 for min_{t<=T} E[||grad f(x_t)||^2] <= O(1/T). CORRECT application, preconditions (L-smoothness, bounded variance) are reasonable for a transformer with smooth activations.

3. **CRITICAL GAP: Translation from gradient norm to loss gap.** The proof claims "Near a local minimum where the Hessian is approximately constant, ||grad f||^2 approximately equals H * (L_T - L*), so E[L_T] - L* <= O(1/T)." This step has three problems:
   - (a) At T=500, the M2P is explicitly stated to NOT be near a local minimum (the whole hypothesis is that more steps are needed). The local-quadratic approximation does not hold far from a minimum.
   - (b) Even near a minimum, the relationship ||grad f||^2 ~ H * (L - L*) requires the Hessian H to be well-conditioned and approximately constant in a neighborhood. This is an additional assumption not listed in Section E.
   - (c) The Ghadimi & Lan bound is on min_{t<=T}, not on the final iterate. It says the BEST iterate in the trajectory has small gradient, not that the final iterate does.

4. **CRITICAL GAP: O(1/T) applies to training loss, not generalization.** The convergence theorem is about optimization (driving training loss down). The quality metric measures eval loss on held-out data. The proof implicitly assumes training loss = eval loss, which is violated when overfitting occurs. With only 500 training samples and up to 2000 steps (4 epochs over the same data), overfitting is the primary risk, and the proof has no machinery to address it.

5. The corollary (diminishing returns) follows correctly from O(1/T) if the O(1/T) bound itself applies. Conditional CORRECT.

### Theorem 2 (Bidirectional Attention)

The set-inclusion argument is valid: A_causal is a subset of A_bidir, so min over A_bidir <= min over A_causal. CORRECT. The caveat that the gain may be zero is appropriate.

**However:** This is a statement about the MINIMUM of the loss over attention patterns, not about what SGD finds within T steps. A larger feasible set does not guarantee SGD reaches a better point -- it can also make optimization harder (more local minima, saddle points). The theorem applies to the global optimum but says nothing about the optimization trajectory.

### Quantitative Predictions

The predictions in Section D assume:
- 1 - q(T) proportional to C/T (pure O(1/T) on eval loss)
- The baseline q(500) = 91.9% from Finding #357 is reproducible

**Problem:** The measured T=500 baseline is 89.4%, not 91.9%. The MATH.md uses 91.9% (from a DIFFERENT experiment run) as the baseline, but the actual T=500 measurement in this experiment is 89.4%. This 2.5pp discrepancy (91.9% vs 89.4%) already exceeds the per-run variance discussed in Assumption 3 (2-5pp) and demonstrates the fragility of single-seed extrapolation.

### Kill Criteria Derivation

K876 (quality(2000) > quality(500) + 2pp): Appropriately derived from the pessimistic prediction of +3.1pp. The 2pp margin is reasonable.

K877 (quality(2000) >= 97%): This is aspirational rather than derived from the proof. The proof gives a range of 95-98%, and 97% falls in the middle. Reasonable as a success gate but not a tight prediction.

K878 (|q(2000) - q(1000)| < 1pp): This tests for saturation. Appropriately derived from diminishing returns prediction. Note the K878 label is slightly confusing: K878 PASS = plateau = budget exhausted, which is a kill outcome. K878 FAIL = still improving. The labeling is inverted from intuition but internally consistent.

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table. All predictions are WRONG:

| Prediction | Predicted | Measured | Delta |
|---|---|---|---|
| T=500 baseline | ~91.9% | 89.4% | -2.5pp |
| T=1000 improvement | +3.0pp (to ~95%) | -4.7pp (to 84.7%) | -7.7pp error |
| T=2000 improvement | +4.6pp (to ~96.5%) | -6.4pp (to 83.0%) | -11.0pp error |
| Bidirectional gain | +1-2pp | -4.6pp | -6pp error |
| Monotone improvement | YES | NO (inverted) | Wrong direction |

The magnitude of prediction error (11pp in the wrong direction) is not a normal miss -- it indicates the proof's model is fundamentally wrong for this setting. The proof predicted monotone improvement; the data shows monotone DEGRADATION.

## Root Cause Analysis (the review's independent assessment)

The post-mortem in the context claims "fresh M2P reinitialization per step count = different random seed per point." **This is incorrect per the code.** Line 672 shows `mx.random.seed(SEED)` is called at the start of every `phase_train_m2p()` call, so all M2P models start from identical initialization.

The actual root cause is **overfitting on cyclic data:**
- There are 500 training samples per domain.
- At T=2000, the M2P cycles through these 500 samples 4 times (line 696: `tokens = train_batches[step % len(train_batches)]`).
- The results.json confirms this: for the reverse domain, training loss drops from 2.01 (T=500) to 1.45 (T=2000), but eval loss increases from 2.80 to 3.86. This is textbook overfitting.
- For arithmetic, training loss at T=2000 is 2.25 (higher than T=1000's 1.85), which could indicate instability from repeated passes over memorized data.

The O(1/T) convergence theorem assumes access to fresh i.i.d. samples (or at minimum, an epoch-based analysis). With cyclic data, the gradient variance term sigma^2 in the bound is not bounded by the population variance -- it becomes zero as the model memorizes, but generalization degrades. This is a fundamental violation of Assumption 3 (Section E) that MATH.md fails to address.

**Secondary root cause for bidirectional attention hurting:** The repeat domain collapses from 89.3% to 81.2% with bidirectional attention. This is consistent with Theorem 2's global-minimum guarantee NOT implying better SGD convergence -- the larger attention space may create more local minima that trap the optimizer. At 500 steps on 500 samples, the optimization landscape matters more than the expressiveness of the function class.

## NotebookLM Findings

NotebookLM was not used for this review due to the straightforward nature of the mathematical claims and the clear experimental failure. The issues are identifiable through direct code and proof inspection.

## Novelty Assessment

Low novelty. The experiment applies standard SGD convergence theory to a hypernetwork training sweep. The O(1/T) bound is a textbook result. The set-inclusion argument for bidirectional attention is elementary. The SHINE citation provides relevant empirical backing but operates at orders-of-magnitude larger scale (6B tokens vs 500 samples).

No prior art was found that specifically sweeps M2P training budget in this architecture, so the experiment is novel within the project's context, if not in the broader literature.

## Macro-Scale Risks (advisory)

1. At macro scale with sufficient data, the overfitting confound disappears -- the O(1/T) prediction may actually hold. This experiment does NOT prove that training budget is irrelevant; it proves that the micro-scale data is too small to test the hypothesis.
2. The bidirectional attention result is likely an artifact of the tiny scale. At larger scale with more optimization steps and diverse data, the set-inclusion guarantee should eventually manifest.
3. The per-domain variance (2-5pp) at micro scale means any effect smaller than ~6pp is undetectable with single-seed runs. The predicted improvement of 3-6pp falls within the noise floor.

## Verdict

**KILL** -- correct status, but the interpretation needs correction.

The experiment is correctly killed: all three kill criteria failed, and quality degraded with more training steps. However, the PAPER.md and post-mortem diagnose the wrong root cause.

**Specific issues:**

1. **The claimed confound ("fresh M2P reinitialization per step count = different random seed") is factually wrong.** The code uses `mx.random.seed(SEED)` at the start of every M2P training call (line 672). All models start from identical initialization.

2. **The actual root cause is overfitting on 500 cyclic samples**, which the O(1/T) proof does not address because the theorem assumes i.i.d. sampling (or at minimum, bounded generalization gap). The proof makes no prediction about what happens when you cycle through 500 samples 4 times.

3. **The proof's Assumption 3 acknowledges between-run variance of 2-5pp but claims this is eliminated by "shared infrastructure."** In reality, shared infrastructure eliminates variance in the base model and SFT adapters, but the M2P training itself still has variance from the stochastic optimization trajectory -- and more importantly, it has systematic bias from overfitting on the same 500 samples.

4. **MATH.md Section E does not list "overfitting on cyclic/finite data" as an assumption or breaking condition.** This is a critical omission. The theorem's guarantee is for population-level convergence; the experiment measures sample-level generalization. These are different quantities.

**What the killed experiment actually proves:** The M2P training setup with 500 samples is in a data-limited regime where more optimization steps cause overfitting. The ~92% ceiling is not a convergence problem (more steps make it worse), nor an architecture problem (width/depth already closed). It may be a data quality/quantity problem or an intrinsic limitation of the B-matrix regression task at micro scale.

**If a follow-up is planned, it should:**
1. Diagnose overfitting explicitly: plot train loss vs eval loss curves (not just endpoints) to confirm the train-eval gap widens with T.
2. Either increase training data (more samples per domain) or add regularization (dropout, weight decay, early stopping) -- and derive the math for why this would help.
3. Correct the impossibility structure: the theorem about O(1/T) convergence is not wrong, but it is inapplicable to the finite-data cyclic-sampling regime used here. The finding should record "O(1/T) convergence assumes i.i.d. sampling; with 500 cyclic samples, overfitting dominates" as the impossibility structure.

# Peer Review: M2P Scale Calibrated

## Experiment Type
Claimed as Type 1 (verification) with Type 2 component for lambda. In practice, this is closer to Type 2 (guided exploration) masquerading as Type 1 -- the "proof" does not prove the key claim that was tested and killed (self-calibration across contexts).

## Hack Detector
- Fix count: 2 mechanisms (L_task + L_preserve). Not flagged -- this is standard constrained optimization with two terms.
- Is MATH.md a proof or a description? **Description dressed in equations with a QED label.** Theorem 1 proves the existence of a fixed point alpha* for a given input context. It does NOT prove self-calibration (that alpha* varies across contexts). Self-calibration is stated as "Step 5" of the proof but is actually an unproven claim -- see Mathematical Soundness below.
- Metric used as evidence: Coefficient of Variation (CV) of adapter magnitude across contexts. This is the correct metric for the self-calibration claim. Degradation percentage for K849 is also appropriate.
- Kill criteria source: K849 is derived from the proof (KKT equilibrium implies bounded degradation). K850 (CV > 0.05) is claimed to be derived from Theorem 1 Step 5 but the threshold itself (5%) is stated to come from the prediction that "different operand sizes should differ in required adapter magnitude by at least 5%" -- this is an assertion, not a derivation.

## Self-Test Audit

1. **One-sentence impossibility property:** "The KKT stationarity condition h(alpha) = ... has a unique crossing from negative to positive (by IVT + sign analysis), making it impossible for unconstrained gradient descent to drive alpha to infinity." -- This is a clear, single property. PASS. However, note this property addresses unbounded scale growth, not self-calibration. The self-calibration impossibility (constant-output M2P should be impossible) is not stated.

2. **Cited theorems:** KKT conditions (real, correctly cited), IVT (real, correctly applied to sign analysis), Davis-Kahan (real but loosely connected -- serves as motivation not formal application), Brouwer Fixed-Point Theorem (real, cited but not actually used -- the proof uses IVT on a 1D function, not Brouwer on a compact convex set). FLAG: Brouwer is unnecessary window dressing. The proof works with IVT alone.

3. **Predicted numbers:** K849 < 10pp (specific, falsifiable), CV > 0.05 (specific, falsifiable), scale in [3,15] (specific, falsifiable). PASS on specificity.

4. **Falsification condition:** "The proof is wrong if L_preserve does NOT monotonically increase with alpha." This correctly targets the core assumption. Also lists M2P ignoring input, which targets the capacity assumption. PASS.

5. **Hyperparameter count:** 1 (lambda). Correctly acknowledged that lambda encodes user preference and is irreducible. PASS.

6. **Hack check:** Claims this is not fix #N but a fresh approach (single domain + preservation loss). Reasonable -- the prior experiments (#341, #342) failed due to multi-domain gradient conflicts, and this experiment explicitly avoids that. PASS.

## Mathematical Soundness

### What holds

**Steps 1-3 of Theorem 1 are correct.** The gradient decomposition, KKT stationarity condition, and sign analysis are standard results applied correctly. The IVT argument for existence of alpha* where h(alpha) = 0 is valid under the stated assumptions.

**K849 is well-grounded.** The prediction that L_preserve constrains degradation follows directly from the KKT equilibrium argument. The experiment confirms this: -59.01pp degradation (improvement) with L_preserve vs -3.20pp without.

### What does NOT hold

**Step 4 (convergence to alpha*) is hand-waved.** The claim "Since L_task and L_preserve are both convex in B in a neighborhood of B*" is asserted without proof. Cross-entropy loss of a neural network is NOT generally locally convex in the parameters. This is a known fact. The step should say "gradient descent finds a stationary point" (which it does, by construction) but cannot claim this stationary point is the alpha* from Step 3 without additional convexity or Polyak-Lojasiewicz arguments.

**Step 5 (self-calibration) is NOT a proof -- it is a claim.** This is the critical issue. Step 5 says:

> "Hard domains (large task gradient) -> large alpha* ... Easy domains (small task gradient) -> small alpha*"

This would follow IF the task gradient magnitude at alpha* varies monotonically with task difficulty AND M2P has sufficient capacity to represent this variation. Neither of these is proven. The statement "alpha* emerges from the data" is true by definition of gradient descent convergence, but the claim that alpha* DIFFERS across contexts requires proving that:

(a) The per-context optimal alpha*(c) varies meaningfully across contexts c.
(b) M2P can represent the function c -> alpha*(c).

Neither (a) nor (b) is addressed in the proof. (a) is an empirical claim about the loss landscape. (b) is an architectural capacity question. The proof simply asserts both.

**The proof conflates "per-context fixed point exists" with "M2P discovers per-context fixed points."** Theorem 1 proves: for ANY fixed context, there exists alpha* balancing task and preserve gradients. This does not imply that when M2P is trained on a MIXTURE of contexts with alternating batches, it will learn to output context-dependent alpha. It could equally well (and DID) converge to a single alpha that is a weighted average of per-context optima.

### Bounds analysis

The predicted scale range [3, 15] is based on interpolation from Finding #330 (scale=5: 0pp, scale=20: -42pp). This is empirical curve-fitting, not a derivation from the proof. The actual learned scale was 37.43, which is 2.5x above the upper bound. This falsifies the empirical prediction but does not falsify the mathematical structure (which only guarantees existence of a finite alpha*, not its location).

### The real problem: mean-pooled context injection

Line 446 of run_experiment.py reveals the architectural bottleneck:

```python
task_ctx = mx.mean(task_emb, axis=1, keepdims=True)  # (1, 1, D_M2P)
mem = self.memory + mx.broadcast_to(task_ctx, (1, N_MEMORY, D_M2P))
```

The task context is mean-pooled across ALL token positions into a SINGLE vector in R^64, then additively broadcast to all 8 memory tokens identically. This is the exact same additive conditioning mechanism that Finding #342 killed for centroid collapse. Different arithmetic sequences (easy vs hard) differ primarily in token statistics (digit count, number of '+' symbols). After mean-pooling, "5+3=8" and "987+456=1443" both produce a 64-dimensional mean embedding. The information distinguishing "easy" from "hard" is compressed into a single vector that is then overwhelmed by the 8 learned memory tokens and 2 layers of self-attention.

This is not an assumption violation discovered by the experiment. It is a known limitation of additive conditioning from Finding #342 that was carried forward without architectural change.

## Prediction vs Measurement

PAPER.md contains a prediction vs measurement table. Assessment:

| Prediction | Measured | Match | Review Assessment |
|---|---|---|---|
| gen degradation < 10pp | -59.01pp | YES | Valid confirmation of KKT equilibrium |
| magnitude CV > 0.05 | 0.0093 | NO | Falsified -- M2P outputs constant magnitude |
| scale in [3,15] | 37.43 | NO | Falsified -- empirical prediction was not derived from proof |
| hard > easy ratio > 1.0 | 0.998 | NO | Falsified -- no self-calibration signal |
| preserve reduces degradation | -59.01 vs -3.20 | YES | Valid but paradoxical -- see below |
| Grassmannian cos ~ 0 | 0.0814 | PARTIAL | Expected given d=256 and shared QR seeds across slots |

### The K849 paradox is correctly identified but underexplored

The -59.01pp "degradation" is actually an enormous improvement in general CE (12.18 -> 4.99). This is NOT what the KKT equilibrium predicts. The equilibrium predicts that the adapted model performs SLIGHTLY WORSE than the base model on general data (bounded degradation), not dramatically BETTER. The fact that general CE drops by 59% with the adapter applied means M2P is generating an adapter that improves general text modeling -- it is NOT generating a task-specific adapter at all. The adapter has become a general-purpose improvement, which is the opposite of specialization.

PAPER.md does identify this paradox ("suggests the M2P is learning a general-purpose improvement adapter rather than a narrow task adapter") but does not fully grapple with its implications for Theorem 1. If the adapter improves general quality rather than degrading it, then the monotonicity assumption (Assumption 2: L_preserve increases with alpha) is violated at the operating point. This means the KKT equilibrium argument does not apply in the standard form -- the "equilibrium" found is at a point where BOTH gradients point in the same direction (both want larger alpha), not in opposing directions.

### The K850 threshold of 0.05 is generous, and the experiment still fails catastrophically

CV = 0.0093 is not "slightly below 0.05" -- it is 5x below. The baseline without L_preserve has CV = 0.0124, which is actually HIGHER. This means L_preserve is making M2P MORE constant, not less. This is the opposite of the self-calibration prediction.

## NotebookLM Findings

NotebookLM was not run for this review due to the experiment being already killed with clear structural reasons. The mathematical analysis above is sufficient.

## Novelty Assessment

The KKT-constrained adapter training idea (L_task + lambda * L_preserve) is not novel. This is standard multi-objective optimization / regularization. Specific prior art:

- **EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017):** Uses a Fisher-information penalty to prevent catastrophic forgetting. L_total = L_task + lambda * L_EWC. Same mathematical structure.
- **Progressive Neural Networks (Rusu et al. 2016):** Freeze old columns to prevent forgetting (architectural constraint rather than loss-based).
- **LoRA with weight decay:** Standard practice already regularizes adapter magnitude.
- **Preservation loss in adapter tuning:** Common in instruction tuning (see Alpaca, LLaMA-Adapter).

The novelty claimed here is that M2P + L_preserve enables "self-calibration" where the scale varies by context. This claim was falsified.

## Is the Kill Justified?

**Yes, the kill is justified and the failure is structural, not parametric.**

Could the experiment have passed with different thresholds? No.

1. K850 threshold of 0.05 is already lenient. Measured CV=0.0093 means the M2P output is effectively constant. No reasonable threshold change would make this "self-calibration."

2. The learned scale of 37.43 is not a near-miss of the [3,15] range -- it is 2.5x above. And the K849 "pass" is paradoxical: the adapter IMPROVES general quality rather than degrading it, suggesting it is not functioning as a task-specific adapter at all.

3. The baseline (no L_preserve) has HIGHER CV (0.0124 vs 0.0093), meaning L_preserve made the problem worse for self-calibration, not better.

## Accumulated Impossibility Structure Across #341, #342, #343

Three consecutive M2P experiments have been killed. The pattern:

| Finding | Failure Mode | Root Cause |
|---|---|---|
| #341 | B-matrix centroid collapse across domains | Gradient conflicts in round-robin multi-domain training |
| #342 | Additive domain embeddings insufficient | Jacobian d_B/d_e is effectively low-rank; additive conditioning overwhelmed by memory tokens |
| #343 | Constant-magnitude output (no self-calibration) | Mean-pooled additive context injection collapses to single output; same architectural bottleneck as #342 |

**The common structural problem is the M2P context injection mechanism.** All three experiments use additive conditioning: context information is added to memory tokens, then memory tokens attend only to each other (not to context tokens). The context signal is diluted by:
1. Mean-pooling across token positions (information compression)
2. Additive injection (can be overwhelmed by memory token learned values)
3. Self-attention over memory tokens only (no cross-attention to context)

**Is M2P hypernetwork approach structurally impossible?** No -- but the specific architecture used in all three experiments has a provably weak context injection path. The SHINE paper (arXiv:2602.06358) that M2P is based on uses cross-attention between memory tokens and hidden states, not mean-pooled additive injection. The three kills demonstrate that the SIMPLIFIED M2P architecture (additive conditioning, memory-only attention) is insufficient, not that hypernetwork approaches are impossible.

**What would need to change:**
- Cross-attention from memory tokens to input tokens (not mean-pooling)
- Or: eliminate the Transformer bottleneck entirely and use a direct MLP from context features to scale parameter
- Or: per-context gradient steps (inner-loop optimization, MAML-style)

## Macro-Scale Risks (advisory)

1. The fundamental question -- whether M2P can generate context-dependent B-matrices -- remains unanswered at any scale. Until the architecture is fixed, scaling up parameters will not help.
2. The L_preserve regularization mechanism IS confirmed to work for bounding degradation. This is a useful result for production even without self-calibration.
3. The K849 paradox (adapter improving general quality) warrants investigation -- if M2P learns general-purpose adapters, this might be a feature rather than a bug for certain use cases.

## Verdict

**KILL** (confirmed)

The kill is correct and well-justified. Specific findings:

1. **Theorem 1 Step 5 (self-calibration) was never a proof** -- it was an unsubstantiated claim about how M2P would behave, dressed in the language of the KKT result from Steps 1-3. The actual theorem proves only that a fixed point exists for a fixed context. The extension to "different contexts produce different fixed points in M2P" requires additional architectural capacity guarantees that were never provided.

2. **The K849 pass is misleading.** The -59.01pp "degradation" is an improvement, not bounded degradation. The adapter is learning to be a general-purpose helper, not a task specialist. This violates Assumption 2 (monotonicity of L_preserve in alpha) and means the KKT interpretation does not apply cleanly.

3. **The context injection architecture is the same bottleneck killed in #342.** Mean-pooled additive conditioning was already shown to be insufficient. Carrying it forward without modification predictably produced the same result.

4. **Three consecutive kills with the same root cause (weak context injection) establish that the simplified M2P architecture is a dead end.** Future M2P experiments must use cross-attention or a fundamentally different context encoding before any self-calibration or per-context generation claims can be tested.

### What was learned (positive)

- L_preserve DOES bound adapter impact on general quality (K849 mechanism confirmed)
- This is useful independently of the self-calibration question
- The experiment was well-designed with proper baselines and controls
- PAPER.md honestly reports the paradoxical K849 result and does not oversell

### For the finding record

The impossibility structure should be updated to specifically identify: **additive mean-pooled context injection into self-attending memory tokens cannot distinguish task difficulty at the architectural level, independent of M2P capacity or training procedure.** This is the common root cause across #341, #342, and #343, and should be recorded as a permanently closed path for the simplified M2P architecture.

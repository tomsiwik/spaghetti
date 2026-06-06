# Peer Review: Fisher-Weighted Adapter Composition

## Experiment Type
Guided exploration (Type 2)

**Framework:** Grassmannian-orthogonal adapters compose near-losslessly;
Frobenius equalization works but the 50% compression factor is empirical.
**Unknown:** Whether diagonal Fisher provides per-adapter importance weights
that differ meaningfully from raw Frobenius norms.

Type classification is correct. The experiment identifies a proven framework
(Grassmannian composition + Frobenius equalization from Finding #279) and a
specific unknown (decorrelation of Fisher from Frobenius). The exploration
successfully narrows the unknown: Fisher importance does NOT decorrelate from
Frobenius for shared-base LoRA adapters. This is a valid guided-exploration
outcome even though the answer is negative.

## Hack Detector

- Fix count: 1 (Fisher-weighted composition). Clean, single-mechanism experiment. No stacking.
- Is MATH.md a proof or a description? Mixed. Theorem 1 has a proof sketch (not
  full QED-level rigor). Theorem 2 is more of an analysis than a proof. The
  Proposition is a decomposition identity (trivially true). The worked example is
  useful. Overall: mechanism analysis with some formal structure, not a tight proof.
- Metric used as evidence: Mixed PPL and Spearman rho. K708 (rho) is the
  strongest kill signal -- it directly tests the proof's core prediction (Fisher
  decorrelates from Frobenius). K706 (PPL) is a secondary consequence. Using both
  is appropriate.
- Kill criteria source: K708 (rho <= 0.9) derived from the proof's prediction P2.
  K706 (PPL < partial eq) derived from the logical consequence: if Fisher provides
  new information, it should improve PPL. K707 (compute < 10 min) is a practical
  bound. All three are well-motivated.

## Self-Test Audit

1. **One-sentence impossibility property:** "Fisher information weights each
   parameter by its task-relevance (gradient sensitivity), making it impossible
   for training-artifact scale to dominate composition weights."
   - This is a claim about what Fisher *should* do, not what it must do. It is
     aspirational rather than an impossibility guarantee. The experiment showed
     this claim is false: scale DOES dominate even with Fisher weighting.
   - However, for a guided exploration, the self-test correctly identified the
     hypothesis to be tested. PASS (with caveat that the hypothesis was refuted).

2. **Cited theorems:** Fisher Merging (Matena & Raffel 2022, arXiv:2111.09832) and
   EWC (Kirkpatrick et al. 2017). Both are real. The Fisher merging theorem applies
   to full-model merging; the adaptation to LoRA composition is the experiment's
   own contribution (correctly flagged as the unknown being explored).
   PASS.

3. **Predicted numbers:** P1: <60s, P2: rho in [0.7, 0.9], P3: PPL < 6.508.
   These are specific and falsifiable. P2 was falsified (rho=1.0). P3 was
   falsified (PPL=7.034). P1 was approximately met (67s vs 60s, but K707 threshold
   of 10 min was met).
   PASS.

4. **Falsification condition:** "If Fisher diagonal has zero variance within each
   adapter, then Fisher = rescaled Frobenius." This is the correct falsification
   condition. The experiment showed that while Fisher has nonzero variance (CV 1.1-1.5
   per key), the *cross-domain* variation in Fisher is too small (~2x) to overcome
   the 400x energy ratio from Delta^2.
   PASS -- good identification of the structural failure mode.

5. **Hyperparameter count:** 1 (N_samples). Acknowledged, practical constraint
   noted. PASS.

6. **Hack check:** "No. This replaces the unprincipled 50% log-compression with a
   single theoretically justified weighting scheme." Honest assessment. PASS.

**Self-Test verdict:** Complete and honest. All 6 items addressed. No evasions.

## Mathematical Soundness

### What holds

**1. The Fisher importance decomposition (Proposition) is correct.**

w_i = sum(F_i[j] * Delta_i[j]^2) = ||Delta_i||_F^2 * E_j[F_i[j] * Delta_i[j]^2 / ||Delta_i||_F^2]

This is algebraically trivial (multiply and divide by ||Delta_i||_F^2). The key
insight -- that constant F_i reduces Fisher to Frobenius -- is correct and is the
hypothesis under test.

**2. The root cause analysis in PAPER.md is excellent.**

The explanation of why Fisher fails (Delta^2 spans 5 orders of magnitude while
Fisher diagonal varies only ~2x across domains because the base model architecture
dominates the Fisher pattern) is mathematically sound and well-supported by data.

**3. Theorem 2's warning about scale double-counting is valid.**

The observation that w_i ~ O(s_i^4) in the naive case and the decision to compute
Fisher on the base+adapter model (not adapter parameters alone) shows good
mathematical awareness.

### What is weak

**4. Theorem 1's "proof sketch" is not a complete proof.**

The proof sketch follows Matena & Raffel 2022 for the full-model case, then
claims that aggregating to per-adapter level via trace inner product gives
alpha_i proportional to w_i. This step skips a critical detail: the original
Fisher merging minimizes per-*parameter* KL divergence. Aggregating to per-
*adapter* importance via trace implicitly assumes that the Fisher diagonal is
uniform enough across the adapter's parameter positions that a single scalar
alpha_i per adapter is sufficient. This is exactly the assumption the experiment
tested (and found to be inadequate, but for a different reason).

The proof sketch is honest about being a sketch and explicitly cites its source.
For a guided exploration this is acceptable -- the point was to test whether the
approach works, not to prove it must work.

**5. Prediction P2 was wrong by a wide margin.**

Predicted: rho in [0.7, 0.9]. Measured: rho = 1.0. The prediction assumed
meaningful Fisher-Frobenius decorrelation. The paper correctly identifies why
this was wrong (shared base model dominates Fisher pattern), but the magnitude
of the miss (perfect correlation) suggests the prediction was not well-grounded
in prior analysis of the Fisher diagonal structure for shared-base models.

This is not a flaw -- the point of a guided exploration is to discover the
unknown. But it is worth noting that a back-of-envelope calculation (Fisher ~
base model sensitivity, which is shared; Delta^2 ~ s_i^2 * ||B_i||^2, which
varies 400x) could have predicted rho would be very high. The 50x ratio between
the dynamic range of Delta^2 (400x) and Fisher (~2x) makes it near-certain
that the product w_i will track Delta^2.

## Prediction vs Measurement

PAPER.md contains a proper prediction-vs-measurement table. Assessment:

| Prediction | Predicted | Measured | Match? | Comment |
|-----------|-----------|----------|--------|---------|
| P1: Fisher compute | <60s | 67s | Approx yes | K707 (10 min) passed. 67s slightly over 60s prediction but within practical range |
| P2: Spearman rho | [0.7, 0.9] | 1.000 | NO | Decisively falsified. rho=1.0 means Fisher adds zero ranking information |
| P3: Mixed PPL | <6.508 | 7.034 | NO | 8.1% worse than partial equalization |
| P4: Domain pattern | High-scale ~5%, low-scale improved | All domains worse | NO | Fisher hurts all 5 domains vs raw sum |
| P5: Fisher/Frob ratio | Highest for finance, lowest for medical | legal:2.40, finance:2.04, math:1.12, med:0.99, code:0.82 | PARTIAL | Direction correct (low-scale higher), but legal beats finance. Magnitude insufficient to overcome energy ratio |

The table is complete and honest. 1 pass, 1 partial, 3 failures. The kill is
clearly justified by the measurements.

## Kill Justification

The kill is well-justified on two independent grounds:

**K708 (rho=1.0, threshold <=0.9):** This is the structural kill. Fisher
importance weights have perfect rank correlation with Frobenius norms. With
N=5 domains, Spearman rho=1.0 means the rankings are identical. The
Fisher/Frobenius ratio does vary (0.82 to 2.40), but this variation is
insufficient to change the rank ordering, let alone compensate for the 400x
energy ratio.

Statistical note: with N=5, Spearman rho has limited resolution (only 120
possible rank orderings). rho=1.0 means all 5 Fisher weights rank identically
to all 5 Frobenius weights. A single rank swap would drop rho to 0.9. So the
kill threshold of 0.9 is actually lenient -- it allows exactly one adjacent
rank swap. The fact that even this does not occur confirms the kill.

**K706 (PPL 7.034 > 6.508):** Fisher-weighted composition is worse than both
partial equalization (6.508) and raw sum (6.585). It hurts all 5 domains.
The Gini coefficient increases from 0.490 (raw) to 0.563 (Fisher), meaning
Fisher amplifies spectral pathology rather than reducing it.

Both kills are clean. The mechanism failed for a well-understood mathematical
reason (shared base model makes Fisher pattern domain-invariant, so
Fisher*Delta^2 is dominated by Delta^2).

## Loose Ends

**1. The Fisher/Frobenius ratio signal exists but is unused.**

The paper's Section "Implications for Future Work" correctly identifies that
alpha_i = w_Fisher_i / ||Delta_i||_F^2 (the ratio, not absolute Fisher) could
normalize out the scale dominance. However, with only 5 data points and a 2x
ratio range, this is a thin signal. The paper correctly notes this uncertainty
without overclaiming.

**2. N_FISHER_SAMPLES=10 vs predicted 20.**

MATH.md predicts N_FISHER_SAMPLES=20, but the code uses 10. PAPER.md notes
this in Limitations. The Fisher per-key CV (1.1-1.5) suggests the estimates
are reasonably stable, so this likely does not invalidate the kill. But it is
a minor discrepancy between plan and execution.

**3. FISHER_SEQ_LENGTH=128 vs MAX_SEQ_LENGTH=256.**

Fisher was computed on shorter sequences than evaluation. The paper acknowledges
this ("Fisher is a local property"). This is unlikely to change the rho=1.0
result but is worth noting.

**4. The "equalization scale" design may be suboptimal.**

The code computes `fisher_eq_scales[d] = fisher_weights[d] / mean_weight`.
This means high-Fisher-weight domains get scales > 1 and low-Fisher-weight
domains get scales < 1. Since Fisher weights track Frobenius norms (rho=1.0),
this AMPLIFIES the existing energy imbalance. The paper correctly identifies
this in its root cause analysis ("Fisher equalization scales amplify imbalance").
However, an alternative design -- using Fisher weights to INVERT the scale
pattern (e.g., `1/fisher_weights[d]`) -- was not tested. This would have been
a natural variant to try. The paper's Section on Fisher/Frobenius ratio
partially addresses this, but does not implement it.

This is not a review deficiency -- the kill at K708 (rho=1.0) means even
cleverer use of Fisher weights cannot fix the fundamental problem that Fisher
adds no ranking information.

## NotebookLM Findings

Manual deep review conducted. Key insights:

1. The experiment correctly identifies that Fisher merging (Matena & Raffel 2022)
   was designed for full-model merging, not shared-base adapter composition.
   This is the central insight: when all adapters share the same base model
   parameters, the Fisher diagonal at shared positions is nearly identical across
   domains, so Fisher*Delta^2 collapses to a scale-dependent measure.

2. The worked example (Section G) is pedagogically useful but the chosen Fisher
   values (uniform 0.01 vs uniform 0.5) are unrealistic -- real Fisher diagonals
   would not be uniform across parameters. The example illustrates the best case
   for Fisher, not the typical case.

3. The assumption chain (Section F) is well-articulated, and Assumption 1
   (diagonal Fisher sufficient) correctly predicts the failure mode. The actual
   failure is not that diagonal Fisher has zero variance, but that cross-domain
   Fisher variance is tiny compared to cross-domain Delta^2 variance.

## Novelty Assessment

The negative result has genuine novelty:

- Fisher merging is widely assumed to be a principled alternative to simple
  averaging for model composition. This experiment demonstrates that for
  shared-base LoRA adapters with heterogeneous scales, diagonal Fisher
  collapses to rescaled Frobenius norms.

- The root cause analysis (shared base model makes Fisher pattern domain-
  invariant) is a contribution to understanding when Fisher merging fails.

- The Fisher/Frobenius ratio observation (low-scale domains have 2x higher
  per-parameter Fisher importance) is a minor positive finding that could
  inform future work.

No prior art was found demonstrating this specific failure mode of Fisher
weighting for heterogeneous-scale LoRA composition.

## Value Assessment

This is a well-executed negative result. The experiment:

1. Asked a clear, well-motivated question (does Fisher decorrelate from Frobenius?)
2. Made specific, falsifiable predictions (P2: rho in [0.7, 0.9])
3. Measured cleanly (rho=1.0, no ambiguity)
4. Identified the root cause (shared base model dominates Fisher)
5. Extracted a useful secondary finding (Fisher/Frobenius ratio varies by domain)
6. Correctly scoped the implications (Fisher merging unsuitable for this setting)

The kill closes off a natural research direction (Fisher weighting as principled
replacement for empirical compression) and the impossibility structure (shared-
base Fisher pattern is domain-invariant) prevents wasting compute on variants
of the same idea.

## Macro-Scale Risks (advisory)

Not applicable -- experiment killed. However, the insight that shared-base
models make Fisher patterns domain-invariant likely holds at any scale, since
the mechanism (base model architecture dominates the Fisher diagonal) is
structural, not scale-dependent.

## Verdict

**KILL**

The kill is clean, well-justified, and well-understood. Both kill criteria
(K706: PPL regression, K708: rho=1.0) are decisively triggered. The root
cause analysis is thorough: diagonal Fisher on shared-base LoRA adapters
is dominated by the shared base model's parameter sensitivity, making
Fisher*Delta^2 approximately proportional to Delta^2 = Frobenius energy.

The experiment is a model of how to execute and document a negative result.
The MATH.md framework is adequate for a guided exploration (not a
verification), the predictions were specific and falsifiable, and the
PAPER.md honestly reports all failures with a clear mechanistic explanation.

No revisions required. The finding should be recorded as killed with the
impossibility structure: "Diagonal Fisher on shared-base models is domain-
invariant (base architecture dominates), so Fisher*Delta^2 collapses to
Frobenius energy when adapter scale heterogeneity exceeds Fisher variation."

# Peer Review: Purpose-Trained Module Adapters (Re-Review)

## Experiment Type
**Guided exploration (Type 2).** Corrected from prior claim of Type 1 (verification).

Proven framework: module separability (Finding #300), domain-optimal module sets
(Finding #304), SFT training with response-only masking (Finding #180).
Unknown explored: does training with the deployment module set produce better
B-matrices than post-hoc ablation of full-module adapters?
Unknown narrowed: co-adaptation during full-module training is beneficial for
behavioral quality. Module selection is a serving optimization, not a training
decision.

This meets Type 2 requirements: proven framework stated, unknown identified precisely,
experiment narrows the unknown to a directional answer.

## Hack Detector
- Fix count: **1** (single mechanism: train with deployment module set). Clean.
- Is MATH.md a proof or a description? **Mechanism analysis with proof sketch.** Labeled honestly as "proof sketch" (MATH.md line 109). Acceptable for Type 2 guided exploration, which does not require formal Theorem/Proof/QED.
- Metric used as evidence: PPL and factual_recall behavioral score. PPL explicitly shown uncorrelated with behavioral (r=0.08). Behavioral is a keyword-overlap proxy. The PPL-behavioral dissociation is itself a finding.
- Kill criteria source: K778 threshold (0.39) from Finding #304's post-hoc result. Reasonable empirical baseline comparison. K779 threshold (3.43) also from Finding #304. K780 now correctly flagged as non-discriminating.

## Self-Test Audit

All 6 self-test items present and completed. No blanks.

1. **One-sentence impossibility property:** "Training with the deployment module set eliminates gradient mismatch." Acceptable -- describes the key mechanism under test. Not a formal impossibility property, but adequate for Type 2.

2. **Cited theorems:** Finding #300, Finding #304, PLoP (arXiv:2506.20629), Geva et al. (arXiv:2012.14913). All real references. Note: PLoP and Geva predicted the direction that was falsified (purpose-training should help), which MATH.md acknowledges in the self-test item 4 (falsification condition). The experiment is honest about this.

3. **Predicted numbers:** P1 (medical behavioral >= 0.39), P2 (math PPL <= 3.43), P3 (code behavioral >= 0.25), P4 (5-15% improvement), P6 (cosine < 0.95). Specific and falsifiable. PASS.

4. **Falsification condition:** "If purpose-trained attn-only underperforms post-hoc attn-only, then co-adaptation is BENEFICIAL." Well-stated. The experiment observed exactly this. PASS.

5. **Hyperparameter count:** 0 new. Correct. PASS.

6. **Hack check:** "No. This resolves a specific confound (Limitation 2) in Finding #304." Clean single-question experiment. PASS.

## Prior Review Fix Verification

### Fix 1: results.json verdict corrected
**APPLIED CORRECTLY.** Line 177 now reads: `"coadaptation_verdict": "H2 (co-adaptation): post-hoc outperforms purpose-trained on behavioral; co-adaptation BENEFICIAL"`. This matches PAPER.md's analysis.

### Fix 2: Experiment type reclassified to Type 2
**APPLIED CORRECTLY in MATH.md and PAPER.md.** Both documents now state "Guided Exploration (Type 2)" with the proven framework and unknown clearly identified.

**Minor residual inconsistency:** results.json line 5 still says `"type": "verification"` and run_experiment.py docstring line 4 still says "Verification (Type 1)". These are cosmetic (the authoritative documents are MATH.md and PAPER.md), but should be cleaned up if convenient. NOT BLOCKING.

### Fix 3: Finding status "supported" confirmed
**APPLIED CORRECTLY.** PAPER.md uses "supported" language throughout. Appropriate for Type 2 with directional evidence from N=5 behavioral eval at single seed.

### Fix 4: K780 marked as non-discriminating
**APPLIED CORRECTLY.** PAPER.md lines 147-149 explicitly state K780 is non-discriminating because code uses identical module sets for both conditions, with B-matrix cosine = 1.0 and norm ratio = 1.0.

### Fix 5: Scale confound added as Limitation 5
**APPLIED CORRECTLY.** PAPER.md lines 168-173 document the scale confound: purpose-trained B-matrices have 21-37% larger norms, meaning effective perturbation at s=20 is larger. No scale sweep performed. This is correctly flagged as an uncontrolled confound.

## Mathematical Soundness

### Type 2 Assessment

For Type 2 (guided exploration), the mathematical requirement is: state the proven framework and identify the unknown. MATH.md does both clearly.

The "Theorem 1 (Gradient Mismatch)" is labeled as a proof sketch, not a formal proof. This is appropriate for Type 2. The sketch correctly identifies that gradient flows differ between full-module and attn-only training (the MLP perturbation changes the hidden states that attention B-matrix gradients are computed against). The O(...) bound is not numerically evaluated, but for Type 2 the purpose is to identify the mechanism and set up discriminating predictions, not to prove a tight bound.

The mechanism analysis correctly sets up two competing hypotheses (H1: independence, H2: co-adaptation) and designs predictions that discriminate between them. The experiment conclusively selects H2 with a twist: co-adaptation is beneficial (the opposite of the MATH.md prediction that purpose-training would be better).

### What the proof sketch gets right
- Gradient mismatch exists (confirmed: B-matrix cosine 0.908-0.938, well below 1.0)
- The mismatch is substantial at s=20 (confirmed: norm ratios 1.21-1.37)
- H1 (independence) is rejected (confirmed: >5% relative difference on behavioral)

### What the proof sketch gets wrong
- Predicted purpose-training would be better or equal. It was worse on behavioral.
- PLoP and Geva references predicted the wrong direction.
- The explanation for WHY co-adaptation helps (MLP-as-memory routing knowledge) is speculative and post-hoc. No formal argument supports it.

### Assessment
The proof sketch is adequate for Type 2. It identified the unknown, made falsifiable predictions, and the experiment narrowed the unknown. The fact that the directional prediction was wrong does not invalidate the framework -- it is the expected outcome of guided exploration where the answer is genuinely unknown.

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table (lines 17-24). Complete and honest.

| Prediction | Measured | Match |
|-----------|----------|-------|
| P1 (K778): medical behavioral >= 0.39 | 0.333 | FAIL |
| P2 (K779): math PPL <= 3.43 | 3.393 | PASS |
| P3 (K780): code behavioral >= 0.25 | 0.865 | PASS (non-discriminating) |
| P4: Purpose-trained outperforms 5-15% | medical -28.7%, math -12.5% | FAIL (opposite direction) |
| P5: Independence if < 5% diff | medical -28.7%, math -12.5% | FAIL |
| P6: B-matrix cos < 0.95 | 0.925 | PASS |

The experiment is honest about the K778 FAIL and draws the correct reversed conclusion: co-adaptation is beneficial. This is exactly how guided exploration should work -- the experiment resolved the unknown, even though the answer was the opposite of the initial prediction.

## Remaining Issues (Non-Blocking)

### 1. Minor type inconsistency in ancillary files
results.json says `"type": "verification"` and run_experiment.py says "Verification (Type 1)". The authoritative documents (MATH.md, PAPER.md) are correct. Cosmetic fix recommended.

### 2. Post-hoc explanation is speculative
The PPL-behavioral dissociation explanation (PAPER.md lines 119-131: "attention B-matrices learned WHERE to look for domain content") is plausible but speculative. This is flagged as an interpretation, not a proven mechanism. For Type 2, this is acceptable -- it generates a follow-up hypothesis. No action required.

### 3. The scale confound remains unresolved
Limitation 5 correctly flags that purpose-trained B-matrices have larger norms (21-37%) and the optimal scale may differ. This is the strongest alternative explanation for the behavioral deficit: the purpose-trained adapters may be over-perturbing at s=20. A scale sweep for purpose-trained adapters would strengthen the finding. However, for micro-scale directional evidence, this is acceptable as an acknowledged limitation. Not blocking.

### 4. N=5 behavioral evaluation
Already acknowledged as Limitation 1. The direction is consistent across two independent domains (medical -28.7%, math -12.5%), which adds credibility. Not blocking for "supported" status.

## NotebookLM Findings

Not used for this re-review. The documents are short and the prior review was thorough. The 5 fixes are verifiable by direct inspection.

## Novelty Assessment

The finding that co-adaptation during full-module LoRA training benefits behavioral quality even when MLP modules are removed at serving time is a useful negative result. It answers a natural question arising from Finding #304 (post-hoc ablation). The B-matrix divergence measurement (cosine 0.908-0.938, norm ratio 1.21-1.37) provides quantitative characterization of co-adaptation effects in ternary LoRA that I have not seen measured elsewhere.

The practical implication -- train with all modules, select at serving time -- is consistent with standard LoRA practice but now has experimental evidence in the ternary adapter setting.

## Macro-Scale Risks (advisory)

1. **Scale dependence of co-adaptation benefit.** At smaller perturbation scales (s < 5), gradient mismatch shrinks. The co-adaptation benefit may vanish or reverse. Not blocking for micro.

2. **Norm amplification at scale.** The 21-37% norm increase in purpose-trained B-matrices suggests the optimizer compensates aggressively. At production scale with larger models, this amplification pattern warrants monitoring.

3. **Behavioral eval needs higher N.** N=5 is adequate for micro directional signal. Macro requires N >= 50 for statistical confidence.

## Verdict

**PROCEED**

All 5 fixes from the prior review were applied correctly to the authoritative documents (MATH.md, PAPER.md, results.json). The experiment meets Type 2 (guided exploration) requirements: proven framework stated, unknown identified precisely, experiment narrows the unknown to a directional answer.

The finding -- co-adaptation during full-module training is beneficial for behavioral quality, making module selection a serving optimization not a training decision -- is directionally supported by consistent evidence across two independent domains (medical, math), with acknowledged limitations (N=5, single seed, uncontrolled scale confound).

**Recommended finding status: supported.**

Justification: Type 2 guided exploration that successfully narrowed an unknown. Directional evidence consistent across domains. Limitations honestly acknowledged. The "supported" threshold ("proof mostly verified, or exploration narrowed an unknown") is met.

### Minor cleanup (non-blocking):
- Update `results.json` line 5 from `"type": "verification"` to `"type": "guided_exploration"`
- Update `run_experiment.py` docstring from "Verification (Type 1)" to "Guided Exploration (Type 2)"

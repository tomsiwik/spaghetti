# Peer Review: m2p_cross_domain_graph (RE-REVIEW after 5 fixes)

## Experiment Type
Guided exploration (self-declared, correct classification)

## Hack Detector
- Fix count: 3 options (A, B, C) + dissolve-recrystallize cycle = 2 mechanisms with 3 variants. Self-Test #6 acknowledges this is scope creep. Not blocking, but noted.
- Is MATH.md a proof or a description? **Theorem 1 is a re-cited proof (QR orthogonality, QED by construction). Conjectures 2 and 3 are correctly labeled as conjectures.** The revision honestly distinguishes proven from conjectured.
- Metric used as evidence: Quality ratio = (base_loss - adapted_loss) / (base_loss - sft_loss). Reasonable proxy. Guard condition described but NOT implemented (see below).
- Kill criteria source: K863 from transfer prediction. K864 from enrichment conjecture. K865 from Theorem 1. K866 is an arbitrary threshold (>50% median quality) that passes trivially.

## Assessment of the 5 Requested Fixes

### Fix 1: Self-Test section added to MATH.md -- PASS
All 6 items are completed with substantive answers. The Self-Test is honest: it acknowledges all 5 hyperparameters are "unjustified by theory," that Conjecture 2 was refuted, and that Options B and C are scope creep. This is a model Self-Test section.

### Fix 2: Theorem 2 -> Conjecture 2 (REFUTED), Theorem 3 -> Conjecture 3 (UNSUBSTANTIATED) -- PASS
Both are correctly reclassified. The circularity of Conjecture 2's proof sketch is explicitly identified ("the hidden assumption 'M2P quality ratio is roughly constant' is exactly what the conjecture claims to prove"). The empirical refutation is stated. Conjecture 3 correctly notes points 2-3 are "assertions without proof."

### Fix 3: Parity regression analyzed (10 adapters x scale mismatch = 25x amplification) -- PASS
PAPER.md contains a thorough analysis: the parity enriched base loss went from 0.59 to 3.73 (6.3x regression), root cause quantified as PROMOTE_SCALE=5 / LORA_SCALE=2 = 2.5x amplification per adapter x 10 adapters = 25x total effective scale. This is well beyond Finding #333's single-adapter validation. The fix recommendation (PROMOTE_SCALE / N) is sensible.

### Fix 4: Quality ratio with parity exclusion guard -- PARTIAL PASS (see below)
PAPER.md describes the guard condition `(base_loss - sft_loss) < 0.1` and reports a recalculated median of 93.55% over 4 domains (parity excluded). The arithmetic is correct: the 4 non-parity quality ratios are [0.885, 0.9154, 0.9558, 1.0326], median = (0.9154 + 0.9558) / 2 = 0.9356 = 93.56%.

**However, the guard condition is NOT implemented in the code.** The run_experiment.py (line 956) still uses `quality = (base_losses[name] - rl) / (base_losses[name] - sft_losses[name] + 1e-8)` with no exclusion logic. The results.json reports `median_quality_ratio: 0.9154` (91.54%), computed over all 5 domains. The 93.55% figure in PAPER.md is a post-hoc manual recalculation, not a measurement from the actual experiment.

This matters for K866: the code and results.json say 91.54% (passing the >50% threshold easily). PAPER.md says 93.55% (also passing). Both pass K866, so the verdict is unaffected. But the discrepancy between the experiment's actual output and the paper's claim should be noted. The PAPER.md should either: (a) report the code's actual output (91.54%) with a note that parity exclusion would yield 93.55%, or (b) re-run with the guard condition implemented. Currently the paper and code disagree.

### Fix 5: Directional limitation clarified (10 unidirectional pairs only) -- PASS
Both MATH.md (new section "Directional Asymmetry Limitation") and PAPER.md (Limitation #1) explicitly state that only 10 unidirectional pairs (a->b where a < b by index) were tested, the reverse direction is NOT trained, and the shared slot design prevents per-direction specialization. This is honest and well-documented.

## Mathematical Soundness

### Theorem 1 (Grassmannian Orthogonality) -- PASSES
This is a re-application of a proven result (Finding #3, #341 K848) to an extended slot set (15 instead of 5). The proof is correct:
- QR construction guarantees A_i^T A_j = 0 for all i != j
- Cyclic trace permutation: trace(A_i^T B_i^T B_j A_j) = trace(B_j (A_j A_i^T) B_i^T) = 0
- Capacity check: 15 slots x rank 4 = 60 <= 256 dimensions

Measurement confirms: max_cos = 1.02e-08 across 1050 pairs. This passes trivially because it is structural.

### Conjecture 2 (Enrichment Monotonicity) -- CORRECTLY MARKED AS REFUTED
The circularity is identified, the refutation is documented, the root cause (25x scale amplification) is quantified. No issues.

### Conjecture 3 (Slot Recycling) -- CORRECTLY MARKED AS UNSUBSTANTIATED
Honestly labeled. Not tested. Appropriate for a "future work" callout.

### Quality Ratio Denominator -- ISSUE REMAINS
The `+ 1e-8` epsilon in the code prevents division-by-zero but does not prevent the pathological -2200% values. The PAPER.md describes a `(base_loss - sft_loss) < 0.1` guard but the code does not implement it. The experiment's actual median (91.54%) includes parity at the bottom of the sorted list, "passing" only because the median of 5 values takes the 3rd element and the pathological value sorts below. This is fragile but not incorrect for the reported result.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Assessment:

| Prediction | Predicted | Measured | Verdict |
|---|---|---|---|
| Cross-domain useful pairs | 3-5/10 | 8/10 | Exceeded (poorly calibrated, not wrong) |
| Useless pairs | 3-5/10 | 2/10 | Only parity targets fail |
| Enriched base improvement | 5-15% | +36.3% repeat, +7.2% reverse, -3% arithmetic, parity 6.3x regression | Mixed -- 2 domains improved significantly, 1 regressed slightly, 1 catastrophically |
| Option B > Option A | Yes | No (A: 91.5% vs B: 87.7%) | FAIL |
| Option C >= max(A,B) | Marginal | No (A > C) | FAIL |
| Grassmannian cos | < 1e-8 | 1.02e-08 | PASS (by construction) |
| Per-domain quality >93.3% | Yes | 2/4 exceed, 2/4 below (excluding parity) | FAIL as stated |

Four of seven substantive predictions are wrong. The correctly predicted items (Grassmannian cos, broad cross-domain transfer) are either structural (guaranteed by construction) or directionally correct but poorly calibrated. The framework made wrong predictions about transfer direction (B>A) and enrichment universality (>93.3% all domains).

This is acceptable for a guided exploration: the experiment narrowed the unknown (which option works best, which domains benefit). The wrong predictions are informative findings, not methodology failures.

## Self-Test Audit

1. **One-sentence impossibility property:** "Grassmannian QR orthogonality makes parameter-space interference between any two adapters geometrically impossible." Correct and genuinely one property. Honestly notes that quality regression after dissolve has NO proven impossibility. PASS.

2. **Cited theorems:** Finding #3 / K848 (Grassmannian orthogonality via QR). Correctly noted as classical linear algebra. Conjecture 2 correctly marked as NOT grounded. PASS.

3. **Predicted numbers:** |cos| = 0, measured < 1e-8. Conjecture 2 predicted >=5% quality improvement, ACTUAL: 6.3x regression for parity. 3-5/10 useful pairs, ACTUAL: 8/10. These are specific and falsifiable. PASS.

4. **Falsification condition:** Theorem 1 falsified if |cos| > machine epsilon. Conjecture 2 already falsified. Targets the proof, not just the experiment. PASS.

5. **Hyperparameter count:** 5, all acknowledged as unjustified by theory. Honest. PASS.

6. **Hack check:** Acknowledges Options B and C are scope creep relative to core question. PASS.

All 6 items complete and honest. No blanks or evasions.

## Kill Criteria Evaluation

| Criterion | Code Result | PAPER.md Claim | Correct? |
|---|---|---|---|
| K863: >=3/10 useful pairs | 8/10 (all options) | 8/10 | PASS -- correct |
| K864: >=3/5 domains improved | 3/5 (Option A) | 3/5 | PASS -- correct |
| K865: max cos < 1e-5 | 1.02e-08 | 1.02e-08 | PASS -- correct |
| K866: median quality >50% | 91.54% (5 domains) | 93.55% (4 domains, parity excluded) | PASS -- both exceed threshold by wide margin; the discrepancy is cosmetic |

All kill criteria evaluations are correct in outcome. K866 has a reporting discrepancy (code says 91.54%, paper says 93.55%) but both pass the >50% threshold by a wide margin, so this does not affect the verdict.

## Novelty Assessment

Cross-domain transfer via separate LoRA adapters is well-studied. The Grassmannian slot allocation + M2P generation + dissolve-recrystallize cycle is a novel combination. ReLoRA (arXiv:2307.05695) does merge-and-continue but not with M2P-generated cross-domain adapters on Grassmannian slots.

The genuine novel findings are:
- Cross-domain transfer is broad, not sparse (8/10 pairs, only parity targets fail)
- Option A (cross-prediction) beats Option B (residual), contrary to intuition
- Dissolve at 25x effective scale is catastrophic for near-optimal domains
- Dissolve selectively helps high-base-loss domains (repeat: +36.3%)

No prior findings in the project test this exact question. No redundancy.

## Macro-Scale Risks (advisory)

1. **Multi-adapter dissolve scale.** The 25x amplification from merging 10 adapters at PROMOTE_SCALE=5 is a fundamental problem. At macro scale with more domain pairs, this gets worse. Need PROMOTE_SCALE / N or validated per-N scaling.

2. **Near-optimal domain vulnerability.** Any domain already near SFT quality will be damaged by large-scale dissolve. Macro systems with heterogeneous domain competence will hit this.

3. **Quality ratio metric.** Needs the guard condition implemented, not just described. At macro scale, more domains may have near-zero denominators.

4. **Directional asymmetry.** At macro scale, asymmetric transfer is likely common (e.g., "medical helps legal" != "legal helps medical"). The symmetric slot design would need replacement.

## Remaining Issues (non-blocking)

1. **K866 reporting discrepancy.** PAPER.md claims 93.55% (4 domains, parity excluded) but the code/results.json report 91.54% (5 domains). The guard condition `(base_loss - sft_loss) < 0.1` is described but not implemented. Both values pass K866 (>50%), so this is cosmetic. However, the PAPER.md should note that 93.55% is a post-hoc recalculation, not the experiment's actual output, or the code should be updated to implement the guard.

2. **Kill criteria remain weakly derived.** K863 (>3/10 useful) and K866 (>50% median) are arbitrary thresholds, not derived from the proof. This is acceptable for guided exploration but should be acknowledged.

## Verdict

**PROCEED**

### Justification

All 5 blocking fixes from the previous review are properly applied:
1. Self-Test: complete, honest, no evasions
2. Theorem 2/3 -> Conjecture 2/3: correctly reclassified with status annotations
3. Parity regression: thoroughly analyzed with root cause (25x effective scale)
4. Quality ratio: parity exclusion described with guard condition (implementation gap is cosmetic -- both metrics pass K866 by wide margin)
5. Directional limitation: explicitly documented in both MATH.md and PAPER.md

Theorem 1 (the only actual theorem) holds and is verified. Conjecture 2 is honestly marked as refuted. Conjecture 3 is honestly marked as unsubstantiated.

The experiment produces genuine findings:
- Cross-domain transfer is broad (8/10 pairs)
- Option A > Option B (contrary to prediction, informative)
- Dissolve at high effective scale is catastrophic for near-optimal domains
- Dissolve selectively benefits high-base-loss domains

Finding #353 status should be capped at **provisional** because:
- The core theoretical contribution (Conjecture 2, enrichment monotonicity) was refuted
- The genuine findings are empirical without a backing theorem
- This is appropriate for a guided exploration that narrowed unknowns but did not verify a proof

The one remaining issue (K866 code vs paper discrepancy) is non-blocking because both values pass the threshold by a wide margin. An ideal revision would implement the guard condition in code, but this is cleanup, not a scientific concern.

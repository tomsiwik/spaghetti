# Peer Review: Boundary Detection via Sliding-Window Domain Classification

## Experiment Type
Guided exploration (Type 2)

**Framework:** MATH.md correctly identifies the proven framework (Finding #305: segment isolation eliminates cross-attention contamination, PPL-based classification achieves 95.2% at N=5) and the unknown (optimal window size for boundary detection, end-to-end PPL impact of detected vs oracle boundaries).

This is a legitimate Type 2 experiment. The proven framework is stated. The unknown (window size trade-off, boundary detection accuracy as a function of window size) is well-defined and the experiment narrows it.

## Hack Detector

- Fix count: 1 (sliding window change-point detection). No layered fixes or auxiliary losses. CLEAN.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 has genuine proof structure (Lemma 1, Theorem 1, QED). Theorem 3 is closer to a derivation/description than a rigorous proof -- it hand-waves the NLL additivity across misassigned tokens. Corollary 1's false-positive analysis (the refinement step) is informal. Overall: honest mathematical reasoning, not pure description, but not fully rigorous either.
- Metric used as evidence: F1 for boundary detection (behavioral: can it find boundaries?), PPL gap for end-to-end quality (behavioral: does detection help?), latency for practicality. F1 is reasonable for the boundary detection task. PPL gap is the actual behavioral outcome.
- Kill criteria source: K775 (F1>=0.70) is loosely connected to the proof (predicted 94%, threshold at 70% gives margin). K776 (PPL within 5%) is derived from Theorem 3 (predicted 1.1%, threshold 5x more generous). K777 (latency <5ms) is derived from a back-of-envelope computation in MATH.md. The criteria are derived from the proof, though K775's threshold is very generous compared to the prediction.

## Self-Test Audit

1. **One-sentence impossibility property:** "The sliding window guarantees that the detected change-point is within w/2 tokens of the true boundary, because pure-domain windows on either side of the boundary are classified correctly with probability p^2 > 0.9." This is a single property. PASS.

2. **Cited theorems:** Basseville & Nikiforov 1993 (change-point detection). This is a real reference. However, the experiment does not actually use CUSUM or any formal change-point detection algorithm -- it uses argmax change detection, which is the simplest possible approach. The citation is more contextual than operational. MARGINAL PASS -- not misleading, but the experiment does not use any of the formal machinery from Basseville & Nikiforov (no CUSUM statistics, no sequential probability ratio test, no optimality guarantees from the cited work).

3. **Predicted numbers:** F1 >= 94%, localization error <= 16 tokens, PPL gap <= 1.1%, FP rate <= 0.013. These are specific and falsifiable. PASS.

4. **Falsification condition:** "(a) p(w=32) is much lower than p(w=128), or (b) NLL difference is not approximately additive across misassigned tokens." These target the proof's assumptions, not just the experiment. PASS.

5. **Hyperparameter count:** 2 (window size w, stride s). Acknowledged, with constraints from theory. The optimal w is the Type 2 unknown. PASS.

6. **Hack check:** "No. Single mechanism applied to proven framework." PASS.

Self-Test section is complete with no blanks. PASS.

## Mathematical Soundness

### Lemma 1 (Window Purity): SOUND
Trivial geometric argument. Correct.

### Theorem 1 (Boundary Detection via Argmax Change): MOSTLY SOUND, ONE GAP
The core claim that |tau_hat - tau| <= w/2 with probability >= p^2 is correct as stated. However, there is a subtlety: the proof assumes tau_hat is the CORRECT change detection (true positive), not ANY change detection. The detected boundary could be a false positive change-point that happens to be near the true boundary. The proof conflates "the classification correctly identifies both pure-domain flanks" with "the first detected change-point is at the true boundary." In practice, the first change-point is often a false positive BEFORE the true boundary, as the PAPER.md Phase 2 results show (the implementation uses detected_boundaries[0]).

This is a genuine gap: Theorem 1 proves that the true boundary WILL produce a classification change within w/2, but does not prove that no FALSE change occurs BEFORE it.

### Corollary 1 (F1 Prediction): FLAWED
The corollary predicts F1 >= 94% but the key step -- estimating false positives as (1-p)^2 * (1-1/N) per window pair -- assumes INDEPENDENCE between adjacent window classifications. MATH.md's post-experiment analysis correctly identifies this: overlapping windows have correlated PPL noise, creating bursts of argmax flickering. The independence assumption is the root mathematical error.

Measured false positive rate: 0.26/sequence (at w=64) vs predicted 0.013/sequence. This is a 20x discrepancy. The independence assumption is catastrophically wrong for overlapping windows.

### Theorem 2 (Window Size Optimality): NOT A THEOREM
This is labeled "Theorem" but has no proof. It states the expected boundary error formula but does not derive it or prove any optimality result. It is a modeling assumption, not a theorem. The formula E[|tau_hat - tau|] = w/2 * p(w)^2 + T/2 * (1 - p(w)^2) treats detection as binary (succeed within w/2 or fail to random guess over T), which ignores the case where a false positive is detected near-ish to the true boundary.

### Theorem 3 (PPL Degradation): FLAWED
The proof assumes boundary error only misassigns epsilon tokens at the boundary. It does not account for:
1. False positive boundaries creating entire misrouted segments
2. Short segments having noisier PPL estimation (degrading classifier accuracy)
3. The implementation using only the FIRST detected boundary, which may be a false positive

Predicted 1.1% gap, measured 32.91%. This is a 30x discrepancy. The theorem is not merely imprecise -- it models the wrong failure mode entirely.

The post-experiment correction in MATH.md is honest but still under-predicts (1.6% corrected vs 32.91% measured), acknowledging additional unmodeled effects.

### Latency Estimate: WRONG BY FACTOR OF 800
MATH.md estimated 0.05ms per forward pass of 32 tokens. Actual: ~40ms. This is not a mathematical error in the proof structure -- it is a wrong empirical constant plugged into a correct formula. But K777's kill criterion was derived from this estimate, meaning the criterion was not really "derived from the proof" so much as "derived from a guess about hardware performance."

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Honest reporting:

| Prediction | Measured | Match? |
|-----------|----------|--------|
| F1 >= 94% (w=32) | 61.2% (w=32), 88.2% (w=64) | NO (w=32), NO but close (w=64) |
| Localization <= 16 tokens (w=32) | 10.5 (w=32), 19.6 (w=64) | YES (w=32); YES (w=64, <= w/2=32) |
| PPL gap <= 1.1% | 32.91% | NO (30x off) |
| FP rate <= 0.013/seq | 0.26/seq | NO (20x off) |
| Latency ~4ms | 3017ms | NO (750x off) |

Out of 5 quantitative predictions, only 1 (localization error) matches. The experiment is honest about this, which is commendable.

## Tolerance Inflation Issue

A critical issue in the experimental design: the tolerance parameter for boundary matching is set to `tolerance = w` (line 283 of run_experiment.py), meaning a detected boundary within one FULL window width of the true boundary counts as a true positive. However, MATH.md's Theorem 1 predicts localization within w/2 (half window width). At w=64, the tolerance is 64 tokens -- out of a 256-token sequence with a boundary at position 128, any detected boundary between position 64 and 192 counts as correct. That is 50% of the sequence.

This inflates K775's F1 metric. The 88.2% F1 at w=64 uses tolerance=64, but the theorem predicts accuracy within w/2=32. If the tolerance were set to w/2=32 (matching the theorem's prediction), F1 would likely be lower. PAPER.md does not acknowledge this discrepancy between the theorem's w/2 bound and the experiment's w tolerance.

## NotebookLM Findings

NotebookLM was not used for this review (skipping per workflow -- the mathematical analysis above is sufficient to identify all critical issues).

## Novelty Assessment

Sliding-window domain classification for boundary detection is not novel. MATH.md cites Basseville & Nikiforov 1993 (CUSUM) and Page 1954. The specific application to per-adapter PPL comparison is a straightforward combination of existing ideas. No novelty claim is made beyond the specific application to the SOLE routing pipeline. The experiment correctly frames this as exploration within a known framework, not a novel contribution. PAPER.md cites TM-TREK (arXiv 2404.00899) as relevant prior art for token-level boundary detection.

The real value is empirical: learning that PPL-based sliding-window detection is impractical (too slow, too noisy) is a useful negative result that prunes the search space.

## Macro-Scale Risks (advisory)

1. At N=24 domains (where classification drops to 40%), boundary detection will likely fail entirely -- argmax will flicker constantly.
2. Real text has gradual domain transitions, not synthetic sharp boundaries. The w/2 localization bound becomes meaningless when the "boundary" is a 50-token transition zone.
3. The latency problem (3 seconds per 256 tokens) makes this approach dead on arrival for any real serving scenario, regardless of scale.

## Critical Issues (numbered)

### Issue 1: Finding status SUPPORTED is inappropriate -- should be KILLED

Two of three kill criteria FAIL (K776 and K777). The one that passed (K775 F1=0.882) used inflated tolerance (w instead of w/2) and was tested at w=64, not the predicted-optimal w=32. Moreover, the detected-boundary routing is WORSE than the no-boundary-detection baseline (detected PPL 5.261 vs per-sequence PPL 4.694). The experiment proved that this approach HURTS quality rather than helping.

The PAPER.md conclusion says "boundary detection F1 is viable; the mechanism works in principle but the PPL-based implementation is impractical." But the whole point of K776 was to test whether the mechanism works END-TO-END, and it does not -- it makes things worse. A mechanism that correctly identifies boundaries 88% of the time but produces worse results than not using boundaries at all has not been "supported." The positive F1 result is a mirage created by generous tolerance and disconnected from the actual behavioral outcome (PPL improvement).

### Issue 2: Theorem 3 predicted 1.1%, measured 32.91% -- proof falsified, not "partially confirmed"

MATH.md's post-experiment analysis calls Theorem 1 "CONFIRMED" and Theorem 3 "FALSIFIED." But the overall finding is recorded as "supported." When a proof's central quantitative prediction (the PPL gap that the entire pipeline depends on) is wrong by 30x, the proof has been falsified. The localization bound holding is necessary but not sufficient -- the whole point was to achieve near-oracle PPL, and this completely failed.

### Issue 3: Tolerance = w instead of w/2 invalidates K775 comparison to Theorem 1

Theorem 1 predicts |tau_hat - tau| <= w/2. The experiment uses tolerance = w. At w=64 with T=256 and boundary at position 128, tolerance=64 means any detection between token 64-192 is "correct" -- that is 50% of the sequence. This is too generous and makes the F1 metric misleading as a verification of Theorem 1.

### Issue 4: Phase 2 implementation bug (using first detected boundary)

Line 523 of run_experiment.py: `det_b = detected_boundaries[0]`. If there are false positive boundaries BEFORE the true boundary, the first detected boundary is a false positive, and the entire sequence is split at the wrong position. This is identified in PAPER.md but not in the kill criteria analysis. The correct implementation would use the boundary nearest to the sequence center, or apply minimum segment length filtering. This implementation choice makes K776 results worse than the approach inherently requires.

### Issue 5: Corollary 1's independence assumption is acknowledged but not repaired

The post-experiment analysis correctly identifies that adjacent overlapping windows have correlated noise, violating the independence assumption in Corollary 1. But no corrected bound is provided. For a Type 2 experiment, the exploration should have narrowed the unknown -- instead, it revealed that the mathematical model (independent window classifications) is fundamentally wrong for overlapping sliding windows. This is a useful finding but it means the framework itself needs revision, not just the parameters.

## Verdict

**KILL**

### Justification

1. **2/3 kill criteria FAIL.** K776 and K777 are hard failures by large margins (30x and 600x respectively). K775 passes only with inflated tolerance.

2. **The behavioral outcome is negative.** Detected-boundary routing (PPL 5.261) is WORSE than the no-detection baseline (PPL 4.694). The mechanism, as implemented, actively degrades quality. This is the definition of a killed experiment -- the approach causes harm rather than providing benefit.

3. **Central proof prediction falsified.** Theorem 3 predicted 1.1% PPL gap, measured 32.91%. The independence assumption in Corollary 1 is wrong by 20x on false positive rate. The proof framework does not model the actual failure mode (cascading misrouted segments from false positive boundaries).

4. **Recording this as SUPPORTED sends the wrong signal.** The useful finding (PPL-based sliding-window detection is impractical) is a NEGATIVE result. The correct status is KILLED with the impossibility structure clearly identified: "PPL-based boundary detection requires O(N*W) forward passes (latency-impossible) and produces correlated false positives from overlapping windows that cascade into misrouted segments (quality-impossible)."

### If REVISE were attempted (but KILL is the correct verdict)

1. Change Finding #307 status from SUPPORTED to KILLED
2. Add impossibility structure: "Overlapping sliding-window PPL classification violates the independence assumption required for low false-positive rates. Correlated noise in overlapping windows creates burst false positives that cascade into misrouted segments, making detected-boundary routing WORSE than no-boundary routing."
3. Add failure mode: "PPL-based boundary detection is O(N*W) forward passes (3 seconds for 256 tokens). Even with perfect F1, the latency makes it dead on arrival."
4. Correct tolerance in K775 evaluation to match Theorem 1's w/2 bound, or acknowledge the discrepancy
5. The positive localization result (Theorem 1 confirmed) and the identification of the false-positive cascade failure mode are genuinely useful negative results that should be preserved in the finding -- but as a killed finding, not a supported one

## Closure-Review Verification (2026-04-18)

Audit-rerun closure (researcher this iteration) fixed the PAPER.md verdict line from "supported" → "KILLED" and added a three-theorem closure addendum (C1 latency O(N_adapters*N_windows) forward passes, C2 overlapping-window independence violation, C3 K775 tolerance inflation does not rescue). Re-checked adversarial checklist on the amended artifacts:

- (a) results.json verdict=KILLED, all_pass=false — consistent with DB status=killed ✓
- (c) PAPER.md verdict line now reads "KILLED" on lines 30 and 198 (previously "supported") ✓
- (e–g) KC integrity unchanged; K775 passes only with tolerance=w inflation, K776/K777 hard FAIL — no post-hoc relaxation ✓
- (r) prediction-vs-measurement table preserved ✓

Structural closure (C1+C2) is robust to the two code bugs (tolerance=w → w/2, detected_boundaries[0] → nearest-to-centre): neither fix can change the forward-pass count (C1 600× latency gap) nor the correlated-noise ceiling on overlapping windows (C2 20× FP inflation). Fifth structural closure this audit sweep; extends closure-rule family `base-ceiling-blocks-routing` (Finding #563) to mechanism-cost-floor × correlated-noise-ceiling substrate.

**Closure verdict: KILL (reaffirmed).** Route to analyst.

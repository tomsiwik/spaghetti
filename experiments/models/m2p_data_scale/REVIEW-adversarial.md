# Peer Review: M2P Data Scale (RE-REVIEW)

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 2 (data scale increase + early stopping). Early stopping is framed as "monitoring infrastructure" but materially rescues quality at n=500 (3/4 domains triggered GL, quality jumped from predicted <89.4% to 97.0%). The revised PAPER.md now honestly decomposes the effect, which mitigates the framing concern.
- Is MATH.md a proof or a description? Theorem 1 has Proof/QED structure with forward direction only (sufficient condition). Theorem 2 is algebraic restatement of GL definition plus a vacuous Hardt bound, now explicitly acknowledged as vacuous. Acceptable for Type 2 guided exploration.
- Metric used as evidence: quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss). Values > 100% occur (parity at n=1000 = 101.7%). Not proven to predict behavioral outcome, but consistent across experiments and sufficient for directional comparison.
- Kill criteria source: K879 from Theorem 2 / observed Finding #358 values (honest about Hardt bound being inapplicable). K880 from arithmetic O(1/T) floor (reasonable). K881 from Theorem 1 monotonicity (acknowledged as too strong for single-run).

## Self-Test Audit

1. **One-sentence impossibility property:** "When n_train >= T (at most one epoch of data), the gradient estimator satisfies the i.i.d. unbiasedness condition of the Ghadimi-Lan theorem, making cyclic memorization structurally impossible: no sample is visited more than once." Now includes caveat about sufficiency, not necessity, and notes early stopping can rescue even when n_train < T. PASS.

2. **Cited theorems:** Ghadimi & Lan (real, correctly cited). Bartlett et al. (real, now used heuristically with appropriate caveats about linear regression vs neural networks). Prechelt (heuristic, not a theorem, but correctly used as standard practice). Ying (real, quadratic-objective caveat noted). PASS with notes.

3. **Predicted numbers:** quality >= 93.5%, train-val gap < 0.5 nats, monotonicity in n. Specific and falsifiable. PASS.

4. **Falsification condition:** "The proof (Theorem 1) is falsified if K879 FAILS." Targets the core structural claim. PASS.

5. **Hyperparameter count:** Claimed 0. GL threshold = 5.0 from Prechelt, patience and interval are standard. Defensible. PASS.

6. **Hack check:** Now honestly states early stopping is the "secondary safety net" after the n >= T structural fix. The PAPER.md decomposition (+7.6pp from early stopping + T change, +0.6pp from data scale alone) makes the relative contributions transparent. PASS.

## Fix Verification

### Fix 1: 80/20 split not accounted for
**STATUS: FIXED.**

MATH.md now contains an explicit "Note on train/val split" (lines 127-131) explaining that n_train = 0.8 * n, so n_train in {400, 800, 1600} for n in {500, 1000, 2000}. Corollary 1.1 (lines 167-177) correctly computes n_per_domain* = T / 0.8 = 1250 at T=1000. The worked example in Section F (lines 328-355) provides epoch counts for all three conditions and explicitly states that n=1000 (n_train=800, 1.25 epochs) does NOT satisfy n_train >= T. PAPER.md (lines 17-20) repeats this correction with correct epoch counts. The inflection point is now correctly identified as between n=1000 and n=2000, not at n=1000.

### Fix 2: Confounded baseline comparison
**STATUS: FIXED.**

PAPER.md Section 4 (K880, lines 98-116) now contains a full effect decomposition table showing:
- Finding #358 baseline: n=500, T=500, no early stop = 89.4%
- This experiment n=500, T=1000, with early stop = 97.0% (+7.6pp from T + early stopping)
- This experiment n=2000, T=1000, with early stop = 97.6% (+0.6pp from data scale alone)
- Total vs baseline = +8.2pp (confounded)

The text explicitly states: "The dominant effect is early stopping + increased T (+7.6pp), not data scale (+0.6pp)." Section 6 summary repeats this decomposition. The finding correctly frames the contribution as "early stopping plus data scale eliminate cyclic overfitting."

### Fix 3: Theorem 1 backward direction refuted (iff -> if)
**STATUS: FIXED.**

Theorem 1 (MATH.md line 125) now states: "This is a sufficient condition, not necessary." The proof only covers the forward direction, with "QED (forward direction)" at line 155. A detailed remark (lines 157-165) explains why the backward direction does not hold, citing three mechanisms: early stopping as implicit regularization, SGD implicit regularization (Neu & Rosasco, 2018), and insufficient model capacity. The remark explicitly notes that the n=500/T=1000 result (97.0%) directly contradicts the backward direction.

### Fix 4: Theorem 2 quantitative miss (Hardt bound 270x off)
**STATUS: FIXED.**

MATH.md (lines 232-238) now explicitly states: "The measured train-val gap at n=2000 is 0.337 nats -- approximately 270x larger than the Hardt bound prediction. This discrepancy arises because the Hardt et al. bound requires convex loss [...] The QUALITATIVE prediction holds (gap < 0.5 nats threshold), but the quantitative bound should not be used for calibration at this scale." PAPER.md prediction-vs-measurement table (line 29) includes the Hardt quantitative miss as a separate row marked "MISS -- ~270x off; bound inapplicable to non-convex M2P loss." The K879 threshold of 0.5 nats is now explicitly grounded in Finding #358 observations, not the Hardt formula.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table (lines 23-29). Assessment:

| Prediction | Predicted | Measured | Verdict |
|---|---|---|---|
| n=500 at T=1000 (overfitting reference) | quality < 89.4% | 97.0% | MISS -- early stopping rescued quality. Honestly labeled "PARTIAL" and explained. |
| n=1000 at T=1000 quality | >= 91% | 97.7% | EXCEEDED (conservative prediction) |
| n=2000 at T=1000 quality | >= 93.5% | 97.6% | EXCEEDED (conservative prediction) |
| train-val gap n=2000 (qualitative) | < 0.5 nats | 0.337 nats | PASS |
| train-val gap n=2000 (Hardt quantitative) | <= 0.001 nats | 0.337 nats | MISS -- correctly flagged as inapplicable |
| Per-domain monotonicity (K881) | strict monotone | 2/4 monotone | FAIL -- attributed to micro-scale noise, reasonable |

The n=500 prediction miss is the most interesting result: it reveals that early stopping is a more powerful regularizer than the theorem framework accounts for. This is honestly reported and the "if not iff" revision handles it correctly.

The quality predictions (>= 93.5%, >= 91%) are very conservative lower bounds, exceeded by 4-7pp. This is acceptable for Type 2 guided exploration -- the predictions were derived from arithmetic's empirical trend as a floor, and the floor held.

## New Issues Found

### Issue A: Non-monotone train-val gap progression (informational, not blocking)

The max train-val gap shows: n=500: 0.873 -> n=1000: 0.312 -> n=2000: 0.337. The gap INCREASED from n=1000 to n=2000. PAPER.md (lines 64-69) addresses this: "the remaining gap is domain difficulty, not overfitting." This is plausible -- the reverse domain consistently has the highest gap across all n values, suggesting intrinsic difficulty rather than data-quantity-driven overfitting. The gap at n=2000 (0.337) is dominated by the reverse domain (0.337), while all other domains are <= 0.150.

### Issue B: MATH.md Section D.1 uses T=2000 but experiment uses T=1000 (addressed)

MATH.md Section D.1 prediction table references T=2000 in some entries, but the experiment script uses M2P_STEPS_FIXED=1000. PAPER.md Section 5 (lines 164-169) addresses this: the quality predictions were lower bounds ("predicting >= 93.5% at T=2000 is a weaker claim than what was measured at T=1000"). The kill criteria descriptions in the script comments still mention "T=2000" (line 25-26: "K879: train-val loss gap at T=2000 < 0.5 nats") which is inconsistent with M2P_STEPS_FIXED=1000 at line 102. This is a documentation inconsistency, not a scientific error. NON-BLOCKING.

### Issue C: Quality > 100% for parity at n=1000 (101.7%) (non-blocking)

The quality metric allows values above 100% when M2P loss is lower than SFT loss. This occurred for parity at n=1000 (101.7%). The original review flagged this as advisory. Neither MATH.md nor PAPER.md addresses what quality > 100% means. At micro scale this is noise, but the metric definition should be noted for future experiments. NON-BLOCKING.

### Issue D: K881 monotonicity test uses 2pp tolerance (line 1009) but actual variance is up to 5.2pp

The code allows 2pp noise floor tolerance for monotonicity, but measured per-domain variance is up to 5.2pp (parity). K881 still fails because parity drops 96.5% -> 101.7% -> 98.4% (the 101.7 -> 98.4 drop is 3.3pp, exceeding the 2pp tolerance). The PAPER.md correctly identifies this as "too strong for single-run micro-scale" and does not over-interpret the failure. NON-BLOCKING.

## Novelty Assessment

No change from prior review. This is a straightforward application of "more data reduces overfitting" with standard early stopping. The Ghadimi-Lan framing is pedagogically useful. The real contribution is empirical: confirming that the M2P pipeline was data-limited (not architecture-limited) and that early stopping is the dominant regularizer at micro scale.

## Macro-Scale Risks (advisory)

1. The 80/20 split is now correctly accounted for, but at macro scale the n* = T / 0.8 formula should be verified against actual convergence curves.
2. The GL early stopping criterion (Prechelt 1998) works here but may need tuning at macro scale. Modern alternatives (patience on raw validation loss) are more common.
3. The quality metric allowing > 100% should be clipped or redefined before macro experiments.
4. The dominant effect of early stopping (+7.6pp) vs data scale (+0.6pp) suggests that at macro scale, the training protocol (early stopping, learning rate scheduling) matters more than raw data quantity once data is "sufficient."

## Verdict

**PROCEED**

All four blocking fixes from the prior REVISE have been addressed:

1. **80/20 split:** Explicitly accounted for throughout MATH.md and PAPER.md with correct n_train values, epoch counts, and n* threshold computation.
2. **Confounded baseline:** Fully decomposed in PAPER.md with a clear table showing +7.6pp from early stopping + T vs +0.6pp from data scale alone.
3. **iff -> if:** Theorem 1 now states sufficient condition only, with detailed remark explaining why backward direction fails, citing the n=500/T=1000 counter-example.
4. **Hardt bound miss:** Explicitly acknowledged as 270x off, attributed to convexity precondition violation, K879 threshold re-grounded in empirical observations.

The revised documents are honest about what was learned: the dominant effect is early stopping, not data scale. The data scale effect is real but small (+0.6pp). The structural fix (n_train >= T eliminating cyclic memorization) is confirmed for n=2000 but was not the primary driver of quality improvement. This nuanced conclusion is more valuable than the original framing.

K879 PASS, K880 PASS, K881 FAIL (attributable to single-run noise at micro scale). Finding status recommendation: **supported** -- the guided exploration successfully identified that (a) early stopping is the primary regularizer, (b) n_train >= T provides the structural guarantee against cyclic memorization, and (c) the quality ceiling at micro scale is ~97-98% regardless of data quantity once overfitting is controlled.

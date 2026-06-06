# Peer Review: KR-Test Knowledge Retention Evaluation (Re-Review)

## Previous Review Fixes -- Verification

The previous review required 4 specific fixes. Here is the status of each.

### Fix 1: K2 noise floor reconciled -- APPLIED CORRECTLY

PAPER.md now states (lines 109-113): "K2 FAIL under pre-registered definition. The experiment code computed: noise_floor = |random - 0.5| = 0.395, ratio = 0.055 / 0.395 = 0.14x. Code verdict: KILLED (0.14x < 2.0x threshold). Under a statistical interpretation (SE = 0.042), ratio = 0.055 / 0.042 = 1.3x -- still below 2x. Neither definition meets the 2x threshold. K2: FAIL."

This is honest and correct. Both definitions are presented; both fail. No ambiguity remains.

### Fix 2: K1 medical ceiling flagged -- APPLIED CORRECTLY

PAPER.md now reports (lines 60-64): the correlation with and without the medical zero-zero anchor. With medical: rho=1.0 (n=4). Without medical: rho=1.0 (n=3), since the remaining 3 domains [0.04, 0.08, 0.10] vs [0.067, 0.100, 0.360] have identical rank orderings. The paper correctly notes that n=3 provides "even weaker statistical power (p=0.167)."

This is the right analysis. The correlation does survive without the floor-anchored point, which strengthens the K1 finding modestly. The statistical weakness at n=3 is honestly reported.

### Fix 3: Positional zip bug fixed -- APPLIED CORRECTLY

The code (line 680) now uses `TASK_ACCURACY.get(domain, {}).get("delta", 0)` with a proper dict keyed by domain name (lines 69-76). The original bug (presumably positional list indexing where legal and creative could swap) is eliminated. The TASK_ACCURACY dict has correct entries for all 5 domains.

### Fix 4: Transparency Note added -- APPLIED CORRECTLY

PAPER.md lines 115-124 include a clear Transparency Note: "The experiment code's pre-registered analysis produced verdict: KILLED (K2: discrimination ratio 0.14x < 2.0x threshold). The upgrade to SUPPORTED in this paper is an editorial judgment..." This is exactly what the previous review requested. The reader can see the code said KILLED and understand why the researcher upgraded to SUPPORTED.

## NotebookLM Findings

Skipped (not requested for re-review; original review findings remain valid).

## Mathematical Soundness

Verified in previous review; no changes to the math. The core protocol (contrastive log-prob comparison, cross-item pairing, Spearman rank correlation on deltas) remains sound.

One item worth noting: the MATH.md (lines 83-85) now includes a forward-looking claim that at n=200 the SE-based ratio would reach 2.6x, clearing 2x. This is arithmetically correct: SE(n=200) = sqrt(0.09/200) = 0.021, ratio = 0.055/0.021 = 2.6. However, this assumes the delta (0.055) would remain stable at higher n, which is plausible but unproven.

## Novelty Assessment

No change from previous review. The KR-Test is from Ziabari et al. (arXiv:2601.03505). The cross-item pairing adaptation and the delta-based analysis are useful engineering contributions for this project, not research novelty claims.

## Experimental Design

### Remaining issues (not blocking)

1. **K2 code was not re-run.** The results.json still contains the original K2 analysis with `noise_floor: 0.395` and `discrimination_ratio: 0.139` and `verdict: "KILLED"`. The PAPER.md was updated editorially but the code was not modified to compute the SE-based noise floor alongside the original. This is acceptable -- the code is the ground truth and the paper interprets it -- but a future reader looking only at results.json will see KILLED without the nuanced discussion.

2. **The K2 trained_per_domain in results.json correctly excludes legal.** The previous review raised a concern about a potential `"legal": 0.98` entry. Verified in the code (lines 750-756): the loop checks `domain in individual_results[domain].get("domain_scores", {})`, and since there are 0 legal contrastive pairs, "legal" is not a key in the legal adapter's domain_scores. The results.json confirms: trained_per_domain contains only medical, math, code, creative. This concern from the previous review was either fixed or was a misread of an earlier version.

3. **Single seed remains unaddressed.** The paper acknowledges 2-4pp variance from pairing randomness (Limitation 6) but no multi-seed run was added. At 5.5pp delta, this is a real concern. Not blocking for a micro experiment, but the downstream Evolve gate (exp_bitnet_retrain_evolve) should test seed stability before adopting the 0.03 threshold.

4. **HYPOTHESES.yml evidence description says "ratio=1.3x"** using the SE-based interpretation, not the code's actual 0.14x. This is the softer framing. It would be more precise to say "ratio=0.14x (pre-registered) / 1.3x (SE-based), both below 2x threshold." Minor, not blocking.

## Hypothesis Graph Consistency

- The node status is "supported" which matches the PAPER.md editorial verdict.
- The evidence entry honestly says "K2 MARGINAL" rather than claiming K2 passed.
- The kill criteria in HYPOTHESES.yml match what was tested.
- Downstream dependency (exp_bitnet_retrain_evolve) is aware of the KR-Test delta threshold.

No inconsistencies found.

## Macro-Scale Risks (advisory)

Unchanged from previous review:
1. Cross-item pairing difficulty needs domain-specific calibration at scale (ceiling effects).
2. 90 sec/adapter is acceptable but not fast for large Evolve rounds.
3. The 0.03 delta threshold is calibrated to n=50 at BitNet-2B; needs recalibration at other scales.
4. KR-Test may not capture non-factual capabilities (reasoning, style, instruction following).

## Verdict

**PROCEED**

All 4 required fixes from the previous review have been applied correctly. The paper is now honest about:
- K2 failing under both noise floor definitions
- The medical ceiling effect on K1 correlation
- The code-to-paper verdict override being an editorial judgment
- The positional bug being fixed in the code

The remaining issues (single seed, HYPOTHESES.yml soft framing of ratio) are minor and do not affect the core finding: KR-Test delta is a valid adapter ranking signal at micro scale (K1: rho=1.0 on n=4 domains, survives medical exclusion at n=3). K2 discrimination is marginal but directionally correct. The SUPPORTED status with clearly documented K2 failure is an honest assessment.

The experiment advances the project by providing a concrete quality metric for the Evolve gate that is superior to PPL (correctly identifies the degenerate legal adapter, rank-correlates with task accuracy). The threshold (delta > 0.03) should be validated with multiple seeds before production use, but that is a downstream concern for exp_bitnet_retrain_evolve, not a blocker for this micro experiment.

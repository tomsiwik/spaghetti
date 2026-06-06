# Peer Review: m2p_layer_depth_36 (RE-REVIEW after REVISE fixes)

## Experiment Type
Frontier extension (Type 3) -- correctly identified. Prior proven result: Finding #363 (L<=16). Extension: L=36 (Qwen3-4B depth).

## Hack Detector
- Fix count: 0 new mechanisms. Clean parameter sweep of proven recipe. No flags.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 is a valid (trivial) proof. Theorem 2 is a necessary condition proof. Theorem 3 is a curve fit dressed as a theorem -- but MATH.md correctly self-undermines it ("Confidence in this model: LOW"). Acceptable for Type 3 frontier extension.
- Metric used as evidence: quality_ratio (median across valid domains). Well-defined proxy.
- Kill criteria source: K894 derived from Finding #363 precedent. K895 from GL framework. K896 is a replication sanity check. Reasonable for frontier extension.

## Verification of 4 Original Fixes

### Fix #1 [MAJOR]: PAPER.md Section 5 rewritten -- per-domain quality table added, non-monotone claim removed

**STATUS: PROPERLY APPLIED.**

Section 5 now includes Table 4 showing per-domain quality (sort, reverse, arithmetic) across all L values. The old "non-monotone trajectory" claim is replaced with: "Sort domain shows monotone improvement with plateau: 78.6% -> 88.7% -> 89.1%" (line 129) and "The per-domain view shows clean monotone-then-plateau behavior consistent with the Aghajanyan intrinsic dimensionality model" (line 137). The artifact of arithmetic inclusion/exclusion at the parity guard boundary is explicitly identified (lines 119-120, 133-137). This is a substantive improvement -- the real scaling signal is now clearly separated from the parity guard noise.

### Fix #2 [MODERATE]: Code kill criteria IDs fixed -- K891/K892/K893 replaced with K894/K895/K896

**STATUS: PARTIALLY APPLIED.**

The main kill criteria logic (lines 1099-1113) now correctly uses K894/K895/K896 with thresholds 0.85, 0.7, and 0.50. The evaluate_kill_criteria function (lines 1160-1249) uses the correct IDs and thresholds. The final summary (lines 1387-1390) uses the correct IDs.

However, the experiment banner at lines 1306-1309 STILL references K891/K892/K893 with the old thresholds and old descriptions:
```
log(f"  K891: Option A quality >= 85% at L=16 -> PASS")
log(f"  K892: Option B quality >= 85% at L=16 -> PASS")
log(f"  K893 (KILL): Option A quality < 50% at L=4 -> KILL (bottleneck)")
```
This is a log-only cosmetic issue (the actual evaluation logic is correct), but it creates confusion if anyone reads the logs.

### Fix #3 [MODERATE]: L=16 replication speculation replaced with explicit unknown

**STATUS: PROPERLY APPLIED.**

Section 6 (lines 148-161) now explicitly states: "We do not have the per-domain breakdown from Finding #363 to verify whether arithmetic was excluded by the parity guard in that run." The attribution to parity guard behavior is stated as a hypothesis, not a fact: "is attributed to parity guard boundary behavior [...] but this cannot be confirmed without Finding #363's domain-level data." This is honest and properly scoped.

### Fix #4 [MINOR]: Stale docstring cleaned

**STATUS: PARTIALLY APPLIED.**

The top-level module docstring (lines 1-42) has been correctly updated: it references the correct theorems, correct sweep values {16, 24, 36}, and correct kill criteria K894/K895/K896.

However, interior code still contains stale references:
- Line 1307-1309: Banner still shows K891/K892/K893 (same as Fix #2 residue)
- Line 1320: Comment says "Depth sweep: L=2, 4, 8, 16" -- should be L=16, 24, 36
- Line 1340: Experiment name is "exp_m2p_layer_depth" -- missing the "_36" suffix

These are cosmetic issues that do not affect the computation or results.

## New Issues Found After Fixes

### NEW ISSUE 1 [MINOR]: results.json is post-processed, not raw code output

The code at line 1340 would output `"experiment": "exp_m2p_layer_depth"`, but results.json says `"experiment": "exp_m2p_layer_depth_36"`. The code at line 1336 references `option_b_quality_ratio_L{lval}` which would be present in raw output, but results.json contains no Option B data. The code at lines 1362-1366 references `finding_359_d256_L2`, `finding_361_d512_L2`, `finding_362_d1024_L2`, but results.json references `finding_363_L2` through `finding_363_L16`. This confirms results.json was manually edited after the run.

Impact: None on correctness of Option A numbers. All Option A metrics in results.json are internally consistent and match the code's computation logic. But the raw output file has been modified without documentation.

### NEW ISSUE 2 [MINOR]: Option B code still trains and wastes compute

The code trains Option B at every L value (lines 1046-1054) even though the experiment title says "Option A Only" and PAPER.md reports no Option B results. This approximately doubles the runtime (the measured 429s includes Option B training that was then discarded from results.json).

### NEW ISSUE 3 [NON-BLOCKING]: Finding status should be "provisional" not "supported"

PAPER.md line 7 says `Status: supported`. Per the experiment framework, Type 3 frontier extensions have finding status capped at "provisional." The distinction: "supported" means "proof mostly verified," while "provisional" means "frontier extension, or empirical observation awaiting proof." Since the Aghajanyan intrinsic dimensionality claim is an empirical claim (not proven for toy scale), and this experiment extends prior findings into unproven territory, "provisional" is the correct cap.

## Self-Test Audit

1. **One-sentence impossibility property:** Correctly states there is none (Type 3). Cites Aghajanyan d_int < 64 as the conditional claim being tested. PASS.
2. **Cited theorems:** Ghadimi-Lan (arXiv:1309.5549), Aghajanyan (arXiv:2012.13255), Ha et al. (arXiv:1609.09106). All real. Conditions correctly stated with toy-scale caveats. PASS.
3. **Predicted numbers:** Two competing models with specific numbers. Log-linear: 86.4%, 83.8%, 81.2%. Intrinsic-dim: 83-90% range. Discrimination criterion at lines 275-280. PASS.
4. **Falsification condition:** Articulated for each theorem. Theorem 3 falsifiable by >5pp deviation. PASS.
5. **Hyperparameter count:** 0 new. Correct. PASS.
6. **Hack check:** Clean extension, no fixes added. PASS.

## Mathematical Soundness

**Theorem 1 (n_train >= T at L=36): SOUND.**
The Ghadimi-Lan bound depends on {L_smooth, f*, sigma^2, b, T}, none involving n_layers. The claim that Xavier initialization bounds spectral norm regardless of output dimension is standard. The sub-epoch training argument (T/n_train = 0.625 < 1) is correct.

**Theorem 2 (necessary condition on effective rank): SOUND.**
Linear algebra is correct: R^64 -> R^147456 has at most 64-dimensional range. The necessary condition properly distinguishes "can represent" from "can learn."

**Theorem 3 (log-linear model): CORRECTLY SELF-UNDERMINED.**
Two-point fit with two parameters = zero degrees of freedom. The authors honestly note this and present it as a pessimistic bound to be tested against, not as a prediction. The experiment's discrimination value (log-linear vs intrinsic-dim) is genuine.

No sign errors, dimensionality mismatches, or off-by-one issues found.

## Prediction vs Measurement

PAPER.md Table 1 contains the prediction-vs-measurement comparison. Cross-checked against results.json:

| L  | Log-linear | Intrinsic-dim | Measured | results.json | Match? |
|----|-----------|---------------|----------|-------------|--------|
| 16 | 86.4%     | ~86.4%        | 78.6%    | 0.7856      | YES    |
| 24 | 83.8%     | 84-90%        | 93.2%    | 0.9316      | YES    |
| 36 | 81.2%     | 83-90%        | 89.1%    | 0.8912      | YES    |

All numbers in PAPER.md Table 1 match results.json. Per-domain numbers in Table 3 and Table 4 also verified against results.json -- all match.

The discrimination criterion from MATH.md: "If q(L=24) > 85% AND q(L=36) > 83%: intrinsic dimensionality model wins." Both conditions hold (93.2% > 85%, 89.1% > 83%). PAPER.md correctly concludes intrinsic-dim model wins.

**Important nuance correctly handled:** The L=36 measured value of 89.1% is the MEDIAN including arithmetic (-1132%). The median of {-1132%, 89.1%, 97.8%} = 89.1% (the middle value). This is sort domain quality, which happens to be the median. The K894 test at 89.1% >= 85% passes on the sort domain's performance. PAPER.md Section 3 Table 3 makes this transparent.

## NotebookLM Findings
Not generated -- all issues identified through direct analysis.

## Novelty Assessment
Straightforward parameter sweep extending Finding #363. The model discrimination (log-linear vs intrinsic dimensionality) is the novel contribution. Prior art correctly cited. No missing references identified.

## Macro-Scale Risks (advisory, not blocking)

1. **Toy-to-real intrinsic dimensionality gap.** Sort/reverse may have artificially low d_int due to high cross-layer correlation in simple pattern tasks. Real language tasks at Qwen3-4B scale may require d_M2P > 64.

2. **Output head parameter scaling.** At d_model=3072, the fc1 output head alone is ~113M parameters. Training dynamics of a single 113M-parameter linear layer may differ qualitatively from the 9.4M toy case.

3. **Single decisive domain.** K894 PASS depends on sort domain (89.1%). With only one domain above the threshold providing the median, the evidence is thin. Macro needs more domains.

## Verdict

**PROCEED**

The 4 original issues have been substantively addressed. The 2 major/moderate fixes (Section 5 rewrite and kill criteria IDs) are properly applied in the areas that affect correctness. The remaining residue (stale log banner, stale comments, post-processed results.json) is cosmetic and does not affect the scientific validity of the result.

**Conditions for proceeding:**

1. Finding status MUST be recorded as `provisional` (not `supported`) per the Type 3 frontier-extension cap.
2. The stale K891/K892/K893 banner at lines 1306-1309 and stale comments at lines 1320, 1340 should be cleaned in a housekeeping pass but are not blocking.
3. The evidence statement for the finding should note that Option B was trained but not reported, and that results.json was post-processed to remove Option B data and correct the experiment name.

**Core finding (provisional):** Option A (single M2P call generating all L layers' adapters) achieves sort=89.1%, reverse=97.8% at L=36, passing K894 (>=85%). The Aghajanyan intrinsic dimensionality model is supported over the log-linear degradation model. The 64-dimensional M2P bottleneck appears sufficient to capture cross-layer adapter structure at 36 layers (2304:1 compression on fc1 head).

# Peer Review: m2p_activation_scaling (Revised)

## Experiment Type
guided-exploration

**Stated framework:** Grassmannian parameter-space orthogonality (Findings #3, #341).
**Stated unknown:** How activation-space interference scales with N composed adapters.

Valid guided-exploration framing. The proven framework is real. The unknown is genuinely open, structurally motivated, and directly addresses Critique #6 and VISION.md Level 2B.

## Hack Detector

- **Fix count:** 1 mechanism (measure activation cosine, fit power law). No hacks.
- **Is MATH.md a proof or a description?** Measurement protocol with theoretical baseline. Acceptable for guided-exploration -- no formal proof is expected.
- **Metric used as evidence:** Per-token activation cosine (max and mean) and power-law exponent alpha. Both are geometrically well-defined and appropriate for the question.
- **Kill criteria source:** K903 and K904 derived from the random-B theoretical analysis and CLT accumulation argument. K905 is a practical quality threshold. All grounded.

## Prior Review Fixes -- Verification

### Fix 1: Per-token cosine metric (BLOCKING fix) -- VERIFIED
The code at lines 500-513 of run_experiment.py correctly iterates over each token position t and computes `|cos(ai_t, aj_t)|` where `ai_t` and `aj_t` are `(d_out,)` vectors. This matches the MATH.md definition exactly. The prior bug (flattening across tokens into a global trajectory cosine) is fixed.

### Fix 2: Power-law fit reliability (BLOCKING fix) -- VERIFIED
R-squared improved from 0.52 to 0.90. The data is now monotonically increasing across all 5 N-values (0.189, 0.204, 0.219, 0.317, 0.339). The step-function artifact is gone. The fit is reliable.

### Fix 3: fc1 module measured alongside wq (non-blocking) -- VERIFIED
Both wq (d_out=256) and fc1 (d_out=1024) are measured separately with their correct inputs (norm1 for wq, norm2 for fc1). Per-module breakdown is reported in both results.json and PAPER.md. The worst case across both is the primary metric.

### Fix 4: K905 FAIL honestly reported (non-blocking) -- VERIFIED
PAPER.md devotes an entire section ("Root Cause of K905 FAIL") to the failure analysis. It explicitly states "K905 is a genuine FAIL," identifies the 4/10 domains where comp_loss > base_loss, and correctly attributes the root cause to equal-weight 1/N dilution -- not a metric artifact. This is honest and well-diagnosed reporting.

## Self-Test Audit

The Self-Test section appears in PAPER.md (lines 216-254) rather than at the end of MATH.md as structurally required. The content is complete:

1. **Impossibility structure:** Adequate. States random B would keep cos at 1/sqrt(d_out) floor. Correctly identifies measured 0.339 as above floor but sub-linear. PASS.

2. **Cited theorems:** Random projection theorem for E[|cos|] = O(1/sqrt(d_out)) with orthogonal inputs. Correctly applied as a baseline, with honest acknowledgment that trained B-matrices are not random. The CLT accumulation argument (alpha=0.5 for i.i.d. increments) is used as an upper-bound reference. PASS with caveat: these establish baselines, not guarantees.

3. **Specific predictions:** MATH.md Table predicts alpha in [0.3, 0.5] and max_cos at N=10 in [0.30, 0.50]. Both specific and falsifiable. PASS.

4. **Falsification condition:** alpha >= 0.5 or max_cos > 0.5 at N=10. Clear and derived from the framework. PASS.

5. **Hyperparameter count:** d_model=256, rank=4, 10 domains. Sensitivity to rank acknowledged as follow-up concern. PASS.

6. **Hack check:** Self-test item 6 addresses the prior metric issue (per-token vs global). Correctly diagnoses the plateau in the prior run as a measurement artifact. PASS.

**Structural note:** The self-test should be in MATH.md, not PAPER.md. Non-blocking.

## Mathematical Soundness

No formal proof is claimed (correct for guided-exploration). The framework consists of:

1. **Metric definition:** activation_cos(i,j,t) = |cos(B_i A_i x_t, B_j A_j x_t)|. Geometrically well-defined. Implementation matches definition. SOUND.

2. **Random-B baseline:** E[|cos|] = O(1/sqrt(d_out)) for random B with orthogonal inputs. Applied at d_out=256 gives floor ~0.063. Measured wq mean cosine: 0.056-0.064 across N values -- right at the predicted floor. For fc1 (d_out=1024): floor ~0.031, measured 0.041-0.061. The fc1 mean is ~2x floor at N=10, indicating weak learned correlation in wider modules. This baseline comparison is informative.

3. **Power-law fit:** Log-log linear regression with R-squared=0.90. The fit is applied to the composite max_cos (worst of wq and fc1), which exhibits a "knee" at N=8 where fc1 starts dominating. This module switch inflates the apparent growth rate somewhat -- the wq-only data shows gentler growth (0.158 to 0.248, roughly alpha~0.28 for wq alone). The reported alpha=0.379 is thus influenced by the module switch and should be understood as an upper bound on within-module scaling. This is not a flaw -- measuring the worst case across modules is the conservative choice -- but the composite nature of the measure should be noted.

4. **CLT comparison:** The claim "alpha=0.379 < 0.5 falsifies i.i.d. growth" is slightly overstated. The CLT argument predicts expected max ~ sqrt(N) for i.i.d. additions, but (a) max_cos is not a sum, it is a maximum over pairs, and (b) the Bonferroni correction for C(N,2) pairs growing as N^2 would push the expected maximum even higher than sqrt(N). So alpha < 0.5 is a weaker statement than claimed -- it means the B-matrices are NOT adding interference independently, which is plausible given they are trained on the same base model. The directional conclusion (sub-linear) holds; the mechanistic explanation could be tighter.

## Prediction vs Measurement

PAPER.md contains the required table (lines 35-42). Verification against results.json:

| N | Predicted | Measured | results.json | Match |
|---|-----------|----------|--------------|-------|
| 2 | 0.10-0.20 | 0.189 | 0.189213 | Within range |
| 5 | 0.20-0.35 | 0.219 | 0.219233 | Within range |
| 10 | 0.30-0.50 | 0.339 | 0.338491 | Within range |
| alpha | 0.3-0.5 | 0.379 | 0.3789 | Within range |

All predictions confirmed. Numbers in PAPER.md match results.json to rounding precision. Quality_frac calculations verified manually (e.g., reverse at N=10: (2.2211 - 2.795) / (2.2211 - 2.1221) = -5.797, reported as -5.80).

## Novelty Assessment

This experiment directly answers Critique #6 from a prior adversarial review and fills an open question in VISION.md (Level 2B: "Measuring how [activation interference] scales with N"). The result -- sub-linear scaling with alpha=0.379 -- is a genuine empirical finding within this project's framework.

No prior art was found that answers this specific question for Grassmannian A-slot LoRA composition. The general question of multi-adapter interference has been studied in LoRA merging literature, but the Grassmannian setup and per-token worst-case analysis are novel.

## Macro-Scale Risks (advisory)

1. **Module width dominance:** fc1 (4x wider) becomes the binding module at N >= 8. At macro scale, feed-forward dimensions are typically 4-8x model dimension, so this pattern will persist and potentially worsen.

2. **Rank sensitivity:** At rank=4, each A-projection captures 1.6% of d=256. Higher ranks at scale would increase B-matrix expressiveness and potentially increase learned correlation. The sub-linear exponent likely holds (the argument is that B-matrices trained on different domains have limited structural correlation), but the constant c in max_cos = c * N^alpha may increase.

3. **Equal-weight composition failure:** The 1/N dilution that causes K905 failure at N=10 is architectural, not scale-dependent. Any deployment at N > 5 will need routing or minimum-weight thresholds.

4. **Cross-layer accumulation:** This experiment measures layer 0 only. In deeper networks, residual stream accumulation could amplify or attenuate per-layer interference. This is unmeasured.

## Minor Issues (non-blocking)

1. Self-test section is in PAPER.md rather than MATH.md.

2. The n_pairs column in the scaling table counts (pair, token) observations from a single batch. With T=8 tokens (short arithmetic sequences), the per-batch sample is small. However, the max is taken over 30 batches, giving 240 token positions examined per pair -- adequate for a worst-case measurement.

3. The fc1 max_cos is identical at N=2 and N=3 (0.189213 to 6 decimal places). This is expected (the same adapter pair 0-1 dominates), but the reader might wonder whether different adapters were actually measured. The code confirms the first N adapters are used at each N level, so adapter pair (0,1) appears in all measurements.

## Verdict

**PROCEED**

All four fixes from the prior REVISE verdict have been applied and verified. The blocking issues (cosine metric mismatch, unreliable power-law fit) are resolved. The experiment is well-designed for its type (guided-exploration), the measurements match predictions, the failures are honestly reported, and the scope limitations are clearly stated. The finding status of "supported" is appropriate given 2/3 kill criteria pass and the activation-space scaling question is answered within the measured scope.

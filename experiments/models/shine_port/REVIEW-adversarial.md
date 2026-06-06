# Peer Review: shine_port (Revision 4)

## Experiment Type
**Guided Exploration (Type 2)** -- correctly declared. The proven framework is SHINE's M2P architecture (arXiv:2602.06358). The unknown is whether the MLX port produces outputs statistically distinguishable from random projections. No formal theorem is claimed. This is appropriate for a porting exercise with a structured statistical test.

## Hack Detector
- Fix count: 0. Clean architecture port, no stacked mechanisms.
- Is MATH.md a proof or a description? **Neither -- correctly declared as "no formal theorem claimed."** Appropriate for Type 2 guided exploration using RMT as the null-model framework.
- Metric used as evidence: two-sample Welch t-test (p < 0.05) with effect size guard (|diff| > 0.05). Sound for the stated question.
- Kill criteria source: K827 uses standard statistical threshold + practical significance guard. Acceptable for Type 2.

## Self-Test Audit

**H.1:** Single property ("whether M2P outputs are distinguishable from random noise"). PASS.
**H.2:** JL 1984, Vershynin 2018 Ch. 3: real references, correctly applied. Independence violation now explicitly acknowledged in Section D. PASS.
**H.3:** Predictions specific and falsifiable. P1 correctly marked as NOT VALID for M2P. P2, P3, P4 remain. PASS.
**H.4:** Falsification condition targets the scientific question (K827 FAIL = no demonstrated structure). PASS.
**H.5:** 0 hyperparameters. Configuration choices are SHINE architecture parameters. PASS.
**H.6:** No hack accumulation. Corrections only. PASS.

## Revision 4 Fix Verification

Three text-only fixes were requested in the revision 3 review. Verification:

### Fix 1: Phase 6a prediction table -- PARTIAL MATCH
**APPLIED CORRECTLY.** PAPER.md line 67 now reads "PARTIAL MATCH" with full explanation: std matches (0.0230 vs 0.0221), mean REFUTED by ~19.5 sigma (SE=0.0042, measured=0.0818 vs predicted=0.0000). Cross-references the new Positional Embedding Bias section.

### Fix 2: Positional Embedding Bias section in PAPER.md
**APPLIED CORRECTLY.** PAPER.md lines 75-86 contain a dedicated section that:
- Identifies the RMT independence violation (shared P_layer + P_token via SHINE Section 3.4 Eq. 5)
- Explains the mechanism (shared additive offset pushes inner product positive)
- Quantifies the magnitude (P_layer norm=2.78, P_token norm=4.03)
- Explicitly states the K827 implication: "The t-test PASS (p=0.0023) partly reflects this systematic positional bias"
- Notes the mixed source of structure (Xavier init + training, not training alone)
- Recommends disentangling positional from semantic structure in future work

### Fix 3: MATH.md Section D -- P1 validity note
**APPLIED CORRECTLY.** MATH.md lines 140-155:
- P1 table entry annotated: "NOT VALID for M2P; see note below"
- Detailed validity note explains the shared positional embedding violation of RMT independence
- Shows SHINE Eq. 5 creates E[cos] > 0 (not = 0)
- Provides experimental confirmation (19.5 sigma deviation)
- Explains why P2 (std) remains valid (measures input-conditional variance, unaffected by shared mean shift)
- Declares P1 "should be struck from future M2P experiments"

All three fixes are thorough, technically correct, and address the specific concerns raised.

## Mathematical Soundness

The RMT framework (E[cos]=0, std=1/sqrt(n) for independent Gaussian vectors) is correctly stated and correctly applied as a null model. The critical recognition that M2P outputs violate the independence assumption (due to shared positional embeddings) is now properly documented in both MATH.md and PAPER.md.

The Welch t-test is correctly specified (unequal variances, two-tailed, df=58 via Welch-Satterthwaite). The power calculation (Section C.2) correctly identifies the detectable shift at n=30.

No remaining derivation errors, sign errors, or dimensionality mismatches.

## Prediction vs Measurement

PAPER.md contains the prediction table. All entries are now correctly assessed:

| Prediction | Measurement | PAPER.md Claim | Assessment |
|-----------|-------------|----------------|------------|
| RMT Phase 6a: E[cos]=0, std=0.022 | mean=0.0818, std=0.023 | PARTIAL MATCH | Correct. Std matches, mean refuted. Cause identified. |
| RMT Phase 6b: E[cos]=0 | mean=0.104, std=0.078 | DRIFT | Fair. Training shifts distribution further. |
| K827: M2P != Random | t=3.33, p=0.0023, |diff|=0.0815 | PASS | Correct. Valid t-test, mixed source acknowledged. |
| Architecture compiles | 197K params, 4.1ms | PASS | Correct. Engineering validation. |
| Convergence | loss ratio 0.126 | PASS | Correct. |

No material errors remain in the prediction-measurement table.

## NotebookLM Findings

Skipped. The revision 4 changes are text-only corrections to previously-reviewed material. No new claims or mechanisms require deep literature review.

## Novelty Assessment

Zero novelty claimed. Correctly positioned as an infrastructure porting exercise. The M2P architecture is from SHINE. The contribution is: "SHINE M2P compiles and runs on MLX, produces outputs distinguishable from random noise." Appropriate for a PROVISIONAL finding.

## Macro-Scale Risks (advisory)

1. **Positional embedding bias scales with L and H.** At L=28, H=1024, the shared positional component will be larger. The RMT null model becomes even less appropriate. Future macro experiments must use untrained-M2P as baseline rather than random matrices.
2. **Variance blowup after training.** Post-training M2P cosine std=0.130 (5.5x random). At scale, this may cause instability in generated adapter weights. Monitor.
3. **No real-task validation.** M2P outputs differ from random, but downstream impact (PPL, MMLU) is unaddressed. Required before any production claims.

## Minor Observations (non-blocking)

- MATH.md header still says "Revision 3" despite this being revision 4 of the review cycle. Cosmetic; does not affect findings.
- The revision 3 review noted that a cleaner design would compare trained-M2P vs untrained-M2P to isolate training effect from positional effect. This is mentioned as future work in PAPER.md (line 86: "disentangle positional structure from semantic structure") but not as a formal next step. Acceptable for a provisional finding.

## Verdict

**PROCEED**

All three requested fixes from the revision 3 review have been applied correctly and thoroughly:

1. Phase 6a prediction table: changed from "MATCH" to "PARTIAL MATCH" with 19.5-sigma refutation explanation. DONE.
2. Positional Embedding Bias section: added to PAPER.md with mechanism, magnitude, and K827 implications. DONE.
3. MATH.md P1 validity note: P1 marked as NOT VALID for M2P, independence violation explained, P1 stricken for future experiments. DONE.

No new material errors found. The experiment is a clean Type 2 guided exploration with honest framing, a sound statistical test, correct reporting, and appropriate PROVISIONAL status. The finding -- "SHINE M2P compiles on MLX, produces outputs distinguishable from random noise (partly from positional embeddings, partly from training), ready for downstream evaluation" -- is supported by the evidence and does not overreach.

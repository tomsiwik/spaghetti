# Peer Review: exp_m2p_qwen3_quality

## Experiment Type
Guided exploration (Type 2) -- testing whether a proven recipe (d_M2P=64, L=2, n=2000, GL stopping) transfers from d_model=512 to d_model=1024.

## Hack Detector
- Fix count: 0 new mechanisms. Single change: D_MODEL 512 -> 1024. Clean.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 (n_train >= T) is a genuine proof with QED. Theorem 2 is a conditional argument dressed as a proof -- it restates Aghajanyan et al.'s empirical claim as a premise and derives a trivial consequence. See Mathematical Soundness below.
- Metric used as evidence: quality_ratio = (base - m2p) / (base - sft). **Metric inflation concern flagged below** -- but absolute differences confirm the direction.
- Kill criteria source: K885 (85%) from Aghajanyan framework, K886 (0.7 nats) inherited from prior, K887 (50%) as cliff detector. Reasonably derived, not arbitrary.

## Self-Test Audit

1. **One-sentence impossibility property:** "Intrinsic dimensionality of LoRA B-matrix updates is d_model-independent." This is a single property. However, it is an *empirical finding* from Aghajanyan et al., not a mathematical impossibility. MATH.md correctly caveat-labels this in Assumption 5. PASS with caveat.

2. **Cited theorems -- are they real? Do conditions apply?**
   - Ghadimi & Lan (arXiv:1309.5549, Thm 2.1): Real theorem, correctly applied. The i.i.d. condition is structural and holds by construction when n_train >= T. PASS.
   - Aghajanyan et al. (arXiv:2012.13255, "Theorem 2"): **PROBLEM.** The cited paper does not contain a "Theorem 2" in the formal mathematical sense. Aghajanyan et al. present empirical measurements of intrinsic dimensionality across models (RoBERTa-Base, RoBERTa-Large, etc.) and observe sub-linear scaling. Their key result is Definition 1 (intrinsic dimensionality) and empirical Table 1, not a formal theorem with proof. MATH.md cites "Theorem 2 of Aghajanyan et al." -- this is either a misattribution or reference to an informal proposition. **The argument still works** because the experimental progression (97.6% -> 101.0% -> 99.6%) empirically confirms the d_int claim, but calling it a "theorem" overstates the formal backing. REVISE: cite the specific empirical finding, not a nonexistent theorem number.
   - Ha et al. (arXiv:1609.09106): Real paper, correctly described. The bottleneck-as-regularizer interpretation is a known mechanism. PASS.
   - Prechelt (1998): GL criterion correctly stated. PASS.
   - Hardt et al. (2016): Real theorem, correctly applied qualitatively. As noted in prior MATH.md (m2p_data_scale), the quantitative bound is vacuously loose for this setting. PASS with caveat.

3. **Predicted numbers:**
   - T/n_train = 0.625 (structural, trivially verified). PASS.
   - train-val gap < 0.7 nats: measured 0.2355. PASS.
   - quality_ratio >= 85%: measured 99.6%. PASS.
   - K887 kill < 50%: not triggered (99.6%). PASS.
   All predictions are specific and falsifiable. PASS.

4. **Falsification condition:** "K887 trigger (< 50%) would be definitive falsification of Aghajanyan d_int independence." This targets the core claim. Also acknowledges that 50-85% range is degradation but not falsification. Reasonable. PASS.

5. **Hyperparameter count:** 0 new hyperparameters. Single treatment variable: D_MODEL = 1024. PASS.

6. **Hack check:** No fixes added to existing stack. Pure scaling test. PASS.

## Mathematical Soundness

### Theorem 1 (n_train >= T is d_model-independent)
**Verdict: CORRECT.** The proof is straightforward and valid. The Ghadimi-Lan i.i.d. condition depends only on n_train vs T, not on architectural dimensions. The derivation is clean: n=2000, n_train=1600, T=1000, T/n_train=0.625 < 1. No cycling occurs.

However, Theorem 1 only guarantees that *generalization gap is bounded*. It says nothing about the *level* of the loss. A model could have bounded generalization gap but terrible absolute quality. Theorem 1 is correctly stated but its scope is limited to the overfitting question -- it does not address the capacity question (K885/K887). This is acknowledged in MATH.md.

### Theorem 2 (Aghajanyan intrinsic dimensionality)
**Verdict: CONDITIONAL, NOT A PROOF.** The argument structure is:

1. If d_int is d_model-independent (Aghajanyan claim)
2. And d_M2P=64 >= d_int (empirically supported at d=256, d=512)
3. Then d_M2P=64 is sufficient at d=1024.

This is a valid conditional chain, but it is a *hypothesis test*, not a proof. The Aghajanyan premise is an empirical observation from full-scale LLMs (BERT, RoBERTa), not a mathematical theorem. Applying it to toy transformers on character-level synthetic data is an extrapolation. MATH.md correctly labels this as a Type 2 guided exploration and notes the caveat (line 229-234). The "QED (under Aghajanyan et al. assumptions)" qualifier is honest.

**Critical question: is the Aghajanyan framework the right lens?** Aghajanyan et al. studied intrinsic dimensionality of *fine-tuning updates* on natural language tasks. The M2P is not fine-tuning -- it is a *hypernetwork* generating B-matrices for toy tasks. The analogy is plausible (both involve low-rank parameter updates), but not formally established. This is acknowledged in Assumption 5.

### Compression Ratio Analysis
The compression ratio argument (128:1 -> 256:1 -> 512:1 with no quality degradation) is presented as an "engineering estimate, NOT a theorem." This is appropriate and honest.

### Missing: Smoothness Assumption
Theorem 1 requires L-smoothness of the loss function. For transformer models with RMSNorm + GELU, L-smoothness holds locally but the smoothness constant L can grow with d_model (larger weight matrices -> larger Lipschitz constants). The proof treats L as irrelevant to the i.i.d. condition (correct), but the *rate* of convergence O(L/T) depends on L. At d=1024, L may be larger, meaning T=1000 achieves less convergence. This does not invalidate Theorem 1 but could explain quality differences across scales.

## Prediction vs Measurement

PAPER.md Section 1 contains the prediction-vs-measurement table. All predictions match.

| Prediction | Predicted | Measured | Status |
|---|---|---|---|
| T/n_train | 0.625 | 0.625 | Structural match |
| quality_ratio >= 85% (K885) | >= 85% | 99.6% | PASS |
| train-val gap < 0.7 nats (K886) | < 0.7 | 0.2355 | PASS |
| No compression cliff (K887) | >= 50% | 99.6% | Not triggered |
| Degradation from d=512 | <= 16pp | 1.4pp | PASS |

**Quality_ratio metric inflation -- the central concern:**

The quality_ratio = (base - m2p) / (base - sft) metric becomes increasingly forgiving as d_model grows because the base model degrades disproportionately:

| d_model | base (sort) | sft (sort) | gap (denominator) |
|---|---|---|---|
| 256 | 13.37 | 2.61 | 10.76 |
| 512 | 15.45 | 2.42 | 13.03 |
| 1024 | 17.26 | 2.21 | 15.05 |

The denominator grows ~40% from d=256 to d=1024. A fixed M2P-SFT absolute error of 0.2 nats would register as:
- d=256: quality_ratio = 1 - 0.2/10.76 = 98.1%
- d=1024: quality_ratio = 1 - 0.2/15.05 = 98.7%

**However**, computing absolute M2P-SFT loss differences shows the concern is not critical:
- d=256 (sort): m2p - sft = +0.063 nats
- d=512 (sort): m2p - sft = -0.055 nats (M2P beats SFT)
- d=1024 (sort): m2p - sft = +0.019 nats

The absolute differences are small, non-monotonic, and tightly clustered around zero across all three scales. The quality_ratio metric inflates slightly at larger d_model, but the underlying absolute quality is genuinely stable.

**Advisory (not blocking):** Future experiments should report both quality_ratio and absolute loss differences (m2p_loss - sft_loss) to make the scaling trend unambiguous. The claim "99.6% quality" should be accompanied by "m2p_loss is 0.019 nats above sft_loss in the sort domain."

## NotebookLM Findings

NotebookLM was not used for this review (tool available but not invoked). The analysis was conducted through direct file inspection.

## Novelty Assessment

**Prior art:** The Aghajanyan et al. intrinsic dimensionality claim is well-established in the NLP literature. Ha et al. HyperNetworks is foundational. The M2P architecture (hypernetwork generating LoRA B-matrices) is the novel contribution of this project, not this specific experiment.

**Delta over existing work:** This experiment contributes a third data point (d=1024) to the scaling progression (d=256, d=512, d=1024), extending the intrinsic dimensionality observation to higher compression ratios (512:1) in a toy setting. This is incremental but valuable within the M2P research program.

**Aghajanyan citation accuracy:** MATH.md refers to "Theorem 2 of Aghajanyan et al." The 2021 paper does not contain a numbered "Theorem 2" in the standard mathematical sense. The paper's main formal contribution is Definition 1 (d_int) and the empirical measurements in Table 1. This should be corrected to reference the specific empirical finding rather than a nonexistent theorem.

## Critical Issues Found

### Issue 1 (NON-BLOCKING): Only 2 valid domains

Arithmetic is excluded by parity guard (base-SFT gap = 0.0168 nats < 0.05 threshold). The entire quality claim rests on sort and reverse -- two domains. The median of two values is just their average. No variance estimate is possible. PAPER.md and MATH.md both acknowledge this limitation, and the macro_quality predecessor had the same constraint.

**Why this is not blocking:** (a) The same N_DOMAINS=3 with arithmetic exclusion held at d=256 and d=512, making the comparison valid across scales. (b) The two valid domains agree closely (sort=99.9%, reverse=99.3%). (c) The absolute M2P-SFT differences are consistently small across all three scales.

**Advisory:** Any "path to Qwen3-4B" claim should note that this has been validated on only 2 synthetic toy domains (sort, reverse). Real NLP tasks have fundamentally different structure. The Qwen3-4B claim is a hypothesis to test, not a conclusion from this experiment.

### Issue 2 (NON-BLOCKING): "Theorem 2" of Aghajanyan et al. is not a theorem

See citation accuracy note above. The intrinsic dimensionality claim is an empirical observation, not a formal theorem. MATH.md Theorem 2 is therefore a conditional argument, not a mathematical proof. This is honestly labeled ("QED under Aghajanyan et al. assumptions") but the labeling as "Theorem 2" in MATH.md creates a false impression of formal backing.

**Fix:** Relabel as "Proposition 2 (conditional on Aghajanyan et al. empirical finding)" or similar.

### Issue 3 (NON-BLOCKING): Arithmetic domain failure at d=1024 is unremarked

The arithmetic domain has base_loss=1.5743 and sft_loss=1.5575 at d=1024, a gap of only 0.0168 nats. This means the base model at d=1024 already nearly solves arithmetic -- SFT adds almost nothing. At d=256, the arithmetic gap was 1.5656 - 1.5525 = 0.0131 nats (also tiny). This is consistent: arithmetic at this scale is essentially solved by the base model. The parity guard correctly excludes it. No issue here, but the reduced domain count should temper claims.

### Issue 4 (ADVISORY): Train-val gap uses single-step training loss

Line 904: `train_val_gap = abs(final_train_loss - final_val_loss)`. The `final_train_loss` is the loss on a single random training sample at the last step, not an epoch average. This adds noise to the gap measurement. At d=1024, the measured gaps (0.159, 0.236 nats) are well below the 0.7 threshold, so this does not affect the kill criterion outcome.

### Issue 5 (ADVISORY): The "path to Qwen3-4B is unobstructed" claim

PAPER.md Section 3 states: "The path to Qwen3-4B deployment is unobstructed by the d_model scaling dimension."

This overreaches. What has been shown:
1. d_M2P=64 handles 512:1 compression on 2 synthetic toy domains with a 2-layer ToyGPT.
2. Quality remains near SFT through d=256 -> 512 -> 1024.

What has NOT been shown:
1. Behavior on real NLP data (vocabulary >> 128, sequences >> 48 tokens).
2. Behavior with more than 2 layers.
3. Behavior at d=2048 or d=3584 (actual Qwen3-4B dimensions).
4. Behavior with more than 3 domains.
5. Whether the intrinsic dimensionality of real LLM fine-tuning tasks matches toy arithmetic/sort/reverse.

The claim should be: "The d_model scaling dimension does not appear to be a bottleneck at toy scale through d=1024. The next verification step is d=2048 or direct Qwen3-4B testing."

## Macro-Scale Risks (advisory)

1. **Vocabulary and sequence length:** ToyGPT uses vocab=128, block_size=48. Real LLMs have vocab=32K-128K and seq_len=2K-128K. The M2P's input projection (d_model -> d_M2P=64) compresses across the sequence dimension via mean pooling. Whether the 64-dim bottleneck captures sufficient information from much longer, richer sequences is unknown.

2. **Number of layers:** ToyGPT has 2 layers. Qwen3-4B has 36+ layers. The M2P generates B-matrices for all layers. At 36 layers, the output heads must generate 36x more parameters from the same 64-dim bottleneck. This is the real compression cliff risk -- not d_model scaling, but n_layers scaling.

3. **Task diversity:** Sort and reverse are simple algorithmic tasks. Real NLP domains (medical, legal, code) may have higher intrinsic dimensionality. The Aghajanyan et al. results on BERT/RoBERTa are more relevant here than the toy results, but the connection between the two is not formally established.

## Verdict

**PROCEED**

Finding #362 is correct within its stated scope. The experimental design is clean (single variable change), the kill criteria are reasonable, all three pass, and the absolute M2P-SFT loss differences confirm the quality_ratio metric is not misleading. The mathematical framework is honest about its limitations (Type 2 guided exploration, not Type 1 verification).

**Required for the finding record (not blocking for PROCEED):**

1. Correct the citation: "Theorem 2 of Aghajanyan et al." should reference the specific empirical finding (Table 1 / Section 4 of their paper), not a nonexistent theorem number.

2. Temper the Qwen3-4B claim: change "path to Qwen3-4B deployment is unobstructed" to "d_model scaling does not appear to be a bottleneck through d=1024; direct Qwen3-4B testing is the next required verification."

3. Report absolute M2P-SFT loss differences alongside quality_ratio in the finding summary, to preempt the metric inflation concern.

# Peer Review: Entropy-Gated Expert Selection (v2)

## Context

This is a re-review of v2, which addressed two blocking issues from the v1 review:
1. Dip test was KS-from-uniform, not Hartigan. K1 reframed as "sufficient spread" using CV and Otsu eta.
2. K2 compared incompatible metrics. Now uses token-weighted compose PPL.

Both fixes have been verified in code and results.

## NotebookLM Findings

Skipped -- not available in this environment. Review proceeds with direct analysis.

## Mathematical Soundness

### Fix 1 Verification: K1 Reframing (ADEQUATE)

The v1 review found that the "dip test" was a KS-distance-from-uniform test, not Hartigan's dip statistic. The v2 response is correct: drop the bimodality claim entirely and reframe K1 as "sufficient spread for thresholding."

The two conditions are sound:
- **CV = 0.87 > 0.5:** Standard definition (std/mean), correctly implemented at line 231-238. The threshold of 0.5 is reasonable -- below this, the distribution is too tight relative to its center for any threshold to create meaningfully different groups.
- **Otsu eta = 0.68 > 0.15:** Between-class variance ratio, correctly implemented at lines 241-260. The formula `sigma_B^2 = w_0 * w_1 * (mu_0 - mu_1)^2` and `eta = sigma_B^2 / sigma_T^2` match the standard Otsu formulation. An eta of 0.68 is genuinely strong -- the threshold explains 68% of total variance. The 0.15 minimum is conservative.

The MATH.md explicitly acknowledges that bimodality is not claimed and explains why Otsu does not require it. This is honest and correct.

**Minor note:** The Otsu threshold search (lines 267-284) uses 200 points between the 5th and 95th percentile. This grid search is adequate for 13K tokens but is not the textbook Otsu algorithm (which exhaustively tests all unique values). The difference is negligible here.

### Fix 2 Verification: Token-Weighted Compose PPL (ADEQUATE)

The v1 review found that gated PPL (token-weighted) was compared against arithmetic-mean compose PPL (domain-averaged), inflating the baseline. The v2 fix computes token-weighted compose PPL correctly:

Lines 607-612: All compose losses are flattened across domains, then `exp(mean(all_compose_losses))` gives the token-weighted compose PPL of 6.9965. The gated PPL at Otsu threshold is 7.0756, giving 1.13% degradation. This is a fair apples-to-apples comparison.

The paper correctly notes that the v1 "surprise finding" (gated PPL better than compose) was an artifact of the metric mismatch. The v2 narrative -- "gating trades 1.13% quality for 63% fewer composition operations" -- is the honest framing.

### Fix 3 Verification: Consistent Timing Threshold (ADEQUATE)

The v1 review noted Phase 3 used a hardcoded threshold of 3.0 instead of Otsu's 2.10. In v2, line 836 uses `otsu_tau` which is passed from Phase 1 results at lines 904-905. Confirmed fixed.

### Entropy Computation (UNCHANGED, ACCEPTABLE)

Line 208: Variable named `log_probs` but contains `softmax(logits)` (probabilities, not log-probs). Cosmetic only. The entropy calculation itself is correct: `H = -sum(p * log(p))` with numerical clamping.

### Cost Model (HONEST)

The MATH.md correctly derives that two-pass gating is slower: `E[cost] = (2 - f_skip) * C_fwd` for f_skip=0.63 gives 1.37x. It then presents the correct architecture (pre-filter for routing heads) where the value comes from skipping routing computation, not from skipping forward passes. The paper does not claim latency benefit for the two-pass approach and S3 is honestly reported as FAIL.

### Otsu Between-Class Variance Ratio: One Subtlety

The Otsu eta of 0.68 is computed on the *global* entropy distribution across all five domains. Since domains have very different mean entropies (python 0.79, legal 2.61), some of the "two-class structure" is really "domain structure." If you ran Otsu within a single domain, eta would likely be lower. This does not invalidate the result -- in practice, the gate operates on the global distribution -- but it means the high eta partially reflects domain separability rather than within-domain confident/uncertain token separation.

## Novelty Assessment

### Prior Art

The paper cites CALM, DeeBERT, MoBiLE, pQuant, and EdgeNav-QE. These are the right references. The mechanism (entropy-based confidence gating to skip expensive computation) is well-established in early-exit literature. The specific application to LoRA adapter composition is a reasonable new application, not a novel mechanism.

### Differentiation from Killed exp_entropy_adaptive_router

The paper correctly distinguishes this (binary skip-all gate) from the killed experiment (variable-k within MoE layers). This is a legitimate distinction.

### Delta Over Existing Work

Small but valid. The contribution is not the mechanism but the empirical finding that 63% of tokens can skip composition with only 1.13% PPL degradation on a ternary base with LoRA adapters. This is a useful data point for the VISION.md architecture.

## Experimental Design

### 1. Per-Token Oracle Evaluation vs Sequence-Level Timing (CONCERN, NON-BLOCKING)

Phase 2 evaluates PPL using per-token oracle selection: for each token independently, use base loss if entropy < tau, else compose loss (lines 574-583). Phase 3 timing uses sequence-level gating: if ANY token in the sequence exceeds threshold, re-run the whole sequence (line 836). These test different things.

The PPL numbers from Phase 2 represent the best-case quality of per-token gating. The timing numbers from Phase 3 represent a practical (but suboptimal) sequence-level implementation. This mismatch is acknowledged implicitly but should be stated explicitly.

### 2. Same Data for Threshold and Evaluation (ACKNOWLEDGED)

Otsu's threshold is computed on the same validation set used for PPL evaluation. The paper acknowledges this in Limitations item 2. Since the threshold sweep shows smooth, monotonic degradation (each 10% more skipping costs roughly 0.1-0.2% PPL), the threshold is not overfitting to a particular data point. The risk is low.

### 3. Uniform 1/N Composition as Baseline (ACKNOWLEDGED)

The paper acknowledges (Limitations item 1) that routed top-2 achieves PPL 6.42 vs uniform 7.0. The interesting experiment -- entropy gating as pre-filter for routing -- is correctly identified as next work. This does not block the current experiment, which establishes the basic mechanism.

### 4. Domain Imbalance

Token counts range from 1500 (medical) to 4540 (creative), a 3x ratio. Token-weighted PPL is thus dominated by creative (33%) and legal (25%). The per-domain breakdown (Table in PAPER.md Phase 2) shows consistent small degradation across all domains (0.53% to 1.86%), which mitigates the concern that aggregate results hide per-domain problems.

### 5. Potential Confound: Domain Identity vs Token Confidence

The 89% skip rate for python vs 37% for legal raises the question: is entropy gating mainly detecting "easy domain" vs "hard domain" rather than finding per-token confidence variation? The per-domain f_skip values (37-89%) suggest significant domain-level signal. However, within each domain there is still variation (e.g., python has 11% of tokens above threshold), and the per-domain degradation is small everywhere, so the mechanism works even if part of its power comes from domain-level differences.

This becomes more important at macro scale with similar domains.

## Hypothesis Graph Consistency

No HYPOTHESES.yml entry found for this experiment. It is correctly placed in VISION.md Track B ("Smart Routing Without Heavy Infrastructure"). The kill criteria (K1: spread, K2: PPL, K3: skip fraction) are well-defined and the code implements exactly what MATH.md specifies.

## Macro-Scale Risks (advisory)

1. **Entropy narrowing with better models.** A more capable base model may be confident on more tokens (pushing the distribution toward low entropy) or uniformly uncertain (pushing toward high entropy). Either extreme reduces the separation that makes gating useful.

2. **Calibration.** Raw softmax entropy is known to be poorly calibrated in large models (Guo et al., 2017). Temperature scaling would be needed for reliable thresholds.

3. **Similar domains.** Five trivially-separable domains inflate the domain-level signal in the entropy distribution. With 25+ overlapping domains, per-token entropy variation within domains becomes the dominant signal, and the Otsu split may be less clean.

4. **Integration with routing heads.** The paper correctly identifies that the value is as a pre-filter for routing, not as a standalone two-pass system. The macro experiment must test: does entropy gating + top-2 routing beat top-2 routing alone (in compute, not quality)?

## Verdict

**PROCEED**

The v2 revision adequately addresses both blocking issues from the v1 review:

1. K1 is correctly reframed from "bimodality" to "sufficient spread for thresholding." CV=0.87 and Otsu eta=0.68 are strong evidence that the entropy distribution supports meaningful binary gating. The broken dip test is removed, the failed bimodality coefficient is acknowledged, and no overclaims are made.

2. K2 now uses token-weighted compose PPL (6.9965) as the baseline, producing a fair 1.13% degradation figure. The narrative is correctly reframed from "composition hurts" to "gating trades small quality for large efficiency."

The core finding is solid: 63% of tokens can skip expert composition with only 1.13% PPL degradation. The mechanism works consistently across all five domains (0.53-1.86% degradation each). The two-pass implementation is correctly identified as non-viable, with the correct architecture (pre-filter for routing heads) clearly described.

No new blocking issues were found. The remaining concerns (domain-level confound in entropy, same data for threshold selection, sequence-level vs per-token timing mismatch) are non-blocking at micro scale and are either acknowledged in Limitations or are appropriate to address in the macro integration experiment.

### Recommended (non-blocking) improvements for FINDINGS.md entry:

1. Note that Otsu eta=0.68 on the global distribution partially reflects domain separability, not just per-token confidence variation.
2. State explicitly that Phase 2 PPL is per-token oracle selection while Phase 3 timing is sequence-level -- these measure different things.
3. The next experiment should test entropy gating as a pre-filter for the tiny_routing_heads, not as a standalone system.

# Peer Review: Multi-layer Removal Cascade

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that manual review is sufficient. The mathematical claims are elementary (Lipschitz propagation bounds, random vector composition) and the code is short enough to verify directly.

## Mathematical Soundness

### What holds

1. **Lipschitz bound derivation (Section 3.1).** The worst-case bound `(1 + epsilon)^L - 1` is correctly derived from the product of per-layer Lipschitz constants. The Taylor expansion and numerical estimates (4.8% linear term, 0.11% quadratic at L=24, epsilon=0.002) are correct.

2. **Amplification ratio definition.** The ratio `mean_output_dev / sum_per_layer_weight_error` is a clean, interpretable metric. It correctly distinguishes sub-additive (ratio < 1) from multiplicative (ratio > 1) error propagation.

3. **GS merge and removal logic.** The code correctly implements Gram-Schmidt orthogonalization on flattened weight deltas, naive subtraction of the orthogonalized component, and ground-truth recomputation from N-1 experts. Verified against results.json: L=1 amplification ratios cluster around 1.0 (1.026, 0.957, 0.986), which is the expected degenerate case where output deviation equals weight error.

4. **Sub-additivity claim.** The raw data supports this. Amplification ratio at L=24 (seed 42, depth_ortho) is 0.239, well below 1.0. The trend is monotonically decreasing with depth across all three seeds.

### What requires scrutiny

5. **The three mechanisms (Section 3.2) are hand-waved, not proven.** The paper offers three post-hoc explanations for sub-additivity (activation masking, direction randomization, spectral contraction) but does not isolate which mechanism dominates. Test 3 (activation comparison) partially addresses this: linear activation still shows sub-additivity (amp_ratio=0.27), which means activation masking is NOT the primary mechanism. This contradicts the ordering in the paper, where activation masking is listed as "Mechanism 1." The paper should acknowledge that direction randomization (random walk argument, sqrt(L) scaling) is likely the dominant effect, since it operates even without nonlinear activations.

6. **The sqrt(L) random walk argument is not validated.** If direction randomization is dominant, output deviation should scale as sqrt(L), not sub-linearly. The data: L=1: 1.06%, L=4: 2.11%, L=8: 3.21%, L=24: 5.31%. The ratio 5.31/1.06 = 5.0, while sqrt(24)/sqrt(1) = 4.9. This is consistent with sqrt(L) scaling but not explicitly tested. A log-log regression of output_dev vs L would cleanly distinguish sqrt(L) from other scaling laws. This is a missed opportunity, not a flaw.

7. **Extrapolation to d=896 is two steps removed from evidence.** The paper chains: (a) output_dev ~ 1/d (measured at d=32..256), then (b) SOLE cosines are 90x lower than random, so divide by 90. Step (a) is supported by 4 data points. Step (b) assumes linearity between cosine and output deviation, which is plausible but unverified at production cosines. The paper correctly labels this as "extrapolated" but the 0.01% claim should carry wider uncertainty.

### Hidden assumptions

8. **Independent per-layer experts (Assumption 4).** This is correctly flagged in the Limitations section but its impact is understated. Real LoRA adapters trained on domain-specific data will have correlated weight deltas across layers -- if an expert specializes in "medical terminology," its per-layer deltas will share a consistent semantic direction. This breaks the random walk argument (Mechanism 2), which requires independent per-layer error directions. Correlated errors would compose linearly (amp_ratio approaching 1.0) rather than as sqrt(L). The paper's sub-additivity finding may be an artifact of the synthetic setup.

9. **No residual connections.** Real transformers compute h_{l+1} = h_l + Layer(h_l). The residual stream acts as a linear shortcut that preserves the input signal. This means: (a) errors propagate directly through the skip connection without dampening, and (b) the activation masking mechanism is weaker because the residual bypasses the nonlinearity. The paper notes this in Limitations but the forward pass `h_{l+1} = activation((W_l + Delta_l) @ h_l)` without residual is a fundamentally different dynamical system. The sub-additivity finding may not transfer.

10. **No LayerNorm.** Real transformers apply LayerNorm before or after each layer. LayerNorm renormalizes the hidden state, which has a complex interaction with error propagation: it can both suppress and amplify errors depending on the alignment of the error with the hidden state mean and variance. Omitting it removes a key component of the error dynamics.

## Novelty Assessment

**Prior art check.** The experiment correctly cites Ilharco et al. (2022) for task arithmetic negation and MDM-OC for reversible GS composition. The specific question "does per-layer removal error compound through depth in GS-orthogonalized LoRA composition" appears novel in this specific framing. However, the general question of error propagation through deep networks is extensively studied in the neural network stability literature (Lyapunov exponents, mean field theory of deep networks, etc.). The contribution is applying this question to the specific SOLE architecture, which is appropriate for a micro-experiment.

**Delta over parent.** The parent experiment (expert_removal_graceful) measured single-layer error. This experiment extends to L layers with nonlinear activations. The extension is non-trivial because the composition of nonlinear layers could in principle amplify errors exponentially. The finding that it does not is useful.

**No reinvention detected.** The code builds cleanly on the parent experiment's GS merge logic.

## Experimental Design

### Strengths

- **Six tests covering orthogonal dimensions.** Depth, cosine regime, activation, dimension, expert count, and position. Good coverage for a micro experiment.
- **Three seeds per configuration.** Adequate for directional claims.
- **Both near-orthogonal and clustered regimes tested.** The clustered regime is a useful stress test.
- **The amplification ratio metric is well-chosen.** It directly answers the stated hypothesis without requiring interpretation.

### Weaknesses

11. **K1 kill criterion is assessed on the wrong metric.** K1 states "cumulative removal error across L layers exceeds 3% PPL regression." The experiment measures relative output deviation (||f_naive - f_gt|| / ||f_gt||), not PPL. The paper acknowledges this in Limitation 4 but then still assesses K1 against the output deviation metric. PPL is exponentially sensitive to logit errors, so 5.31% output deviation could correspond to substantially more or less than 5.31% PPL change. The K1 assessment should explicitly state it is using a proxy metric, and the "CONDITIONAL" verdict should note this mismatch.

12. **The "clustered regime paradox" (Section 5 of MATH.md) is a normalization artifact, not a paradox.** The 313% weight error at cos~0.3 is relative to the delta norm, while output deviation is relative to the full output (which includes the base weight contribution). This is correctly explained in the paper, but calling it a "paradox" and dedicating a section to it is misleading. It is simply the difference between relative-to-delta and relative-to-total metrics.

13. **Inputs are Gaussian random vectors scaled by 0.1.** Real transformer hidden states at intermediate layers have very different distributions (structured, sparse after activation, layer-normalized). The choice of small random inputs (line 207: `inputs = rng.randn(n_inputs, d) * 0.1`) keeps activations bounded but may understate tail behavior. The max output deviation of 29.68% at d=64 suggests significant tail variance, and this could be worse with structured inputs.

14. **Test 5 (N scaling) shows a non-monotonic pattern** (amp_ratio: 0.49, 0.25, 0.33, 0.33 for N=4,8,16,32) that the paper dismisses as "weak effect." This non-monotonicity deserves explanation. At N=4, the removed expert is a larger fraction of the total (25% vs 3% at N=32), which means the perturbation is relatively larger. The amp_ratio decrease from N=4 to N=8 may be driven by relative perturbation size, not by N per se.

### Kill criteria consistency

15. **K2 is confusingly stated.** In HYPOTHESES.yml: "K2: per-layer error does NOT compound (stays additive, not multiplicative)." This reads as: the experiment is killed if error stays additive. But the paper's K2 assessment says "K2 is KILLED" meaning the hypothesis that error compounds multiplicatively is killed. The double negation is confusing. The kill criterion should be: "K2: per-layer error compounds multiplicatively (amplification ratio > 1 increasing with L)." If this holds, the approach is killed. It does not hold, so the approach survives.

## Macro-Scale Risks (advisory)

These are not blocking for micro but should be tested at macro:

- **Correlated per-layer errors from real LoRA training.** The strongest risk. If a domain expert has consistent semantic direction across layers, the random walk dampening breaks. Test: remove a trained expert from pilot 50 and measure actual amplification ratio through L=24 on Qwen 0.5B.

- **Residual connections and LayerNorm.** The synthetic model omits both. Real transformers have `h + Layer(LN(h))`, which changes error propagation fundamentally. Residual connections make error propagation closer to additive (amp_ratio near 1.0), while LayerNorm can amplify errors in low-variance directions.

- **PPL sensitivity.** Output deviation is a necessary but not sufficient proxy for PPL. A 1% output deviation in a high-entropy logit region has negligible PPL impact, while 1% in a peaked distribution can be catastrophic. Macro should measure actual PPL change from expert removal.

- **Attention layers.** The synthetic model uses dense matrix multiplication at each layer. Attention layers have quadratic interactions (Q*K^T) that may amplify errors differently than feedforward layers.

## Verdict

**PROCEED**

The experiment competently tests whether per-layer removal error compounds through depth, and conclusively shows sub-additive behavior in the synthetic setting. The core finding -- amplification ratio < 1 and decreasing with depth -- is robust across 3 seeds, 7 depth values, 4 dimensions, 3 activations, and 2 cosine regimes. The math is sound for what it claims.

The main risks are that the synthetic model omits residual connections, LayerNorm, attention, and correlated per-layer experts. These are legitimate concerns for macro transfer but are correctly acknowledged in the Limitations section. The experiment answers the question it set out to answer: "does nonlinear depth composition amplify GS removal error?" The answer is no.

**Recommended improvements (non-blocking):**

1. Rewrite K2 kill criterion in HYPOTHESES.yml to eliminate the double negation. Suggested: "K2: per-layer error compounds multiplicatively (amplification ratio > 1 increasing with L)."
2. Explicitly note that the linear activation result (amp_ratio=0.27) shows activation masking is NOT the primary dampening mechanism; direction randomization dominates.
3. Add a log-log regression of output_dev vs L to test the sqrt(L) scaling hypothesis explicitly.
4. Widen the uncertainty on the d=896 extrapolation, particularly the "~0.01%" claim which chains two unverified assumptions.
5. Rename "The Clustered Regime Paradox" to something less sensational (e.g., "Normalization effects in the clustered regime").

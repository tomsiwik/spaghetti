# Peer Review: Alpha Residual Scaling Ablation

## NotebookLM Findings

Skipped (no NotebookLM invocation needed -- the experiment is mathematically straightforward enough for direct analysis).

## Mathematical Soundness

### The scale-invariance claim is correct in principle

The core argument is: alpha = ||y_naive - y_gt|| / (||y_gt|| * sum_epsilon). If you multiply the residual contribution at every layer by a constant s, both the numerator (perturbation propagated through the network) and the denominator's ||y_gt|| term scale by the same factor of s (accumulated over L layers). The ratio cancels. This is sound.

**Step-by-step verification of the proof sketch (MATH.md Section 3.1):**

1. sum_epsilon is weight-space only -- correct, independent of forward pass. (Eq. 1 holds.)
2. The perturbation recurrence u_{l+1} = u_l + s * J_l @ u_l + s * eta_l is a correct first-order Taylor expansion of the perturbed forward pass, assuming small perturbations. The Jacobian J_l of sigma(W @ RN(.)) is independent of s because it is computed at the operating point, which itself depends on s only through h_l. However...

### Subtle gap: the Jacobian IS s-dependent through h_l

The proof says "both h_l and u_l are driven by the same s factor" and that the ratio ||u_L||/||h_L|| is s-independent. This is true **only in the linear perturbation regime** and **only if the activation function's Jacobian evaluated at s-dependent h_l does not change the ratio**.

For RMSNorm followed by GELU, this works because:
- RMSNorm projects to a sphere (unit RMS), so the pre-activation magnitude is O(1) regardless of s
- GELU's Jacobian at O(1) inputs is well-behaved

This is the critical insight the proof sketch does not fully articulate: the RMSNorm **re-normalizes** the residual stream at each layer, so even though h_l grows with s, the input to the weight matrix is always O(1). This means the operating point of the nonlinearity is genuinely s-independent, and the Jacobian cancellation is exact (not just approximate).

**Verdict: The math is sound.** The proof sketch is incomplete but the conclusion is correct. The RMSNorm normalization is the key reason the cancellation is exact, not just approximate. The paper acknowledges this informally ("RMSNorm constrains...") but could state it more precisely.

### The depth dependence alpha ~ 1/L

The paper reports alpha decreasing from 0.230 (L=4) to 0.007 (L=48). A quick check: 0.230 * 4/48 = 0.019, vs actual 0.007. So it is not exactly 1/L. Fitting the data: 0.230 * (4/L)^p for L=48 gives 0.007 when p ~ 1.36. The paper claims "alpha ~ 1/L" but the data suggest a steeper-than-1/L decay. This is not critical to the scale-invariance result but the characterization is imprecise.

### The 12.3x dampening claim

The paper says feedforward alpha=0.250 and Pre-RMSNorm alpha=0.022, giving 11.4x (MATH.md) or 12.3x (PAPER.md) dampening. 0.250/0.022 = 11.36. The paper uses both numbers inconsistently. Minor issue.

## Novelty Assessment

This is not a novel result in the broader literature. The scale-invariance of relative perturbation measures in residual networks with layer normalization is well-known in the deep learning theory community (see, e.g., analyses of Pre-LN vs Post-LN transformers by Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture"). The specific application to LoRA expert removal is narrowly novel, but the core mechanism (normalization makes relative errors scale-invariant) is standard.

**Prior art in this codebase:** No existing reference or experiment directly tests this, but the result could have been predicted from first principles without running an experiment. The adversarial review that prompted this experiment asked a reasonable question; the answer is "obvious in hindsight" given the normalization architecture.

**Delta over existing work:** The contribution is resolving an open adversarial concern about the removal_safety_complete_bound experiment. It is more of a "sanity check" than a research contribution, which is fine for a micro experiment.

## Experimental Design

### Strengths

1. **Clean A/B comparison.** Same seeds, same weights, same everything except the scale factor. This is the right experimental design.
2. **Multiple axes of variation.** d=64..256, N=8..50, L=4..48. Good coverage.
3. **3 seeds per condition.** Adequate for a deterministic-ish experiment (the forward pass is deterministic given seeds; variance comes from different weight initializations).
4. **Reports what DOES change** (absolute RMS magnitude) alongside what does not (alpha). Good practice.

### Weaknesses

1. **Only tests uniform scaling.** The experiment confirms that a uniform constant s cancels. But the adversarial concern could be reframed: production models have **learned** per-layer scaling via RMSNorm gamma parameters, which are NOT uniform across layers. The paper acknowledges this in Section 7 (Assumptions) and the Limitations section, but it is the **primary remaining risk** and is not tested at all. The kill criteria (K1, K2) do not address this.

2. **The feedforward baseline alpha=0.250 is imported, not measured.** The decomposition table relies on a value from multilayer_removal_cascade. This is fine if that experiment used the same setup, but it introduces a cross-experiment dependency that could mask inconsistencies.

3. **remove_idx = N//2 only.** The experiment always removes the middle expert. If GS ordering causes position-dependent effects (noted in VISION.md as an open question), this could systematically bias results. However, this is a minor concern for the scale-invariance question specifically, since the bias would be the same for both variants.

4. **No statistical test.** The paper reports ratio=1.00x but does not provide a confidence interval or formal test. With 3 seeds, the SEM should be reported. The claim "ratio = 1.000 +/- 0.002" appears in MATH.md but the code does not compute or report this uncertainty.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment:
- Kill criteria K1 (alpha ratio < 10x) and K2 (D < 5% at target config) are the criteria actually tested
- Both pass with extreme margin (1.00x vs 10x threshold; 0.098% vs 5% threshold)
- Status: proven -- appropriate given the results

The evidence string accurately summarizes the findings.

## Macro-Scale Risks (advisory)

1. **Learned RMSNorm gamma is the real risk.** Production Qwen2.5-7B has learnable gamma per layer. If some layers have gamma >> 1 (which is common -- gamma values of 2-5 are typical in trained transformers), the effective scaling becomes non-uniform across layers. The cancellation argument breaks. This is the experiment's own Assumption 1. Measuring gamma variance in production models is a 5-minute macro check.

2. **SiLU vs GELU.** Qwen/Llama use SiLU (swish), not GELU. The derivatives are similar but not identical. The cancellation relies on the Jacobian being evaluated at the same operating point (guaranteed by RMSNorm), so the activation choice should not matter, but it is untested.

3. **Attention softmax.** Self-attention applies softmax, which is NOT scale-invariant (softmax(x/s) != softmax(x)/s). When the residual stream magnitude changes by sqrt(L), the attention logits change, and softmax normalization does not perfectly cancel this. The paper cites the attention_self_repair experiment (2.1% effect) as evidence this is small, but that experiment tested something different (frozen attention repair, not scale-dependent attention behavior).

## Verdict

**PROCEED**

The experiment cleanly resolves the adversarial concern it was designed to address. The scale-invariance of alpha under uniform residual scaling is mathematically sound and empirically confirmed with generous margin. The kill criteria are passed decisively.

The remaining risks (learned gamma, softmax attention, SiLU activation) are genuine macro-scale concerns but are correctly identified in the paper's Limitations section and are outside the scope of what micro experiments can test. They should be checked at macro scale as quick validation items, not as blockers for this experiment.

Minor issues that do not block PROCEED:
1. The "alpha ~ 1/L" characterization is imprecise (actual decay is steeper). Not central to the claim.
2. The 11.4x vs 12.3x inconsistency between MATH.md and PAPER.md should be fixed.
3. The +/- 0.002 uncertainty claimed in MATH.md is not computed in the code; add the actual SEM.

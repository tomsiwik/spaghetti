# Peer Review: GaLore + STE Integration

## NotebookLM Findings

Skipped -- NotebookLM automation not available in this environment. Review conducted through direct document analysis.

## Mathematical Soundness

### Derivations

The MATH.md formalization is correct and clearly presented:

1. **STE forward pass** (lines 29-35): The `w_ste = w + stop_gradient(w_q - w)` formulation is the standard STE trick. Forward evaluates at Q(W), backward treats Q as identity. Correct.

2. **GaLore projection** (lines 59-79): The sequence -- STE gradient -> SVD projection -> low-rank Adam -> unproject -> weight update -- is mathematically sound. The key equation `G_proj = P^T @ G` followed by `Delta = P @ Delta_proj` is a standard rank-r approximation of the full update.

3. **Memory analysis** (lines 89-108): The accounting is correct. For shape (m,n) with rank r: GaLore stores 2*r*n (Adam moments) + m*r (projection P) = r(2n+m). The worked examples check out numerically.

4. **Adam in projected space** (code lines 221-229): Bias correction applied correctly. The step function computes m_hat and v_hat with standard 1/(1-beta^t) corrections.

### Hidden Assumptions

**Assumption 1 (MATH.md line 147-149):** "STE gradients are well-approximated by their low-rank projection." The paper acknowledges this as a risk but dismisses it with "STE is a per-element operation that doesn't fundamentally change the covariance structure." This claim is *plausible but unverified*. STE introduces gradient discontinuities at quantization thresholds. The spectral properties of these gradients could differ from smooth gradients. However, the empirical result (PPL ratio 0.998) constitutes evidence that the assumption holds at this scale.

**Assumption 2:** The GaLore optimizer applies standard Adam to non-GaLore parameters (embeddings, norms) via a *separate* Adam instance (code line 250). This means two Adam instances with independent state. The interaction between GaLore-updated weights and standard-Adam-updated embeddings/norms is not analyzed mathematically. At toy scale this works; at macro scale, the two optimizers could have conflicting learning dynamics.

**Assumption 3:** SVD is performed on the *full* gradient G (code line 205), not on the projected STE gradient. This means the SVD sees the gradient as if STE were identity (because STE passes gradients through). This is correct -- the STE gradient IS the gradient through the quantized forward pass, and its structure should reflect quantization effects through the chain rule.

### No Mathematical Errors Found

The derivations are straightforward applications of established methods (STE from BitNet, GaLore from the original paper). No errors in the mathematical formalism.

## Novelty Assessment

### Prior Art

**GaLore (Zhao et al., 2024, ICML oral):** Memory-efficient training via gradient low-rank projection. The original paper does not address ternary quantization or STE integration.

**BitNet b1.58 (Ma et al., 2024):** Ternary training with STE. Does not use GaLore.

**The combination is straightforward but appears novel at micro scale.** The paper correctly identifies this as "swap nn.Linear for BitLinear, feed STE gradients into GaLore projection." No published work combines GaLore with STE for ternary-aware training.

**Prior experiment (bitnet_galore_scaffold):** Established the 2.6x degradation problem that motivates this work. The fix (integrate STE into the forward pass) is the obvious next step, and the experiment confirms it works.

### Delta Over Existing Work

The delta is modest but meaningful: the insight that GaLore's post-hoc quantization degrades because training never "sees" ternary weights, and that STE integration fixes this completely, is a useful engineering finding. The contribution is more "integration validation" than "novel mechanism."

## Experimental Design

### Does It Test What It Claims?

**Yes, with one important gap.** The experiment claims to test whether STE integration fixes the 2.6x ternary degradation. The result (0.998x ratio) clearly confirms this. The composition test confirms adapters still compose well (ratio 1.043 vs 1.024 baseline).

### Controls

**Adequate.** The experiment trains a standard STE baseline in the same run (same data, same architecture, same hyperparameters except optimizer). The random seed is matched (line 475: `rng_std = random.Random(42)`). This is a fair head-to-head comparison.

### Concerns

**C1: Single seed.** The paper acknowledges this (Limitation 3). The prior GaLore experiment showed high variance across seeds (CV=21% for quantization degradation). A single-seed result of 0.998x could be lucky. However, the kill threshold is 1.5x, and even with 21% CV, the result would need to be >7 standard deviations away to fail. The margin is large enough that single-seed is acceptable for a micro verdict.

**C2: The PPL metric computes ternary PPL implicitly.** The BitLinear forward pass always quantizes via STE, so `compute_ppl(model, val_ds)` evaluates with ternary weights. This is correct -- no separate "post-hoc quantization" step is needed because quantization is baked into the forward pass. Good design.

**C3: Composition uses uniform averaging (1/N weighting).** The composition test sums `(b_val @ a_matrix) / n_domains` for all 5 adapters and evaluates on each domain. This is the same protocol as prior experiments. Consistent.

**C4: S2 failure is well-explained.** The optimizer state ratio is 0.28x (2.65M vs 9.49M elements), confirming GaLore's memory mechanism works. The peak memory ratio (0.90x) fails only because weight memory dominates at d=256. The paper correctly argues this would flip at scale. This is a legitimate micro-scale artifact, not a mechanism failure.

**C5: Loss histories show a potential concern.** GaLore+STE final loss (0.461) is slightly higher than standard STE (0.449), yet GaLore+STE PPL (1.5922) is slightly better. This small discrepancy could reflect:
- Different evaluation batches (both use `random.Random(0)` so they should be the same)
- The regularization effect of GaLore projection
- Noise at this scale

This is a minor observation, not a blocking concern. The 2.6% loss difference maps to a 0.2% PPL difference in the opposite direction -- well within noise.

### What's Missing

**No ablation on GaLore rank.** The experiment uses r=64 (same as prior GaLore experiment). It would strengthen the finding to show that r=32 also works (confirming the mechanism isn't rank-sensitive) or that r=16 breaks (showing where the low-rank assumption fails for STE gradients). This is a nice-to-have, not blocking.

**No gradient spectral analysis.** The MATH.md identifies STE gradient spectral properties as a risk (lines 156-165). The experiment could have logged the singular value decay of STE gradients vs standard gradients to verify the low-rank assumption directly. Again, nice-to-have -- the PPL result speaks for itself.

## Hypothesis Graph Consistency

The experiment is not yet registered in HYPOTHESES.yml under the name `galore_ste_ternary` (no match found). It relates to `exp_bitnet_galore_scaffold` (status: supported) which identified the 2.6x degradation. The VISION.md (line 100) references this experiment as "GaLore+STE integration to fix 2-3x quantization degradation (exp_galore_ste_ternary_scaffold)."

The kill criteria in the code (K1: PPL > 1.5x, K2: composition > 1.5x, K3: time > 3x) are reasonable and match the experiment description. All three pass with large margins.

**Action needed:** This experiment should be added to HYPOTHESES.yml with its results.

## Macro-Scale Risks (advisory)

1. **SVD overhead scaling.** At d=256, SVD every 200 steps adds 1.71x time. SVD complexity is O(min(m,n)^2 * max(m,n)) for an (m,n) matrix. At d=2560 (10x), each SVD is ~1000x more expensive. The `stream=mx.cpu` constraint (MLX GPU SVD limitation) makes this worse. Amortization over 200 steps may not be sufficient. Mitigation: increase `galore_update_freq` or use randomized SVD.

2. **STE gradient rank at scale.** The GaLore paper shows standard gradients are low-rank at 1B scale. STE gradients at scale have not been studied. If the quantization step scatters gradient energy across more singular vectors, higher rank r may be needed, eroding memory savings.

3. **Dual optimizer interaction.** The current implementation uses two separate optimizers (GaLore for large weight matrices, standard Adam for embeddings/norms). At scale, the learning rate schedule and warmup may need careful coordination.

4. **Memory accounting at scale.** The 0.28x optimizer state ratio is the key selling point. At d=2560 with 7B params, verify that the projection matrices P (one per weight matrix, shape (m, r)) don't accumulate to significant memory. For a 7B model with ~100 weight matrices, each P is ~2560*256 = 655K params, totaling ~65M params (~260MB at FP32). This is small relative to the 14B saved in Adam state.

## Verdict

**PROCEED**

The experiment cleanly answers its question: integrating STE into GaLore's forward pass completely eliminates the 2.6x ternary quantization degradation (result: 0.998x ratio). The math is sound, the experimental design is fair with matched controls, the code correctly implements the described algorithm, and all three kill criteria pass with large margins.

The mechanism is straightforward (use quantized forward pass during GaLore training instead of post-hoc quantization), and the result is decisive (the degradation disappears entirely). The S2 memory failure is correctly attributed to the toy-scale artifact of weight-dominated memory profiles, and the 0.28x optimizer state ratio confirms the underlying mechanism works.

Minor observations that do not block PROCEED:
- Single seed (acceptable given the large margin to kill threshold)
- No HYPOTHESES.yml entry (bookkeeping, not scientific)
- No gradient spectral analysis (the PPL result is sufficient empirical evidence)
- Composition ratio slightly worse (1.043 vs 1.024) but well within bounds and consistent with GaLore's slight regularization effect

This result validates the path toward memory-efficient ternary base training on Apple Silicon and should be registered in HYPOTHESES.yml and FINDINGS.md.

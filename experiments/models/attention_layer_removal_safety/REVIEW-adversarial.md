# Peer Review: Attention Layer Removal Safety

## NotebookLM Findings

Skipped -- the experiment is sufficiently self-contained and well-documented that a deep NotebookLM review is not required. The math is tractable by direct inspection.

## Mathematical Soundness

### Cosine of concatenated deltas (Section 2.2 of MATH.md)

The formula is correct:

    cos(delta_i, delta_j) = (delta_i^attn . delta_j^attn + delta_i^mlp . delta_j^mlp) /
                            (||delta_i|| * ||delta_j||)

However, the simplification to a weighted average assumes "comparable norms" between attention and MLP sub-deltas. The code (run_experiment.py, line 77) scales all deltas uniformly (0.008-0.012), so this holds by construction. In production, attention and MLP deltas can have very different magnitudes -- the norm ratio between attention and MLP portions of a real LoRA adapter is not controlled. This is acknowledged in Assumption 4 but should be tested at macro. If MLP norms dominate (likely, since D_mlp >> D_attn), the joint cosine will be even lower than 0.171, making the hybrid case stronger.

**Verdict: Sound with stated assumptions.**

### Hybrid error analysis (Section 3.3)

The bound E_mlp ~ cos_mlp * (N - k - 1) * ||delta^mlp|| is inherited from the parent experiment. The derivation is correct: each subsequent expert in the GS ordering has its projection onto the removed expert subtracted. Naive subtraction leaves these cross-terms. The relative error of 2.4% is a worst-case additive bound. The measured 0.06-0.09% is well below this, which is consistent.

**Verdict: Correct. The bound is conservative but valid.**

### GS recompute complexity (Section 4)

O(N^2 * D) for GS is correct. The speedup factor calculation D_full / D_attn = 5.07 is algebraically correct for Qwen 0.5B dimensions. The extrapolation from D_SINGLE to D_ATTN is linear in D, which is correct since GS dot products scale linearly with dimension.

**Verdict: Sound.**

### Non-monotonic error curve (Section 5)

The claim that error peaks at cos~0.5-0.7 is empirically demonstrated and the qualitative explanation is reasonable: at very high cosine, GS collapses deltas to near-zero, so the absolute error of naive subtraction is small. However, the paper says "the relative error can go either way" without providing a formal derivation of where the peak occurs. This is a minor gap -- the empirical sweep is sufficient to establish the shape.

**Verdict: Empirically sound, theoretical explanation is qualitative but adequate.**

## Novelty Assessment

This experiment is a direct extension of the parent (expert_removal_graceful), which established the regime boundary at cos~0.01. The novelty here is:

1. **Per-layer decomposition of the removal problem.** Treating attention and MLP as separate GS domains rather than concatenating them. This is a natural and correct engineering insight.
2. **Joint GS is wrong for mixed-cosine regimes.** The finding that joint GS on concatenated [attn; mlp] deltas produces >100% error (Test 5) is genuinely useful. This is a trap that production code could easily fall into.
3. **The hybrid strategy.** GS for high-cosine layers, naive subtraction for low-cosine layers. Practical and well-motivated.

**Prior art check:** MDM-OC (arXiv:2507.20997) is cited and uses GS with learned coefficients. The per-layer decomposition is a natural extension not explicitly addressed in MDM-OC. Task arithmetic (Ilharco et al. 2022) covers naive subtraction but not the GS-composed case. The delta is incremental but genuine.

## Experimental Design

### Strengths

1. **Six tests covering different aspects.** The experiment is thorough for a micro-scale study.
2. **Controlled cosine generation is well-implemented.** The alpha/beta mixing (line 64-65) correctly produces vectors with target pairwise cosine. The orthogonalization of the unique component against the shared direction (lines 72-73) ensures the cosine is alpha^2 = target_cos.
3. **K2 tested at actual D_ATTN dimension** (Test 6). This avoids relying solely on extrapolation.
4. **Three seeds** for Tests 1, 3, 4. Adequate for a micro experiment.

### Weaknesses

**W1: The "ground truth" for the hybrid strategy is itself per-layer GS, not end-to-end model quality.** The hybrid strategy (Test 4-5) computes reconstruction error against per-layer GS recompute as ground truth. But the paper's production recommendation assumes per-layer GS recompute IS the correct answer. The question of whether per-layer GS composition (separate GS on attn, separate GS on MLP) produces the same result as single-layer-at-a-time GS on the actual transformer layer is never tested. In a real transformer, the attention output feeds into the MLP -- they are not independent. However, since LoRA deltas are additive (W + BA) and the GS is applied to the flattened deltas before merging into W, the per-layer decomposition is correct in weight space. The cross-layer dependency is a different question (addressed by multilayer_removal_cascade). This is acceptable.

**W2: Test 5 "joint GS" baseline is unfair.** The paper reports "joint GS on concatenated deltas produces >100% error." But the ground truth is per-layer GS. So this is comparing joint GS against per-layer GS, which are fundamentally different operations. The 116% error means joint GS produces a very different merged model than per-layer GS -- but it does not mean joint GS is "wrong" in absolute terms. It means joint GS and per-layer GS produce different compositions. The paper frames this as "joint GS is wrong," but the correct framing is "if you compose per-layer, you must also remove per-layer." This is a framing issue, not a fundamental error.

**W3: Timing is CPU-only (Apple Silicon numpy).** GS recompute in production would likely run on GPU. GPU timing could be very different (likely faster due to batched dot products). The 5.84s at N=50 on CPU could be <1s on GPU. This strengthens the conclusion but the paper should note it. The limitation is acknowledged ("Apple Silicon, pure numpy, no GPU") but not discussed in the production recommendation.

**W4: Only middle-expert removal tested.** All tests remove expert at index N//2. The GS ordering matters -- removing the first expert (which all others projected against) should produce worse naive subtraction error than removing the last. This was tested in the parent experiment but not here. Since the parent already established this, it is acceptable but worth noting.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment:

- **K1:** "naive subtraction error >3% for attention layers at cos=0.85" -- correctly TRIGGERED (expected outcome, confirms parent).
- **K2:** "GS recompute for attention layers takes >10s at N=50" -- correctly PASS at 5.84s.
- **Status: supported** -- appropriate. The hybrid strategy is validated at micro, but real adapter cosine values at macro are unknown.

The experiment depends on exp_expert_removal_graceful (the parent), which is listed as a dependency. Consistent.

## Macro-Scale Risks (advisory)

1. **Attention cosine at macro is unknown.** The 0.85 figure comes from non-converged micro models. If it is lower at macro (plausible), the hybrid strategy becomes even more favorable. If higher, GS recompute is still fast.
2. **GPU timing for GS recompute.** Should be benchmarked. Likely much faster than CPU, making the 10s threshold non-binding.
3. **Per-matrix vs joint-attention GS.** The paper notes that per-matrix GS (separate GS on Q, K, V, O) is 4x cheaper than joint-attention GS. This should be the default production strategy and needs macro validation.
4. **Real adapter norms.** The equal-norm assumption should be checked against pilot 50 adapters.

## Verdict

**PROCEED**

This is a clean, well-executed extension of the parent experiment. The core finding -- that per-layer decomposition of the removal operation is both correct and necessary for mixed-cosine regimes -- is sound and practically useful. The hybrid strategy (GS for high-cosine attention, naive for low-cosine MLP) achieves 0.06-0.09% error, well within tolerance. The joint GS >100% error finding is a valuable guardrail against a production pitfall.

The weaknesses identified are framing issues (W2) and known micro-scale limitations (W3, W4) that the paper already acknowledges. No mathematical errors found. No blocking issues.

The experiment correctly advances the VISION.md roadmap by closing the attention-layer gap in the expert removal story, enabling the per-layer removal protocol recommended for production.

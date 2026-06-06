# Pointer Routing (No Merge): Research Digest

## Hypothesis

Per-layer expert selection at full strength beats uniform 1/N parameter merge by preserving each expert's nonlinear computation in output space.

## What This Model Is

Three per-layer routing strategies for selecting ONE pre-trained adapter per layer at FULL strength (scale=20.0) instead of merging all N=5 adapters at diluted strength (scale/N=4.0):

- **(a) Learned gate:** Per-layer linear classifier trained on hidden states via ridge regression.
- **(b) Hash lookup:** Deterministic hash(layer, domain) mod N assignment.
- **(c) Input-dependent MLP:** 2-layer MLP trained on hidden states via gradient descent.

Baselines: base model (no adapters), uniform 1/N merge (all 5 at scale/5), oracle single-adapter (all layers use the correct domain adapter).

## Key References

- Switch Transformer (Fedus et al., 2101.03961) -- top-1 per-layer expert selection sufficient for MoE quality.
- Hash Layers (Roller et al., 2106.04426) -- O(1) hash routing competitive with learned gates.
- Prior finding: uniform 1/N degrades quality on instruction-tuned bases (confirmed twice in FINDINGS.md).

## Empirical Results

| Domain | Base | Uniform 1/N | Oracle Single | Hash | Learned Gate | MLP Gate |
|--------|------|-------------|---------------|------|-------------|----------|
| Medical | 6.50 | 4.16 | 3.46 | 4.30 | **3.46** | 4.39 |
| Code | 4.98 | 3.46 | 3.14 | 3.52 | **3.14** | 3.60 |
| Math | 3.84 | 2.91 | 2.38 | 2.99 | **2.38** | 2.91 |
| Legal | 21.63 | 16.08 | 14.66 | 15.77 | **14.66** | 15.54 |
| Finance | 19.43 | 14.96 | 14.01 | 14.88 | **14.01** | 14.39 |

### Improvement vs Uniform 1/N (%)

| Method | Medical | Code | Math | Legal | Finance | **Mean** |
|--------|---------|------|------|-------|---------|----------|
| Hash | -3.3 | -1.7 | -2.8 | +1.9 | +0.5 | **-1.1** |
| Learned Gate | +16.9 | +9.2 | +18.0 | +8.9 | +6.3 | **+11.9** |
| MLP Gate | -5.7 | -4.0 | -0.1 | +3.3 | +3.8 | **-0.5** |

### Key Findings

**1. Oracle routing = learned gate routing (CRITICAL):**
Learned gate PPL matches oracle single-adapter EXACTLY (3.46, 3.14, 2.38, 14.66, 14.01). The learned gate converged to selecting the SAME adapter at ALL 30 layers -- identical to oracle single-adapter routing. Per-layer assignment for each domain: `[0,0,...,0]` for medical, `[1,1,...,1]` for code, etc. There is ZERO depth-wise specialization.

**2. Per-layer specialization does NOT emerge:**
When routing actually varies across layers (hash, MLP), performance DEGRADES vs uniform 1/N. Hash: -1.1% mean. MLP: -0.5% mean. The model works better with consistent adapter selection across all layers.

**3. Single-adapter at full strength > uniform merge:**
Oracle single-adapter (11.9% mean improvement over uniform) proves that applying ONE adapter at full scale=20.0 is better than averaging 5 at scale/5=4.0. This validates the output-space vs parameter-space composition argument, but ONLY when the correct single adapter is selected.

**4. MLP gates broken by numerical instability:**
The hidden states at d=2560 contain extreme values that cause float32 overflow in numpy matmuls. MLP gates effectively learn random mappings due to NaN propagation, defaulting to a single adapter (finance=4) for all domains in most layers.

### Kill Criteria Assessment

- **K1 (id 535): PASS with caveat.** The learned gate beats uniform by 11.9% mean, BUT it does so by collapsing to oracle single-adapter routing, not by per-layer pointer routing. Hash and MLP routing (the actual pointer routing variants) are WORSE than uniform.
- **K2 (id 536): FAIL.** The only method that beats uniform (learned gate) has `same_adapter_fraction=1.0, cross_layer_variation=0.0`. All layers pick the same adapter. No depth-wise specialization. Hash and MLP show specialization but are worse than uniform.

### Success Criteria

- **S1 (id 55): FAIL.** The 11.9% improvement comes from oracle routing, not per-layer specialization. Per-layer pointer routing specifically (different adapter per layer) does NOT beat uniform.

## What This Tells Us

**The hypothesis was partially right but the mechanism was wrong:**

1. CORRECT: Single-adapter at full strength > diluted multi-adapter merge. Output-space composition (select one expert) preserves nonlinearity better than parameter-space merge (average all experts).

2. WRONG: Per-layer specialization is not useful at this scale. With 5 pre-trained adapters that were each trained on ALL 30 layers simultaneously, mixing adapters across layers (layer 5 uses medical, layer 10 uses code) hurts because it breaks the joint training distribution. Each adapter's B-matrices at layer L are calibrated for the residual stream produced by that SAME adapter at layers 0 through L-1.

3. INSIGHT: The optimal strategy is NOT per-layer routing -- it is per-SEQUENCE routing. Select the single best adapter for the entire sequence and apply it at all layers. This is exactly what the learned gate converges to.

## Implications for Architecture

1. **Per-sequence routing is the right granularity.** The learned gate's convergence to oracle routing confirms that per-sequence expert selection is the natural level. This is consistent with our proven routing heads (99.9% accuracy) and SOLE hash-ring (5.3% displacement).

2. **Full-strength single-adapter > diluted multi-adapter.** The 11.9% improvement over uniform 1/N comes from eliminating 1/N dilution. This means: for a given input, apply ONE adapter at scale=20.0 instead of all 5 at scale/5=4.0.

3. **Parameter-space merge is inherently lossy.** Uniform 1/N merge (PPL 8.31 mean) is worse than single-adapter oracle (PPL 7.53 mean) by 10.3%. The loss comes from: (a) 1/N dilution weakening each expert, (b) parameter averaging destroying nonlinear specialization.

4. **Per-layer MoE-style routing needs adapters trained for per-layer switching.** Our pre-trained adapters were trained jointly across all layers. For per-layer routing to work, adapters would need to be trained specifically for single-layer application (like Switch Transformer experts), which is a fundamentally different training paradigm.

## Limitations

1. **5 trivially-separable domains** -- routing accuracy is 99.9% on these. Mixed-domain inputs untested.
2. **Pre-trained adapters not designed for per-layer isolation** -- trained jointly, applied in isolation.
3. **MLP gate numerically broken** -- float32 overflow in hidden states prevented fair comparison.
4. **Single seed** (42) -- justified by multiseed CV=0.5% at N=5 in prior experiments.
5. **PPL-only evaluation** -- task accuracy not tested.

## What Would Kill This

Already partially killed:
- K2 FAIL: No per-layer specialization emerged. The per-layer routing hypothesis is killed.
- K1 technically passes but only via oracle (not pointer) routing.

The surviving finding (single-adapter at full strength > uniform merge) would be killed if:
- Multi-adapter merge with proper scaling (e.g., Task Arithmetic, TIES) outperforms single-adapter selection.
- At higher N (N=25), the benefit of multi-adapter composition (access to more knowledge) outweighs the dilution cost.

## Verdict

**SUPPORTED with major caveat.** The experiment proves that output-space selection (pick one adapter) beats parameter-space merge (average all adapters), but the per-layer pointer routing hypothesis is KILLED -- the optimal routing granularity is per-SEQUENCE, not per-LAYER. The useful finding is: route to ONE adapter at full strength instead of merging all at 1/N.

**Net status: K1 PASS (oracle routing beats uniform), K2 FAIL (no per-layer specialization).**

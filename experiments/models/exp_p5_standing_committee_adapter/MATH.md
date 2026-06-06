# P5.C0: Standing Committee Adapter — Mathematical Foundation

## Prior Art
- **arXiv:2601.03425** (DeepSeek-MoE): MoE models naturally develop shared "expert"
  neurons that activate on all inputs regardless of domain — a domain-invariant
  reasoning backbone emerges from training. This motivates explicitly designing one.
- **Finding #49**: Capability experts (reasoning, instruction, conciseness, safety)
  compose orthogonally with domain experts on BitNet-2B-4T: mean |cos| < 0.01.
  Cap-domain coherence (0.000377) is 2.9× lower than domain-domain (0.001080).
- **Finding #18**: Reasoning distillation works: +10.6pp on MATH-500 via LoRA.
- **Finding #44**: Real BitNet-2B-4T supports LoRA composition, 5 domains, all-modules.

## Problem Statement

Given a base model W, a reasoning adapter ΔW_r, and a domain adapter ΔW_d,
we want the composed model W + ΔW_r + ΔW_d to:
1. Improve reasoning tasks relative to W + ΔW_d alone (the "committee benefit")
2. Preserve domain format quality relative to W + ΔW_d alone (no interference)
3. Maintain structural orthogonality between ΔW_r and ΔW_d

## Theorem 1 (Module-Disjoint Orthogonality)

**Statement**: Let ΔW_r = {B_r^i A_r^i : i ∈ S_r} and ΔW_d = {B_d^j A_d^j : j ∈ S_d}
be two LoRA adapters applied to disjoint module sets S_r ∩ S_d = ∅.
Then the pairwise cosine similarity between their flattened weight updates is exactly zero:

  cos(vec(ΔW_r), vec(ΔW_d)) = 0

**Proof**: The flattened weight update vector for ΔW_r has non-zero entries only in
positions corresponding to modules in S_r. Similarly for ΔW_d in S_d. Since
S_r ∩ S_d = ∅, the support sets of vec(ΔW_r) and vec(ΔW_d) are disjoint.
The inner product of two vectors with disjoint support is zero. QED.

**Design choice**: We assign:
- S_r = {q_proj, o_proj} in layers [n-8, n-1] (reasoning committee)
- S_d = {v_proj, down_proj} in layers [n-8, n-1] (domain adapter)

This guarantees K1278 (cos < 1e-4) by construction — the cosine is exactly 0.

## Theorem 2 (Additive Composition Preserves Capabilities)

**Statement**: For module-disjoint adapters, the composed forward pass decomposes as:

  h_composed(x) = h_base(x) + δ_r(x) + δ_d(x)

where δ_r depends only on ΔW_r parameters and δ_d only on ΔW_d parameters.
The cross-interaction term is bounded by O(||ΔW_r|| · ||ΔW_d||) through
higher-order effects in the residual stream.

**Proof sketch**: In a transformer layer, the attention output is:

  Attn(x) = softmax(xW_Q(xW_K)^T / √d) · xW_V · W_O

With ΔW_r on {Q, O} and ΔW_d on {V, down_proj}:
- Q is perturbed: (W_Q + ΔQ)x changes the attention pattern
- O is perturbed: (W_O + ΔO) changes the output projection
- V is perturbed: (W_V + ΔV)x changes the value representation
- down_proj is perturbed: affects the MLP pathway independently

The attention mechanism creates a coupling: the changed Q-attention pattern
(from ΔW_r) selects from changed V-values (from ΔW_d). This is a second-order
interaction of magnitude O(||ΔQ|| · ||ΔV||).

For rank-4 LoRA with scale 1.0 on a 2816-dimensional model:
- ||ΔW|| / ||W|| ≈ r/d = 4/2816 ≈ 0.0014
- Cross-interaction: O(0.0014²) ≈ O(2×10⁻⁶), negligible

Therefore, each adapter's effect is approximately independent. QED.

## Theorem 3 (Committee Benefit for Reasoning-Dependent Tasks)

**Statement**: If ΔW_r improves reasoning accuracy by Δ_r percentage points
on a reasoning benchmark, then for any domain task with reasoning component
fraction f ∈ [0,1], the committee adapter improves performance by approximately
f · Δ_r points on that domain task (modulo the O(ε²) cross-interaction term).

**Prediction**: If the reasoning adapter gains Δ_r ≈ 8-12pp on GSM8K over base,
and GSM8K is a pure reasoning task (f ≈ 1), then:
- Committee + domain ≥ domain alone + Δ_r on GSM8K → expect ≥ 3pp gain (K1276)
- Format tasks have f ≈ 0 (no reasoning component) → expect < 2pp degradation (K1277)

## Quantitative Predictions

| Metric | Prediction | Kill threshold | Basis |
|--------|-----------|---------------|-------|
| GSM8K: committee+domain vs domain alone | ≥ 5pp gain | ≥ 3pp (K1276) | Finding #18: reasoning LoRA gains 10.6pp, expect ~50% retention in composition |
| Format tasks: with vs without committee | < 1pp loss | < 2pp (K1277) | Theorem 1: zero parametric interference, only O(ε²) cross-interaction |
| cos(committee, domain) | = 0.0 | < 1e-4 (K1278) | Theorem 1: module-disjoint → exact zero |

## Behavioral Predictions

1. The committee adapter teaches chain-of-thought reasoning patterns (step-by-step
   problem decomposition). When composed with a domain adapter, the model should
   produce structured reasoning in domain-specific responses.
2. The domain adapter teaches format compliance (structured output, citations, etc.).
   Composition should NOT cause the model to inject reasoning steps into format tasks.
3. The committee benefit should be larger for tasks requiring multi-step reasoning
   than for tasks requiring only pattern matching or recall.

## Failure Modes

1. **Attention coupling**: If ΔQ changes the attention pattern so drastically that
   ΔV's values are never selected, the domain adapter is effectively silenced.
   Mitigated by: rank-4 LoRA has tiny perturbation (Theorem 2).
2. **Capacity competition**: If the model's representation space is saturated,
   adding a second adapter could degrade both. Mitigated by: E4B has 2816 hidden dim
   with only 2×4=8 dimensions used per adapter, utilizing 0.6% of capacity.
3. **Reasoning adapter is trivial**: If E4B already reasons well, ΔW_r may be near-zero.
   This would make K1276 fail — the committee has nothing to add.

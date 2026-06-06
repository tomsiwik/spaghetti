# LEARNINGS: exp_m2p_activation_scaling

**Status:** supported (K903+K904 PASS, K905 FAIL — genuine)
**Finding:** #372
**Closes:** Level 2B (Critique #6 — activation-space interference)

---

## Core Finding

Activation-space interference between composed M2P adapters grows **sub-linearly** with N: `max_cos ~ 0.137 * N^0.379` (R²=0.90, per-token worst-case metric). Geometric interference is **bounded and not the bottleneck** at N=10. The quality failure at N=10 (K905) is a **composition strategy failure** — equal-weight 1/N dilution — not a geometry failure.

---

## Why This Happened

### Sub-linear interference growth (K903, K904 PASS)

The sub-linear exponent α=0.379 < 0.5 is explained by the structure of B-matrices trained on different domains. The CLT upper bound (α=0.5 for i.i.d. increments) is not reached because B-matrices trained on distinct tasks have limited cross-domain correlation — each B_i learns domain-specific output directions that partially avoid each other through task specialization, not by design. This is consistent with:

- **Random projection lower bound**: E[|cos(B_i A_i x, B_j A_j x)|] = O(1/√d_out) ≈ 0.063 for wq (d_out=256). Measured mean cosine (0.052–0.061) sits right at this floor, confirming that *on average* interference is random-walk-level. Only worst-case pairs drive the maximum above 0.133.
- **B-matrix diversity from different task objectives**: Domains with fully different output alphabets (arithmetic sequences vs. character permutations) cannot share B-space directions efficiently. The correlation is below i.i.d. random.

The adversarial reviewer correctly noted that the CLT argument is not exact — the maximum over C(N,2) pairs grows faster than the sum (Bonferroni correction pushes the expected max above √N for i.i.d. pairs). That α=0.379 < 0.5 holds despite this correction is actually *stronger* evidence of anti-correlated B-matrices.

### fc1 dominance at large N

At N≥8, fc1 (d_out=1024, 4×d_model) becomes the worst-case module, exceeding wq (d_out=256). This is consistent with wider B-matrices having more capacity to learn domain-specific output directions that *happen* to overlap. At macro scale (d_ffn typically 4–8× d_model), fc1-class modules will dominate; wq is not the representative module. **Future activation interference measurements must prioritize fc1.**

### Equal-weight dilution (K905 FAIL — genuine)

At N=10, each adapter receives weight 1/N = 10%. For domains where SFT_delta is small (sort: 0.228 nats, reverse: 0.099 nats), 10% weight is insufficient — the remaining 90% from unrelated adapters actively degrades signal. This causes comp_loss > base_loss for 4 of 10 domains. This is NOT a metric artifact:

| Domain | base_loss | comp_loss | Δ |
|--------|-----------|-----------|---|
| reverse | 2.221 | 2.795 | +0.574 nats |
| sort | 2.185 | 2.772 | +0.587 nats |
| cipher | 3.749 | 4.204 | +0.455 nats |
| mapping | 6.639 | 6.693 | +0.054 nats |

This is the **4th independent occurrence** of the parity-class pattern (Findings #363, #364, #366, #372): domains with SFT_delta < ~0.3 nats are nearly solved by the base model and cannot survive dilution in equal-weight composition.

### Measurement artifact in prior run corrected

The prior run measured *global trajectory cosine* (flattened across all T token positions), producing an apparent plateau at 0.128 after N=3. The plateau was a measurement artifact: flattening averages away worst-case positions. The per-token metric takes the maximum over each token separately — worst-case input characterization. The per-token max grows monotonically from 0.189 (N=2) to 0.339 (N=10). **Always use per-token max for adversarial interference claims.**

---

## Confirming Evidence

**DARE** (arXiv:2311.03099, "Language Models are Super Mario"): sparsification of delta parameters eliminates 90–99% of deltas without performance loss, enabling multi-model merging by reducing parameter-space interference. Confirms that *most* of a fine-tuned model's delta is redundant — consistent with low mean activation cosine (0.052–0.061 ≈ random floor) even when worst-case pairs interfere.

**MoLoRA** (arXiv:2603.15965, Microsoft Research): Per-token adapter routing — exactly the mechanism needed to fix K905. Qwen3-1.7B + MoLoRA exceeds Qwen3-8B (4.7× larger) on GSM8K (+14%), MATH (+8%), BBH (+2.5%). This is direct experimental evidence that **per-token learned routing beats equal-weight composition at scale**, validating our diagnosis that 1/N dilution is the failure mode.

**LoraHub** (arXiv:2307.13269): Gradient-free optimization of LoRA combination weights via few-shot examples. Non-uniform weights are strictly better than equal weights for cross-task generalization — the paper specifically avoids equal-weight composition as a baseline.

**Mixture of LoRA Experts / MoLE** (arXiv:2404.13628): Hierarchical learned gating per-layer for optimal composition weights. Treats each LoRA as a distinct expert, trains gating to learn task-specific weights. Confirms the MoE-LoRA direction is well-established for overcoming fixed-weight composition failures.

---

## Contradicting Evidence

**LoRI** (arXiv:2504.07448, "Reducing Cross-Task Interference in Multi-Task LoRA"): focuses on *parameter-space* interference reduction for merging, implying that activation-space interference is not the dominant concern in practice. This partially contradicts our K905 narrative — if parameter-space merging already reduces interference, our equal-weight *composition* (not merging) may be a separate concern. **Key distinction:** LoRI addresses weight merging (permanent); our system composes at inference time (reversible). These are different failure modes.

**Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging** (arXiv:2505.22934): achieves near-zero interference via orthogonal subspace projection of LoRA weights. Their approach is in weight space (A and B matrices projected to be orthogonal across adapters), not our M2P approach (A-slots trained orthogonal, B-slots free). Their finding that orthogonal projections eliminate interference contradicts the assumption that trained B-matrices will naturally decorrelate — they require explicit enforcement.

---

## Alternative Approaches

1. **MoLoRA-style per-token routing** (arXiv:2603.15965): Learn a gating function g_θ(x_t) → softmax weights over N adapters per token. This directly fixes K905 by assigning near-zero weight to non-relevant adapters. Compatible with our M2P framework (replace fixed 1/N with learned weights). **Recommended for Level 3.**

2. **Minimum-weight threshold**: Simple heuristic — route query to top-k adapters by TF-IDF or embedding similarity (Finding #354), discard remaining N-k. Already proven in our TF-IDF routing experiment (Finding #354). Immediate fix for K905 without training a gating network.

3. **DARE sparsification preprocessing** (arXiv:2311.03099): Apply random drop + rescale to M2P deltas before composition. If 90% of delta parameters are droppable without quality loss, per-token interference should be dramatically reduced. Compatible with our framework, no routing network needed. Caveat: tested on SFT merging, not M2P composition; applicability is speculative.

4. **Orthogonal B-matrix projection** (arXiv:2505.22934): Explicitly project B-matrices to be orthogonal across adapters after training. Would guarantee activation-space interference → 0, but at the cost of domain-specific signal. Not recommended as primary approach — our B-matrices need domain specificity.

---

## Implications for Next Experiments

1. **Equal-weight composition is dead at N>5.** All future N>5 composition experiments MUST use routing (TF-IDF, MoLoRA-style, or similarity-based). No equal-weight baselines unless specifically studying the failure mode.

2. **fc1 is the binding module at scale.** Any interference measurement in deeper/wider networks must instrument fc1 (or equivalent FFN expansion) as the primary metric. wq (d_out≤d_model) will understate worst-case interference.

3. **Parity-class detection is mandatory preprocessing.** Any composition experiment MUST identify domains with SFT_delta < 0.3 nats and either (a) exclude them from equal-weight composition or (b) route them exclusively to their own adapter. This is now a 4× replicated requirement.

4. **Level 2B is CLOSED.** Critique #6 (activation-space interference unbounded?) is resolved within the measured scope (layer 0, wq+fc1, N≤10). No further activation-space scaling experiments needed at toy scale.

5. **Next priority: Level 3A** — M2P on Qwen3-0.6B + GSM8K (real language, first NLP domain, resolves Critique #3). With 32k BPE vocabulary, cross-domain transfer at toy-alphabet scale cannot recur.

---

## Recommended Follow-Up

**exp_m2p_qwen06b_gsm8k** (P0 — already designed):
- MOTIVATION: Level 3A gate, resolves Critique #3 (no NLP), first real-language domain
- LITERATURE: Aghajanyan (2012.13255) d_int is task-determined; MoLoRA (2603.15965) demonstrates per-token routing on real reasoning benchmarks
- ROUTING: Use TF-IDF routing (Finding #354) — avoids equal-weight K905 failure mode
- K905-class protection: GSM8K has large delta (high task specificity) — parity-class failure unlikely

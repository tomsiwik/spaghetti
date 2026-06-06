# Peer Review: Delta Rule Interference Ordering

## NotebookLM Findings

Skipped -- NotebookLM authentication not available in this environment. The review proceeds with manual deep analysis of MATH.md, PAPER.md, and source code.

## Mathematical Soundness

### Delta rule formulation: CORRECT

The state update equations in MATH.md (lines 35-37) match the reference GatedDeltaNet implementation:

```
kv_mem_t = S_{t-1}^T @ k_t           -- retrieve
delta_t  = (v_t - kv_mem_t) * beta_t  -- correct
S_t      = g_t * S_{t-1} + k_t * delta_t^T  -- update
```

The code in `delta_rule_attention.py` (lines 128-140) faithfully implements this. Verified step by step:

1. **Decay applied before retrieval** (line 129): `S = g_t[:, :, None, None] * S`. This is correct -- the decay should be applied to the old state before using it for retrieval. The MATH.md notation (line 37) shows decay and update as a single step (`S_t = g_t * S_{t-1} + ...`), and the code correctly factors this into decay-then-retrieve-then-update. This matches the HF reference implementation's `torch_recurrent_gated_delta_rule`.

2. **Retrieval** (line 134): `kv_mem = (S * k_t[:, :, :, None]).sum(axis=-2)`. This computes S^T @ k via element-wise multiply and sum, producing shape (B, h, d). Correct.

3. **Correction** (line 137): `delta = (v_t - kv_mem) * beta_t[:, :, None]`. Matches the math. The `beta_t` is per-head scalar (not per-dimension as in production Qwen3.5), which PAPER.md acknowledges (line 237). This is a deliberate micro-scale simplification, not an error.

4. **State update** (line 140): `S = S + k_t[:, :, :, None] * delta[:, :, None, :]`. Outer product k * delta^T added to (already decayed) state. Correct.

### Decay gate parameterization: CORRECT

```python
g = mx.exp(-A * nn.softplus(a + self.dt_bias))  # line 108
```

This matches the real GatedDeltaNet formulation. `exp(-positive)` is in (0, 1). A = exp(A_log) ensures A > 0. softplus ensures the argument is positive. The result is a proper decay gate.

### Hidden assumptions identified

1. **Interference metric measures capsule divergence, not delta rule state dynamics** (acknowledged in MATH.md line 188-192, PAPER.md line 213-217). The cosine distance between capsule pool outputs is an indirect proxy. The experiment does not directly measure ||kv_mem_cross|| during composition. This is a limitation the paper openly states and proposes a fix for (instrumenting the forward pass). Fair.

2. **Layer 0 excluded from interference ratio** (run script lines 378, 388). The ratio uses only layers 1 and 2 vs layer 3. Layer 0 is excluded because it showed near-zero interference in prior experiments (shared base, minimal specialization). This is consistent with prior work and reasonable.

3. **The interference metric uses base_model for forward propagation through preceding layers** (lines 162-166 of run script). Both model_a and model_b's capsule outputs are computed from the same base-model activations. This is correct -- it isolates the capsule pool differences per layer without letting earlier-layer capsule differences compound. Good experimental design.

### One concern: the ratio metric conflates depth with mechanism

The interference ratio (linear layers 1,2 / full layer 3) compares different layers at different depths. Layers 1-2 see less-specialized representations than layer 3. The original hybrid attention experiment's PAPER.md acknowledged this "depth confound." This experiment does not re-address it, but since the question is specifically "does the delta rule reverse the ordering" (not "why is linear lower"), the depth confound is not blocking for the kill criterion evaluation. The directional comparison (delta rule vs simplified, both at the same layers) is valid.

## Novelty Assessment

### Prior art

The delta rule for gated linear attention is from Yang et al. 2024 (GatedDeltaNet). The production implementation is in Qwen3.5. This experiment does not claim novelty for the mechanism itself.

### What is novel

The specific question -- "does the delta rule's cross-domain retrieval mechanism break composition?" -- has not been addressed in the literature. Published GatedDeltaNet work evaluates single-model quality, not multi-expert composition. The experiment correctly identifies that the retrieval step `kv_mem = S^T @ k` creates a theoretical coupling between domains during composition that the simplified variant lacks. Testing this coupling is a valid and novel micro-experiment.

### Delta over existing work

The prior experiments in this project tested: (a) simplified gated linear recurrence (no delta rule), and (b) L2 QK normalization. This experiment adds the delta rule on top of both. The code properly extends the existing L2 norm model. No reinvention of existing components detected -- the code imports `l2norm` from the prior experiment and `CausalSelfAttention` from the hybrid attention experiment.

### References check

`references/REFERENCES.yml` does not have a dedicated GatedDeltaNet entry, though it references Qwen3.5 (which uses GatedDeltaNet) via `qwen3-coder-next` and `spring-2026-architectures`. The paper cites the right sources (Yang et al. 2024, Qwen3.5, HF Transformers reference impl). Adequate.

## Experimental Design

### Does it test the hypothesis? YES

The hypothesis is: "the delta rule reverses interference ordering (linear > full)." The experiment measures per-layer interference for delta rule linear layers vs the full attention layer, computes the ratio, and compares to a threshold of 1.0x. The simplified variant (L2 norm, no delta rule) runs as a within-experiment baseline. This is a clean A/B/control design.

### Controls adequate? YES

Three conditions tested: (1) full attention only (control), (2) L2 norm simplified linear (baseline), (3) delta rule linear (test). All three share the same protocol, hyperparameters, seeds, and data. The composition protocol is identical across conditions. The full_attn condition serves as both a composition reference and the denominator for the interference ratio.

### Could a simpler mechanism explain the result?

The result is that the delta rule does NOT reverse interference ordering (ratio 0.74x < 1.0x). Could this be explained trivially?

- **L2 normalization dominance**: The L2 norm bounds all QK products to [-1, 1], which bounds the retrieval magnitude. The delta rule's `kv_mem = S^T @ k` with ||k||=1 means kv_mem is bounded by the spectral norm of S, which is itself bounded by the accumulated (decayed) outer products of unit-norm vectors. This could explain why cross-domain retrieval doesn't cause trouble -- the corrections are numerically bounded. The paper acknowledges this (MATH.md line 174-176). This is a real explanation, not a confound.

- **Small state dimension**: At d_h=16, the state S is 16x16=256 entries. This is small enough that associations may wash out quickly under the decay gate, preventing strong cross-domain retrieval. The paper acknowledges this (PAPER.md line 197-199). Valid limitation, not a design flaw.

### Statistical power

7 seeds. The delta rule ratio is 0.74x with a threshold of 1.0x -- that is a 26 percentage point margin. The composition gap is +0.39% with a threshold of +10% -- a 9.6 percentage point margin. Both margins are large enough that even with high variance (and the paper reports std = 0.54% for delta rule gaps), the conclusions would not flip with more seeds. Adequate for a micro-scale directional test.

### One design weakness: joint baseline trains differently

The joint baseline trains for 600 steps alternating domains, while the domain-specific fine-tuning trains for 300 steps per domain. This means the joint baseline sees 300 steps of each domain, same total. However, the joint model trains from scratch with interleaved domains, while the composed model pretrains 300 steps on all data then fine-tunes 300 steps per domain. The total compute differs (joint: 600 steps; composed: 300 pretrain + 2*300 fine-tune = 900 steps). This is consistent with prior experiments in this project and the composition gap metric is relative (composed vs joint), so this is a systematic effect shared across all conditions. Not blocking.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry `exp_delta_rule_interference` (line 286) matches the experiment:
- Kill criteria match: (1) ratio >1.0x, (2) gap >+10% median
- Status: proven (both criteria pass)
- Depends on: `exp_hybrid_attention_composition` (correct -- this builds on the hybrid attention finding)
- Evidence claim matches the paper's results

There is a duplicate entry `exp_delta_rule_interference_proven` (line 932) that repeats the same information. This is redundant but not harmful -- likely an artifact of the status update process. Minor cleanup item.

## Integration Risk

The delta rule linear attention integrates cleanly with the existing architecture:
- Uses the same CapsulePool MLP (imported from capsule_moe)
- Same 3:1 linear:full layer pattern
- Same composition protocol (concatenate capsule groups, calibrate router)
- Properly imports and reuses L2 normalization from prior experiment

No conflicts with existing components detected. The model registers as `delta_rule_hybrid_capsule_moe` with parent `l2_norm_hybrid_capsule_moe`, maintaining the correct lineage.

## Macro-Scale Risks (advisory)

1. **Long sequences amplify state memory**: At T=4096+, the recurrent state accumulates far more associations than at T=32. The delta rule's retrieval mechanism queries a much richer state, potentially creating stronger cross-domain coupling. The micro result (0.74x) may not hold. This is the experiment's most important limitation and is clearly stated (PAPER.md lines 228-230).

2. **Per-dimension beta at macro scale**: The micro experiment uses per-head scalar beta. Production Qwen3.5 uses per-dimension beta. Per-dimension gating could selectively amplify corrections in specific dimensions, creating interference patterns not captured by the per-head scalar. The paper acknowledges this (PAPER.md line 237).

3. **Conv1d preprocessing**: Omitted at micro scale (T=32 < typical kernel size of 4). At macro scale, conv1d creates local mixing that could interact with the delta rule's retrieval. Likely neutral but untested.

4. **Sequential recurrence throughput**: 2.5x slower than full attention at micro scale. At macro scale, chunk-based implementation recovers this, but the chunk-based implementation itself should be validated for composition compatibility (mathematical equivalence is exact, but numerical precision at float16/bfloat16 could differ).

## Verdict

**PROCEED**

The experiment is well-designed, the math is correct, the implementation faithfully follows the GatedDeltaNet reference, and the results clearly pass both kill criteria with large margins. The hypothesis (delta rule reverses interference ordering) is falsified with 7 seeds, a valid A/B/control design, and appropriate statistical margins.

Minor items (non-blocking):

1. Clean up the duplicate HYPOTHESES.yml entry (`exp_delta_rule_interference` at line 286 vs `exp_delta_rule_interference_proven` at line 932).

2. Consider adding a direct measurement of cross-domain retrieval magnitude (||kv_mem_cross|| vs ||kv_mem_within||) as a follow-up diagnostic, as the paper itself suggests (PAPER.md lines 213-217). This would strengthen the mechanistic understanding but is not required to validate the composition-compatibility conclusion.

The experiment successfully de-risks the priority-1 adversarial concern about GatedDeltaNet composition. The macro architecture can proceed with hybrid attention using the delta rule mechanism.

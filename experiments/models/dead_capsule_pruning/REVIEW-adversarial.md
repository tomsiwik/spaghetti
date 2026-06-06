# Peer Review: Dead Capsule Pruning (Experiment 9)

**Reviewer**: Peer Review Agent (2nd pass)
**Date**: 2026-03-04
**Files reviewed**: `MATH.md`, `PAPER.md`, `dead_capsule_pruning.py`, `test_pruning.py`, `test_mechanism.py`, `relu_router.py`, `test_composition.py`, `VISION.md`, `FINDINGS.md`, `ADVERSARIAL_REVIEW.md`

---

## NotebookLM Findings

NotebookLM deep review was not executed due to tool availability constraints. This review is based on line-by-line manual analysis of all relevant source files, cross-referenced against the project's vision, findings, and adversarial review calibration document.

---

## Mathematical Soundness

### Theorem 3.1 (Exact Zero-Change): CORRECT, TRIVIALLY SO

The proof that removing a capsule i where `a_i^T x <= 0` for all x in D produces exactly zero output change is correct. The structure `y(x) = sum_j b_j * max(0, a_j^T x)` is additive across capsules, and ReLU gating means a capsule contributing zero activation contributes zero to the sum. Removing a zero-valued term from a sum leaves the sum unchanged.

This is not really a "theorem" -- it is a definitional property of ReLU. Calling it a theorem with a QED proof overstates the mathematical contribution. It would be more honest to call it a "property" or "observation." That said, it IS the correct formal justification for the lossless pruning claim, and it is important to state explicitly.

**The cross-layer extension is the non-trivial part, and it is handled correctly.** Section 3.3 and its caveat correctly note that for truly dead capsules (zero contribution), the inter-layer effect is also exactly zero. The proof structure is: dead capsule in layer l contributes zero to layer l's output, therefore layer l+1's input is unchanged, therefore layer l+1's activations are unchanged, and the argument cascades. This is sound.

### Theorem 3.2 (Bounded Error for Nearly-Dead): NOTATIONAL ISSUES, SUBSTANCE CORRECT

The stated bound:

```
E_x[||delta_y(x)||] <= epsilon * E_x[|a_i^T x| * ||b_i|| | a_i^T x > 0]
```

The derivation produces:

```
E[||delta_y||] = ||b_i|| * epsilon * E[a_i^T x | a_i^T x > 0]
```

Two notational issues:

1. **The "bound" is actually an equality.** The derivation computes the exact expected error, not an upper bound. The `<=` in the theorem statement is vacuously correct (an equality satisfies `<=`), but calling this a "bound" is imprecise.

2. **`||b_i||` placement inconsistency.** The theorem statement puts `||b_i||` inside the conditional expectation `E_x[|a_i^T x| * ||b_i|| | a_i^T x > 0]`, but `||b_i||` is a constant independent of x. The proof correctly factors it out. This is a cosmetic inconsistency.

**Neither issue affects the substance.** The result correctly shows that the expected output change from pruning a rarely-firing capsule is proportional to its firing frequency epsilon, weighted by the magnitude of its contribution when it does fire.

**However, this bound is never evaluated numerically.** The paper does not compute `||b_i||` or the conditional expectation for any actual capsule. For the recommended `tau=0` this is irrelevant (the error is exactly zero). For `tau > 0`, the bound's practical utility is unknown. This matters if aggressive pruning (tau=0.01 or higher) is promoted at macro scale.

### Section 3.3 (Independence of Pruning): CORRECT WITH IMPORTANT NUANCE

The within-layer independence is correct: `h_j = max(0, a_j^T x)` depends only on `(a_j, x)`.

**The nuance for tau > 0**: The paper notes the cross-layer caveat but does not clearly state that the independence guarantee is ONLY exact for `tau=0`. For `tau > 0`, pruning in layer l changes layer l's output (slightly), which changes layer l+1's input distribution, which could change which capsules appear dead in layer l+1. The current implementation profiles all layers in a single forward pass and then prunes all at once. It does NOT re-profile after pruning. For `tau=0` this is correct (pruned capsules contribute zero, so the cascade is exact). For `tau > 0`, there is a compounding approximation error that is not theoretically bounded.

The empirical results show this is negligible in practice (tau=0.01 adds only +0.1% degradation), but the theoretical gap should be acknowledged.

### Summary of Mathematical Assessment

| Result | Correctness | Significance |
|--------|-------------|--------------|
| Zero-change theorem (3.1) | Correct | Definitionally trivial but useful |
| Bounded error (3.2) | Correct (notation issues) | Never evaluated numerically |
| Independence (3.3) | Correct for tau=0, approximate for tau>0 | Cross-layer nuance underexplored |

---

## Novelty Assessment

### Prior Art Correctly Cited

- Dying ReLU phenomenon -- foundational, correctly referenced
- ReDo (Klein et al., 2024) -- closest prior work for detection mechanism
- MoE-Pruner (2024) -- expert-level pruning in MoE
- Lazy Neuron Phenomenon (Li et al., 2023) -- baseline sparsity rate

### Prior Art Missing or Underacknowledged

1. **Activation-frequency structured pruning in CNNs**: "Pruning Filters for Efficient ConvNets" (Li et al., 2017) uses activation-frequency-based criterion to prune entire convolutional filters, which is structurally identical to pruning entire capsule rows/columns based on activation frequency. The paper should cite this as the direct precedent for the pruning technique.

2. **Wanda (Sun et al., 2024)**: Post-training pruning for LLMs using `|weight| * ||activation||` as importance. Conceptually related: both use activation information to decide what to prune. Wanda is unstructured; this work is structured. Worth citing for context.

3. **SparseGPT (Frantar & Alistarh, 2023)**: Post-training pruning with inverse Hessian correction. More sophisticated than activation-frequency pruning. Worth citing as a more advanced alternative that could be applied if simple frequency-based pruning proves insufficient at scale.

4. **Lottery Ticket Hypothesis (Frankle & Carlin, 2019)**: The observation that large fractions of networks are removable is well-established. The 57% dead rate is consistent with lottery ticket findings.

### Delta Over Existing Work

The **technique** (profile activations, identify dead neurons, remove them structurally) is standard. The **application context** (post-composition of concatenated domain-specific MLPs) is novel. The **analytical findings** (57% dead rate in composed models, layer-dependent death rates, 92% dead-on-both-domains, order-independence with calibration) are genuine contributions.

This is a modest but real novelty: applying a known technique to a new context and producing useful empirical findings. For a micro-experiment, this is appropriate.

---

## Experimental Design

### Does It Test What It Claims?

**Yes.** The claim is that pruning dead capsules preserves quality while reducing parameters. The experiment directly tests this with:
- A threshold sweep (tau from 0.0 to 0.10)
- 3-seed evaluation with aggregation and standard deviations
- Multiple baselines (joint training, concatenation, weight averaging, calibration-only)
- Order-independence test (prune-then-calibrate vs calibrate-then-prune)
- Per-domain profiling analysis

The kill criteria are well-designed, falsifiable, and appropriately calibrated for micro scale:
- Prune(t=0) vs concat < 2% degradation
- Parameter reduction > 30%
- Prune-then-calibrate vs calibrate-only < 3% gap
- Dead ratio std across seeds < 15%

All four pass cleanly.

### Controls: Good, With One Significant Gap

**Present controls (all adequate)**:
- Joint training (upper bound)
- Unmerged concatenation (zero-shot baseline)
- Weight averaging (alternative compression)
- Calibration without pruning (isolates pruning's contribution)
- Both orderings of prune + calibrate

**Missing control 1 -- Single-domain dead capsule rate**: The most significant gap. The paper claims composition creates dead capsules, but the per-domain analysis shows 92% of dead capsules are dead on BOTH domains. This could mean either (a) capsules died during training and composition is not the cause, or (b) composition changes the hidden-state distribution so much that even "home domain" capsules stop firing in the composed context. Profiling the single-domain models before composition would disambiguate these hypotheses. This is the single most important missing experiment.

**Missing control 2 -- Random pruning baseline**: Pruning 57% of capsules at random and measuring degradation would demonstrate that targeted identification of dead capsules matters. Without this, one could argue that ANY 57% pruning of a 2x-overparameterized composed model might preserve quality, making the profiling step unnecessary. This control would close that gap.

### Profiling Dataset Size: Sufficient at Micro Scale

20 batches * 32 samples * 32 tokens = 20,480 token positions. A capsule with true activation probability p=0.001 has probability `(1-0.001)^20480 ~ 0` of appearing dead (false negative). A capsule with true probability p=0.0001 has probability `(1-0.0001)^20480 ~ 0.13` of appearing dead. At micro scale with vocabulary size 28, missing capsules that fire on < 0.01% of inputs is acceptable. At macro scale, this needs scaling.

### Profiling Implementation: Correct but Fragile

Lines 70-91 of `dead_capsule_pruning.py` manually reconstruct the model's forward pass:

```python
x = model.wte(inputs) + model.wpe(pos)
x = model.norm0(x)
for l_idx, layer in enumerate(model.layers):
    x_norm = layer.norm1(x)
    x = x + layer.attn(x_norm)
    x_norm = layer.norm2(x)
    pool = layer.capsule_pool
    h = nn.relu(pool.A(x_norm))  # (B, T, P)
    # ... count fires ...
    x = x + pool.B(h)
```

This duplicates the logic in `ReLUCapsuleBlock.__call__` and `ReLURouterGPT.__call__`. If the model's forward pass were ever modified (e.g., adding dropout, changing normalization order, adding bias), the profiling code would silently diverge, potentially identifying wrong capsules as dead. The test suite (`test_mechanism.py`) does not verify that the profiling forward pass matches the model's own forward pass.

For this experiment this is not a problem (the code is correct as-is), but it is a maintainability risk and should be flagged for macro implementation.

### The "Better Than Joint" Anomaly

Prune-then-calibrate achieves -1.1% vs joint training. This means it is *better* than the supposed upper bound. The paper presents this without adequate explanation. Possible causes:

1. **Joint training is undertrained**: 500 total steps (300 pretrain + 200 fine-tune) at micro scale may leave the joint model undertrained relative to the composed model, which benefits from domain-specific gradients during fine-tuning.

2. **Implicit regularization from pruning**: Reducing the model from 202K to 127K parameters may prevent overfitting on the small validation set.

3. **Within noise**: The joint baseline has std=0.0103. The -1.1% improvement corresponds to an absolute difference of 0.0055, which is 0.53 standard deviations. This is NOT statistically significant by any standard threshold (p > 0.3 for a one-sided test). The paper should not claim pruned+calibrated is "better" than joint -- it should say "comparable" or "within noise."

This is not a blocking issue, but the paper's framing ("better than joint") is misleading for a difference that is well within one standard deviation.

---

## Critical Analysis: The 92% Dead-on-Both-Domains Finding

This is the most interesting and underexplored finding in the paper. The paper correctly identifies it as surprising and updates its narrative, but does not resolve the causal ambiguity.

**Hypothesis A (training-induced death)**: Capsules die during fine-tuning due to the dying ReLU phenomenon. They are dead regardless of which domain's inputs are presented. Composition is irrelevant.

**Hypothesis B (composition-induced distribution shift)**: Composition changes the hidden-state distribution seen by each layer. In the single-domain model, attention processes only domain-A capsule outputs. In the composed model, attention processes domain-A + domain-B capsule outputs, producing a different hidden-state distribution. Capsules that were alive under the single-domain distribution become dead under the composed distribution.

**The data cannot distinguish these hypotheses without profiling single-domain models.** Under Hypothesis A, single-domain models would also show ~50% death. Under Hypothesis B, single-domain models would show ~10% death, and the 57% in composed models would be mostly composition-induced.

This matters for the project's narrative: if Hypothesis A is true, the compression opportunity exists in ALL ReLU models (not specific to composition), and the project should consider pruning as a general technique rather than a composition-specific one. If Hypothesis B is true, the composition protocol inherently creates pruning opportunities, which is a stronger finding for the VISION.md agenda.

---

## Integration Risk

### Fits the Composition Pipeline

The pruning step integrates cleanly into the validated protocol:

```
pretrain -> fine-tune/domain -> compose (concatenate) -> prune dead -> calibrate
```

No conflicts with existing components. The `DeadCapsulePruningGPT` class is a thin subclass of `ReLURouterGPT` (literally `pass`), which is architecturally clean.

### LoRA Transfer Claim is Overstated

PAPER.md's final paragraph claims: "ready for macro validation [...] with LoRA adapters where individual weight rows/columns of the LoRA A and B matrices can be profiled and pruned."

This is misleading. LoRA computes `delta_W = A @ B` with NO ReLU between A and B. The exact zero-change theorem depends fundamentally on ReLU's hard gating. In LoRA, a "dead" rank-1 component (small singular value) contributes a small but nonzero output, making pruning approximate rather than exact. The transfer path to LoRA requires a different theoretical framework (SVD-based rank reduction, not activation-frequency pruning).

If the macro architecture uses ReLU capsule pools (as in the ReMoE / ReLU Router lineage), the transfer is clean. If it uses standard LoRA, the transfer needs reformulation.

### Non-ReLU Risk at Scale

Most production LLMs use GELU or SiLU. The exact pruning guarantee does not transfer. However, recent work (ReluLLM, ReMoE at ICLR 2025) shows ReLU is viable for MoE sparsity. If the macro architecture commits to ReLU (consistent with the project's ReLU Router lineage), this risk is mitigated.

---

## Macro-Scale Risks (advisory, not blocking)

1. **Non-ReLU activations**: GELU/SiLU have no hard zero, making "dead" a matter of degree. Approximate pruning with magnitude threshold needed. The Section 3.2 error bound becomes the primary tool but has never been numerically evaluated.

2. **Rare-token sensitivity**: At macro scale with BPE and 50K+ vocabulary, some capsules may fire only for rare but important tokens. Character-level micro data has no rare tokens. Profiling dataset must include sufficient coverage of the tail.

3. **Dead rate at larger d**: With d=4096 instead of d=64, random vectors are nearly orthogonal. The dead capsule rate may increase (harder for mismatched detectors to fire) or decrease (more dimensions = more opportunities for partial activation). Empirical validation required.

4. **Layer 0 exception scaling**: If more layers behave like layer 0 at macro scale (generic representations, few dead capsules), the compression ratio drops from 37% toward zero.

5. **Distribution shift**: Static pruning assumes the deployment distribution matches calibration. The paper proposes no mitigation for distribution shift (e.g., periodic re-profiling, keeping pruned weights in cold storage).

---

## Verdict

**PROCEED**

### Justification

This is a well-executed micro-experiment that validates a simple, principled compression mechanism for composed ReLU models. The core claim -- that pruning capsules with zero activation frequency produces exactly zero quality change -- is mathematically trivial but practically useful. The empirical results are consistent across 3 seeds, the controls are adequate, and the kill criteria pass cleanly.

The 57% dead capsule rate and 37% total parameter reduction at zero quality cost are real and useful findings. The order-independence with calibration is a clean result. The per-domain analysis revealing that most death is training-induced (not domain-induced) is a genuine insight, even if incompletely resolved.

### Issues Found (None Blocking)

1. **Missing single-domain dead rate control**: Cannot disambiguate training-induced vs composition-induced death without profiling pre-composition models. The 92% dead-on-both-domains finding is potentially explained by composition-induced distribution shift, not training-induced death. This is informative for understanding the mechanism but does not affect the pruning result.

2. **"Better than joint" claim is within noise**: The -1.1% advantage (0.5184 vs 0.5239) is 0.53 standard deviations of the joint baseline. Not statistically significant. Should be reported as "comparable" rather than "better."

3. **LoRA transfer claim is misleading**: LoRA has no ReLU gating. The exact pruning theorem does not apply. If macro uses ReLU capsule pools, the transfer is clean. If it uses standard LoRA, the theoretical framework needs reformulation.

4. **Section 3.2 bound never evaluated numerically**: For tau > 0 at macro scale, compute actual `||b_i||` and conditional expectations to verify the bound is practically useful.

5. **Missing random-pruning baseline**: Would strengthen the case that targeted dead-capsule identification matters vs arbitrary pruning at the same rate.

6. **Profiling code duplicates forward pass logic**: Fragile to future model changes. Should either (a) instrument the model's own forward pass with hooks, or (b) add a test verifying profiling matches model forward pass.

### Advisory for Macro Validation

- Measure single-domain death rates to resolve the causal question
- Scale profiling dataset proportionally to vocabulary size
- Evaluate Section 3.2 error bound numerically for any tau > 0
- Test with non-ReLU activations if macro architecture uses GELU/SiLU
- Report Jaccard similarity of dead capsule sets before/after calibration to strengthen order-independence claim

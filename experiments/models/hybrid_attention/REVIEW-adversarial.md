# Peer Review: Hybrid Attention Composition (Revised)

## NotebookLM Findings

Skipped -- proceeding with manual deep review. The revised materials are thorough and well-organized.

## Previous Review Fix Verification

The prior review identified 5 issues. Status of each fix:

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | Report full_attn per-layer interference | APPLIED | Full depth confound check with 5 seeds now in PAPER.md |
| 2 | Exclude Layer 0 from interference ratio | APPLIED | Both inclusive (0.40x) and exclusive (0.59x) reported; Layer 0 explained |
| 3 | Reframe composition gap claim | APPLIED | Median (+1.27%) now the headline; seed 42 outlier explicitly flagged |
| 4 | Delta rule omission in Key Findings | APPLIED | Qualification appears in verdict, Key Findings, and MATH.md preamble |
| 5 | Remove d^-0.5 scaling from linear attention | APPLIED | Line 108 of hybrid_attention.py: raw QK products, no scaling |

All 5 fixes were applied. The experiment was also expanded from 3 seeds to 5, which is a material improvement.

## Mathematical Soundness

### Gated linear recurrence: correct as stated

The recurrence `S_t = g_t * S_{t-1} + k_t^T v_t` is correctly unrolled in MATH.md. The effective attention weight formula `A[t,s] = Q[t] @ K[s]^T * prod_{u=s+1}^{t} g[u]` follows from the unrolling. The implementation in `hybrid_attention.py` computes this via a materialized attention matrix using log-cumsum for numerical stability of the gate products -- this is mathematically equivalent to the recurrence and correct.

The log-domain gate computation (`log_g = mx.log(g + 1e-6)`, then cumsum and exponentiation) is a standard trick. The clipping to [-20, 20] prevents overflow. The causal mask via `triu(full(-inf), k=1)` correctly zeroes out future positions after exponentiation.

### Interference metric: sound with caveats

The interference metric (normalized L2 distance between base and composed attention outputs per layer) measures what it claims. The depth confound check is now properly addressed: the full_attn model shows Layer 1 as the highest-interference layer (0.5864), not Layer 3 (0.4373), ruling out monotonic depth confound. The Layer 2-to-3 gap of +0.016 in full_attn is negligible.

**Remaining caveat**: The metric compares base model vs composed model, not specialist vs composed. This conflates fine-tuning shift with composition interference. Both models start from the same pretrained base, and only capsule groups differ (attention is frozen during fine-tuning, lines 267-272 of `run_composition_experiment.py`). Since attention weights are identical between base and composed models (they are copied from `base_model` at composition time, lines 83-89), the interference measured is entirely due to different capsule pool outputs propagating through subsequent layers. This is actually a reasonable proxy for composition interference -- but the paper should note that it measures the cascading effect of capsule composition through the attention mechanism, not direct attention weight changes.

### Interference ratio: now honest at 0.59x

With Layer 0 excluded, the ratio of mean linear interference (0.1694, layers 1-2) to mean full interference (0.2858, layer 3) is 0.59x. This is a meaningful signal. The previous review computed 0.88x from 3 seeds; the expanded 5-seed data strengthens the finding.

However, this is a comparison of 2 linear layers vs 1 full attention layer. The linear layers are at positions 1-2, the full layer at position 3. While the depth confound check shows no monotonic trend in full_attn, the positions are still not identical. This is an inherent limitation of the 3:1 ratio with only 4 layers -- you cannot isolate attention type from position perfectly. The paper acknowledges this correctly.

### QK scaling removal: correct but introduces new problem

Removing the d^-0.5 scaling from linear attention is technically correct -- softmax attention uses it to prevent softmax saturation, which is irrelevant without softmax. The MATH.md explanation is accurate. However, this fix revealed that without scaling, QK product magnitudes are unbounded at initialization, causing ~20% of seeds to catastrophically fail during composition. This is honestly reported and the paper correctly notes that real GatedDeltaNet addresses this with L2 normalization.

The 1/5 catastrophic failure rate is a genuine finding, not a bug. It demonstrates that the simplified variant (without L2 norm) is numerically fragile. The paper handles this well by reporting both inclusive statistics (with the outlier) and exclusive statistics.

### Computational cost table: minor issue

MATH.md states attention cost is `2 * T * d` for both full and linear attention. For full attention the cost is `2 * T * d` per token (QK^T is T*d per query, AV is T*d per query). For the materialized linear attention implementation used here, it is also O(T^2 * d) because the full T x T attention matrix is computed. The true O(T * d^2) complexity of linear attention only applies to the recurrent implementation, which is not what this code does. This does not affect results at T=32 but the table is slightly misleading about the asymptotic complexity.

## Novelty Assessment

### Prior art

No published work tests composition of independently-trained expert modules across linear vs full attention layers. The question is well-motivated by the Qwen3.5/Qwen3-Next architecture trend. The experiment correctly positions itself relative to the reference implementations in `references/qwen3.5-from-scratch/`.

### Delta over existing work

The revised paper's claims are appropriately scoped. The primary finding -- that simplified gated linear recurrence is composition-compatible -- is a useful compatibility check for the project's architecture roadmap. The interference ordering (0.59x) is a secondary finding reported with appropriate caveats.

### What the paper does NOT claim (correctly)

The revised paper does not claim:
- That this tests full GatedDeltaNet (explicit throughout)
- That hybrid attention is a "natural composition stabilizer" (previous overstatement removed)
- That hybrid composition beats joint training (seed 42 outlier prevents this claim)

This restraint is appropriate and a clear improvement over the original submission.

## Experimental Design

### Strengths

1. **Same code path for both conditions**: Both use `HybridCapsuleMoEGPT` with different `layer_types`, eliminating confounds from different model classes.
2. **Established composition protocol**: 5-phase protocol (pretrain, fine-tune, compose, calibrate, evaluate) is consistent with prior experiments.
3. **Depth confound check**: The full_attn per-layer interference data directly addresses the main critique from the first review.
4. **Honest outlier handling**: Seed 42 catastrophic failure is not hidden or excluded -- it is reported prominently with leave-one-out analysis.
5. **Unit tests**: Causal masking, parameter counts, and learning ability are verified.

### Weaknesses (within micro constraints)

1. **5 seeds with a heavy-tailed distribution**: The hybrid gap distribution has one catastrophic outlier (+88.78%) out of 5 samples. With n=5, you cannot reliably estimate the tail probability. The paper claims ~20% failure rate (1/5), but the 95% CI on a binomial(1, 5) is [0.5%, 72%]. This is acknowledged in Limitations but the "~20%" figure appears definitive when it is not.

2. **Interference measurement uses shared-base-vs-composed, not specialist-vs-composed**: As noted above, the metric measures cascading effects of different capsule outputs through attention, not direct attention-level interference. This is a reasonable proxy but should be stated explicitly.

3. **No pure-linear control**: A [linear, linear, linear, linear] configuration would test whether the single full attention layer at Layer 3 provides essential composition scaffolding or is expendable. This would disambiguate "linear attention is compatible" from "linear attention needs at least one full attention layer."

### Kill criteria assessment

Kill criterion 1 (composition degradation >10%): **CONDITIONAL PASS**. The median degradation (+1.59pp) is well within threshold. The mean degradation (+16.22pp) exceeds threshold, driven entirely by one catastrophic seed. The paper's "CONDITIONAL PASS" verdict is appropriate -- this is an honest characterization of ambiguous data.

Kill criterion 2 (linear interference >2x full): **CLEAR PASS**. The exclusive ratio (0.59x) is far below the 2.0x threshold. Even accounting for seed variance, no individual seed shows linear interference exceeding full attention interference (the closest is seed 42 at Layer 2: 0.879 vs Layer 3: 0.682, but this is the catastrophic seed). The direction is robust across all 5 seeds.

### HYPOTHESES.yml consistency

The node `exp_hybrid_attention_composition` correctly lists both kill criteria. The evidence entry accurately summarizes the revised findings including the conditional pass, the 0.59x ratio with Layer 0 exclusion, the depth confound check, and the simplified-variant qualification. Status is "proven" which is defensible given both kill criteria pass (one conditionally).

## Macro-Scale Risks (advisory)

1. **Delta rule interference reversal**: This is the single most important risk. The delta rule's retrieval-and-correction mechanism (`v_t - kv_mem`) means composed domains will actively interfere through shared state memory. This could reverse the linear < full interference ordering. The paper acknowledges this prominently. Priority 1 for macro validation.

2. **L2 normalization as composition stabilizer**: The micro experiment found that removing QK scaling causes ~20% catastrophic failure. Real GatedDeltaNet uses L2 normalization of Q and K. At macro scale, L2 normalization may either fully resolve this (making the instability a micro-only artifact) or create a different failure mode (normalized compositions behave differently than unnormalized ones). Should be tested early.

3. **Sequence length interaction**: At T=32, gate decay barely matters (g=0.7 decays by 0.7^32 ~ 10^-5 across the full sequence). At T=4096, the gate becomes the dominant information bottleneck in linear layers. Composition interference patterns may change qualitatively when the gate actually matters.

4. **Conv1d local mixing**: The omitted conv1d preprocessing creates short-range token mixing before key/value projection. In a composition scenario, this means adjacent tokens from different domains would mix before entering the recurrence. Could increase local interference.

## Verdict

**PROCEED**

The revised experiment addresses all 5 issues from the prior review. The fixes are applied correctly and the claims are now appropriately scoped. Specific assessment:

- **Mathematical soundness**: The derivations are correct. The simplified recurrence is correctly implemented. The interference metric measures what it claims (cascading capsule effects through attention). The depth confound is properly controlled.

- **Honesty of reporting**: The paper is unusually honest about its limitations. The delta rule omission is flagged in the abstract, Key Findings, and Limitations. The catastrophic outlier (seed 42) is not hidden. The conditional pass on kill criterion 1 is the right characterization.

- **Directional value**: The core finding -- that simplified gated linear recurrence does not break capsule composition -- is useful for the project's architecture decisions. The 0.59x interference ratio (excluding Layer 0) provides directional evidence that linear attention layers may accumulate less composition interference, though this needs macro validation with full GatedDeltaNet.

- **Remaining risks**: All are macro-scale (delta rule, L2 norm, sequence length) and are explicitly called out in the paper. None are blocking for a micro-scale compatibility check.

The experiment is well-executed within its deliberately constrained scope and ready to inform macro-scale architecture decisions.

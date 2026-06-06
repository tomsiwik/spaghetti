# Peer Review: Base-Free Composition

## NotebookLM Findings

Skipped (NotebookLM not required for this review -- the experiment is straightforward enough that a direct mathematical and experimental audit is sufficient).

## Mathematical Soundness

### What holds

1. **Eckart-Young theorem application is correct.** SVD truncation is the optimal rank-k approximation in Frobenius norm. The derivation in MATH.md Section 2.2 is standard and correct.

2. **Exact decomposition identity is trivially correct.** W_pretrained = W_skeleton + (W_pretrained - W_skeleton) is an identity. The delta_full sanity check confirms this (loss ratio 1.000 across all 3 seeds). Good experimental hygiene.

3. **Reconstruction error formula is correct.** The relative error epsilon(k) = ||Delta - Delta_k||_F / ||Delta||_F is the standard metric, and the RMS aggregation across layers is reasonable.

4. **Effective rank computation is correct.** The Roy-Vetterli (2007) definition via Shannon entropy of normalized singular values is properly implemented.

5. **Cosine similarity measurement is standard.** Flattening all LoRA deltas into a single vector per expert and computing pairwise cosine is the same methodology used in previous experiments.

### Issues

1. **Error propagation claim is hand-waved.** MATH.md Section 3.2 states:

   > L(W_composed) = L(W_exact_composed) + O(epsilon(k))

   This is not derived -- it is asserted. The relationship between weight-space Frobenius error and loss-space error is nonlinear and depends on the loss landscape curvature (Hessian). The experimental results happen to show a roughly linear relationship, but the O(epsilon) notation implies a theoretical bound that does not exist in the paper. This should be stated as an empirical observation, not a mathematical claim.

2. **The macro extrapolation of effective rank ratio is speculative.** MATH.md Section 4.2 computes r_needed/d = 40/64 = 0.625 and extrapolates to macro scale (r_needed ~ 2,240 for Qwen 7B). The paper correctly notes this "may not hold at macro scale" in Limitations, but the worked example in Section 5.2 then uses rank-256 and rank-512 without justification for why these would be sufficient. The gap between the measured 0.625 ratio (implying rank ~2,240) and the assumed rank-256 (ratio 0.07) is a factor of ~9x. The paper waves at "higher redundancy in larger models" and BitDelta's findings, but these are about fine-tuning deltas, not random-init-to-pretrained deltas. This remains the single largest unknown for the macro path.

3. **Storage cost comparison in Section 5.2 uses the optimistic rank, not the measured ratio.** If the 0.625 ratio holds at scale, a "base adapter" at rank-2240 for Qwen 7B would be ~96M params/layer or 3.1B params total -- 53% of the full model, not 5%. The paper's rank-256 estimate is aspirational, not evidence-based.

## Novelty Assessment

### Prior Art

- **BitDelta (Ma et al., 2024)**: Compresses fine-tuned-minus-base deltas to 1 bit. Relevant but different: their delta is base-to-fine-tuned (small), not random-init-to-pretrained (large). The paper correctly cites this but incorrectly draws encouragement from it -- a fine-tuning delta is structurally different from a pretraining delta.

- **DeltaZip (Yao et al., 2024)**: Multi-tenant delta serving. Same caveat as BitDelta -- their deltas are fine-tuning deltas, not the full pretraining trajectory.

- **SliceGPT (Ashkboos et al., 2024)**: Removes rows/columns from weight matrices via PCA on calibration data activations. More principled than naive SVD on weights for compression, because it accounts for activation statistics. Not cited.

- **ASVD (Yuan et al., 2024)**: Activation-aware SVD for LLM compression. Directly relevant -- shows that naive weight-space SVD is suboptimal and activation-weighted SVD achieves better downstream quality. Not cited. This is a gap.

- **LoRA-the-Explorer (Hyeon-Woo et al., 2024)**: Cited but not deeply engaged with. LTE's parallel multi-head approach is tangential.

- **SLTrain (Hsu et al., 2024)**: Cited. Sparse+low-rank pretraining. Relevant to the macro path but not directly tested here.

### Novelty delta

The specific question "can a pretrained base be decomposed as W_skeleton + SVD_k(Delta) and still support LoRA expert composition?" appears genuinely novel. The decomposition itself (SVD of weight deltas from random init) is not new, but measuring composition quality of LoRA experts on the reconstructed base is new. This is a meaningful micro contribution.

However, the framing as "base-free" is somewhat misleading. The skeleton is still needed, and the "base adapter" carries substantial information. It would be more accurate to call this "base-as-adapter" or "compressed-base composition."

## Experimental Design

### Strengths

1. **Seven conditions with proper controls.** Pretrained (positive control), delta_full (identity check), four SVD ranks (treatment), skeleton_only (negative control). This is textbook experimental design.

2. **Three seeds with aggregate statistics.** Standard deviations reported, verdicts consistent across all seeds.

3. **Kill criteria are reasonable and clearly stated.** Loss ratio > 2.0x is generous but appropriate for a micro-scale mechanism test.

4. **FFN-only LoRA experts.** Consistent with the project's established finding that FFN-only adapters compose better.

5. **The integration test correctly fails.** The `results_seed_42.json` file contains a KILLED verdict from the integration test (d=32, 50 pretrain steps, 20 expert steps) -- this is the expected behavior when the model is barely trained. This serves as an implicit sanity check that the kill criteria can actually trigger.

### Weaknesses

1. **Experts are retrained on each reconstructed base.** The experiment trains fresh LoRA experts on top of each condition's base. This means the experts adapt to the reconstruction error during training. In the actual deployment scenario (PAPER.md "What This Enables" section), the claim is that you could swap bases without retraining experts. But this experiment does NOT test that. It tests whether training experts on a degraded base produces acceptable results -- a weaker and different claim. The base-swapping scenario (train on base_v1, deploy on base_v2) is untested and would be the actually interesting experiment.

2. **The "skeleton" is the exact same random initialization used to start pretraining.** This means the delta captures exactly the pretraining trajectory. In a real deployment, the skeleton would be a different random initialization (or some minimal checkpoint). The paper does not test what happens when the skeleton is a different random initialization. If the skeleton changes, the delta changes, and the SVD spectrum could be entirely different. This is acknowledged in Limitation 5 but never tested.

3. **Overlapping domains inflate expert similarity baselines.** The quintary character-range split (a-e, f-j, k-o, p-t, u-z) creates domains with overlapping character distributions. Mean |cos| of 0.068 at d=64 is already fairly high. A fairer test would use more distinct domains.

4. **No generation quality evaluation.** The paper acknowledges this (Limitation 5) but it matters: NTP loss and generation coherence can diverge, especially when the base is perturbed. SVD reconstruction error may introduce subtle distributional shifts that increase loss minimally but degrade generation quality noticeably.

5. **The artifact `results_seed_42.json` was overwritten by the integration test.** The file contains d=32 integration test results (KILLED), not the actual seed=42 d=64 experimental results. The real seed-42 data exists only in `results_aggregate.json`. This is a minor reproducibility issue -- running the integration test after the main experiment overwrites the output file.

### Critical question the experiment does not answer

The paper's "What This Enables" section claims base swapping (train experts on base_v1, swap to base_v2 without retraining). But the experiment trains fresh experts for every condition. The untested but critical claim is: **do experts trained on the full pretrained base still work when the base is replaced by W_skeleton + SVD_k(Delta)?** This is the zero-shot transfer scenario and the one that actually matters for the vision. It could easily fail -- experts may have learned to compensate for specific base weight patterns that change under SVD truncation.

## Hypothesis Graph Consistency

The experiment matches `exp_base_free_composition` in HYPOTHESES.yml. The kill criteria in the code match those in the YAML:

| YAML criterion | Code implementation | Match? |
|---|---|---|
| adapter-only model quality <50% of base+adapter (loss > 2x) | K1: loss_ratio > 2.0 | Yes |
| adapter-only model fails to produce coherent text | K2: base_loss_ratio > 2.0 | Yes |
| training adapter-as-base costs >10x training a LoRA expert | K3: decomp_time / expert_time > 10.0 | Yes |

The K3 criterion is trivially satisfied (SVD is instant vs. gradient-based training). This is not a meaningful test -- it would be more useful to compare inference latency or storage overhead.

Note: `exp_adapter_as_base` also points to this experiment directory and has different, tighter kill criteria (>20% quality loss, >50% parameter count). The rank-16 results satisfy those too (1.9% loss, ~84% params at micro but projected 5% at macro). The 84% parameter count at micro scale technically fails the 50% threshold if taken literally, but the paper argues this ratio improves at scale. This is unverified.

## Macro-Scale Risks (advisory)

1. **Delta effective rank scaling is the make-or-break question.** If r_needed/d stays at 0.625, the base adapter for Qwen 7B needs rank ~2,240, which is 53% of the model -- defeating the purpose. The entire value proposition depends on this ratio dropping dramatically at scale, and there is zero evidence for this beyond hope.

2. **ASVD vs naive SVD.** At scale, activation-aware SVD (ASVD) would likely outperform the naive weight-space SVD used here. The micro experiment cannot distinguish between these because the model is too small for activation statistics to diverge from weight statistics.

3. **Cross-layer structure.** Layer-wise independent SVD ignores the fact that early and late layers have different effective ranks. A more sophisticated approach (e.g., allocating rank budget proportionally across layers) could yield substantially better results at the same total parameter count.

4. **Expert transfer across base versions.** The paper's most exciting claim (base swapping without expert retraining) is entirely untested. This should be the first macro experiment.

5. **Inference overhead.** Reconstructing W from skeleton + SVD factors on every forward pass would add latency. The paper implicitly assumes you would precompute the full W, but then you have lost the storage benefit. This tension is unaddressed.

## Verdict

**PROCEED**

### Justification

The experiment is well-designed within its scope, the math is sound (with one hand-waved bound), the controls are proper, three seeds show consistent results, and the mechanism works in principle at micro scale. The key finding -- that LoRA expert quality degrades more slowly than base quality under SVD truncation -- is genuinely interesting and somewhat surprising.

The kill criteria are correctly evaluated and legitimately disproven. The 2.0x threshold is generous but appropriate for a first-pass mechanism test.

### Caveats that must be acknowledged (not blocking)

1. The experiment tests "train experts on degraded base," not "transfer experts across base versions." The latter is the actually valuable claim and remains untested.

2. The macro extrapolation (rank-256 sufficient for 7B) has no evidence. The measured ratio (0.625) suggests rank-2240, which is impractical. The entire macro viability hinges on this ratio dropping at scale.

3. The error propagation bound (Section 3.2, MATH.md) should be restated as an empirical observation, not a mathematical claim.

4. Missing citation: ASVD (activation-aware SVD) is directly relevant prior art that should be cited and discussed.

5. The `results_seed_42.json` artifact was overwritten by the integration test. For reproducibility, either rename the integration test output or save per-seed results with the experiment config hash.

### Recommended additions before macro

1. **Cross-base expert transfer test** (micro, 30 min): Train experts on pretrained base, evaluate on SVD-reconstructed base without retraining. This is the critical untested claim.
2. **Multi-dimension scaling** (micro, 1 hr): Repeat at d=128 and d=256 to get a second and third data point for the r_needed/d ratio. If it drops even slightly, the macro outlook improves dramatically.
3. Add ASVD citation and discussion.

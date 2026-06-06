# Peer Review: Zero-Shot Base Transfer

## NotebookLM Findings

Skipped -- manual deep review conducted instead. The experiment is straightforward enough that the mathematical and code audit below is sufficient.

## Mathematical Soundness

### Error Decomposition (MATH.md Section 2.2) -- CORRECT

The core derivation is sound:

    E_zs = (W_s + Delta_k + dW_i^{full}) @ x - (W_p + dW_i^{full}) @ x
         = (Delta_k - Delta) @ x
         = -E_delta(k) @ x

The expert delta dW_i cancels exactly because the same A, B matrices are used in both the zero-shot and reference conditions. This is a clean, non-trivial observation and the derivation is correct step-by-step.

### Comparison to Retrained Baseline (Section 2.2) -- CORRECT WITH CAVEAT

The retrained comparison is:

    E_retrained = -E_delta(k) @ x + (dW_i^{(k)} - dW_i^{(full)}) @ x

The second term represents expert adaptation to the perturbed base. The paper correctly identifies this as the "compensation effect" and provides empirical evidence that retrained experts partially compensate (loss degrades slower than base) while zero-shot experts amplify (loss degrades faster than base). The math is correct.

**Caveat**: The retrained loss ratios cited in the comparison table (1.001, 1.014, 1.050, 1.095) come from the parent experiment (base_free_composition), not from re-running retrained experts within this experiment. This is acceptable -- the parent experiment used the same architecture, data, and seeds -- but it means the comparison is across two separate experiment runs, not a within-run control. The paper should be more explicit about this.

### Amplification Bound (Section 3.4) -- WEAK

The claimed bound `L_expert_zs / L_base ~ 1 + c * epsilon(k)` is fit from only 4 data points with c ranging from 0.005 to 0.099. This is not a bound; it is a post-hoc curve fit. The "plateau" observation (c stabilizing at ~0.099 for the two highest epsilon values) is based on exactly two data points.

More problematic: the relationship between loss ratio and epsilon is likely nonlinear (loss is exponential in cross-entropy terms), so a linear bound in epsilon is not theoretically motivated. The paper hedges this with "approximately," but the section title says "Bound" which overstates the rigor.

**Not a kill issue** -- the observation is directional and honestly presented in the data tables. The overstatement is in framing, not substance.

### Kill Criterion K2 (Cosine > 5x) -- VACUOUS

The paper correctly notes that K2 is "SATISFIED BY DEFINITION" because the same LoRA deltas are used across conditions. This means K2 was never actually at risk in this experiment -- it tests nothing. The kill criterion was designed for the parent experiment (where experts are retrained per condition) and was mechanically carried over without adaptation.

This is not a mathematical error, but it means the experiment effectively has only 2 meaningful kill criteria, not 3. The paper should not count K2 as evidence of robustness.

### Code Verification -- CORRECT

I verified the implementation in `zero_shot_base_transfer.py`:

1. `get_delta()` returns `(A @ B * scale)` with shape `(in, out)`.
2. `apply_lora_deltas_to_base()` adds `delta.T` to `linear.weight` (shape `(out, in)`).
3. This is mathematically equivalent to the LoRA forward pass: `base(x) + x @ A @ B * scale = (W + delta.T) @ x`.

The implementation correctly applies the same expert delta to a different base by direct weight addition. No bugs found.

### Aggregate Statistics -- VERIFIED

I spot-checked the aggregate JSON against the per-seed results:

- Seed 42 reference mean_expert_loss: 0.43330 (matches JSON)
- Seed 123 reference mean_expert_loss: 0.42043 (matches JSON)
- Seed 7 reference mean_expert_loss: 0.42915 (matches JSON)
- Aggregate reference: 0.42762 = mean(0.43330, 0.42043, 0.42915) = 0.42763. Close enough (rounding).

The aggregate loss ratios reported in PAPER.md match the JSON data within rounding precision.

## Novelty Assessment

### Prior Art

The zero-shot LoRA transfer question has been explored in adjacent contexts:

1. **LoRAHub (Li et al. 2022)** -- Composes multiple LoRA adapters via gradient-free optimization, but assumes a fixed base model. Does not address base model changes.

2. **BitDelta (2024)** -- Shows that fine-tuning deltas are highly compressible. Relevant to the compression side but does not test transferring adapters across different base model variants.

3. **Model soups / weight averaging literature** -- Merges models fine-tuned from the same base. The implicit assumption is shared base; transferring to a different base is not tested.

4. **LoRA composability papers (InFoLoRA, MoE-Adapters4CL)** -- Assume frozen, fixed base model. No base swapping.

The specific question -- "do LoRA deltas trained on base W_1 work when applied to a systematically perturbed base W_2?" -- does not appear to have direct prior art. The SVD perturbation protocol is a clean way to control the perturbation magnitude, which is a modest methodological contribution.

### Delta Over Existing Work

The novel contribution is the **error amplification asymmetry**: retrained experts compensate for base error (ratio < 1) while zero-shot experts amplify it (ratio > 1). This is a useful empirical observation for the Living Composable Model architecture, though the amplification "bound" is overstated (see above).

## Experimental Design

### Strengths

1. **Clean isolation of the transfer question.** The parent experiment conflated "SVD base supports composition" with "experts transfer across bases." This experiment properly isolates the second claim by training experts once and evaluating on multiple bases.

2. **Good controls.** Pretrained (positive), delta_full (identity check), skeleton_only (negative control). The identity check (delta_full matches pretrained) validates the decomposition is exact.

3. **Multi-seed replication.** 3 seeds with consistent results across all conditions. No cherry-picking.

4. **Monotonic degradation.** Expert quality degrades monotonically with SVD rank, as theory predicts. No anomalous reversals.

### Weaknesses

1. **The comparison to retrained experts is cross-experiment, not within-experiment.** The retrained loss ratios are taken from the parent experiment's PAPER.md. While the architecture and data are the same, this means differences in random initialization (even with same seed) could introduce noise. A stronger design would include retrained experts as a within-experiment control.

2. **Expert evaluation uses the expert's own domain validation set, not a shared evaluation set.** Each expert is evaluated on its domain val_ds. This is appropriate for measuring domain-specific quality, but means the "mean expert loss" averages over different data distributions. This is not wrong, but it limits interpretability.

3. **The amplification claim could be a simpler phenomenon.** When an expert is trained on base W, it learns a correction dW that is optimized for the residual pattern of W. On a different base W', the residuals change, and dW is suboptimal. The "amplification" (expert degrades faster than base) could simply be because the expert correction is now partially misaligned with the new residual pattern -- some components of dW may actively harm rather than help. This is a simpler explanation than "error amplification" and does not require a separate mechanism. The paper's Section 3.3 gestures at this but could be more precise.

4. **No statistical significance testing on the transfer gap.** The gap of 0.2% at rank-32 is very small. With only 3 seeds, it is unclear whether this is distinguishable from noise. The paper should report confidence intervals or at minimum the per-seed spread for the transfer gap.

### Confound: Same Skeleton

Both the training base and the transfer base share the same random skeleton (W_s). In the real base-swapping use case (e.g., Qwen2.5 to Qwen3), the models would have different initializations. The paper acknowledges this in Limitations (item 2) but does not explore its impact. This is a known limitation, not a kill issue, but it means the "zero-shot transfer works" claim is conditional on shared initialization structure.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml entry:

- **Kill criteria match**: K1 (loss > 2x) and K3 (>50% fail) are genuinely tested and disproven.
- **K2 (cos > 5x)** is vacuous for this experiment (identical deltas), as discussed above.
- **Status "proven" is appropriate** given that all meaningful kill criteria are disproven with margin across 3 seeds.
- **Depends on exp_base_free_composition** -- correctly ordered, parent experiment provided the retrained baseline.
- **Evidence entries in HYPOTHESES.yml** accurately reflect the aggregate results.

## Integration with VISION.md

The experiment directly supports the "Base Swapping" use case in Phase 3 (Evolve) and the research question listed at the bottom of VISION.md. It confirms that when a base model upgrade can be expressed as a low-rank delta, existing experts can be reused without retraining. This is architecturally important -- it means the expert library survives base model upgrades, which is essential for a "living" system.

However, the paper's practical recommendation table (Section 5.1 of MATH.md) may be too optimistic. The "use zero-shot freely" recommendation for rank-32 is based on d=64 where rank-32 is half the full dimension. At d=4096 (Qwen 7B), rank-32 would be a much more aggressive truncation (0.8% of full rank), and the transfer gap could be very different.

## Macro-Scale Risks (advisory)

1. **Real base model upgrades are not SVD-structured.** Qwen2.5 to Qwen3 involves architectural changes, different pretraining data, potentially different tokenizers. The delta between such models is unlikely to be well-approximated by low-rank SVD. The "rank-16 delta between versions" assumption in Section 5.2 is speculative.

2. **Scale direction is ambiguous.** The paper argues that zero-shot transfer would work BETTER at scale because "LoRA deltas are proportionally smaller relative to base weights." This is plausible but unverified. Counter-argument: at larger d, the base model contains more nuanced representations, and expert corrections may be more sensitive to perturbations of these representations.

3. **The amplification factor c (~0.1) could be scale-dependent.** If c grows with d, the amplification effect becomes more severe at macro scale, not less.

4. **Tokenizer compatibility not tested.** Real base swapping often involves tokenizer changes. This experiment uses identical tokenizers across all conditions.

## Verdict

**PROCEED**

The experiment is well-designed for what it tests, the math is correct, the implementation is verified, and the results are consistent across 3 seeds. The core finding -- that zero-shot LoRA transfer works with graceful degradation proportional to base perturbation magnitude -- is sound and useful for the Living Composable Model architecture.

Minor issues that should be addressed in FINDINGS.md or future work but are not blocking:

1. The "Amplification Bound" (MATH.md Section 3.4) should be reframed as an "empirical trend" rather than a bound. It is a 4-point curve fit, not a theoretical guarantee.

2. Kill criterion K2 should be noted as vacuous for this experiment design (it tests nothing). Future HYPOTHESES.yml entries should have kill criteria adapted to the specific experimental setup.

3. The practical recommendations table (MATH.md Section 5.1) should note that rank-32 at d=64 is 50% of full rank, and the mapping to macro scale is nonlinear.

4. Confidence intervals on the transfer gap would strengthen the rank-32 claim (0.2% gap from 3 seeds is not distinguishable from noise without error bars).

None of these issues threaten the core conclusion. The experiment successfully disproves its kill criteria and provides directional evidence that zero-shot base transfer is viable for high-quality base approximations.

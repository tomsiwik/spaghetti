# Peer Review: bitnet_meta_scaffold

## NotebookLM Findings

Skipped (experiment already killed; review focuses on validating the kill and extracting learning value).

## Mathematical Soundness

### Bilevel formulation (MATH.md): Correct in principle

The bilevel objective is standard MAML notation. The inner loop minimizes domain loss with scaffold frozen, the outer loop updates scaffold to minimize meta-loss over inner-loop endpoints. No errors in the mathematical specification.

### FOMAML approximation: Correctly identified risk, incorrectly implemented

MATH.md correctly flags that FOMAML drops the second-order term d(theta_d^K)/dW, which accumulates over K=50 inner steps. The analysis that this term can be substantial is sound.

However, there is a deeper problem: **the outer loop gradient is not even a correct FOMAML gradient**. In standard FOMAML, the outer gradient is computed at the inner-loop endpoint parameters (the adapted model). What the code actually does (lines 510-604) is:

1. Save scaffold state before inner loop
2. Run inner loops, collect adapters
3. **Restore the original scaffold** (line 511/562)
4. Apply the **mean adapter** (line 519/563)
5. Compute gradient of loss w.r.t. scaffold at this restored+mean-adapter point

This is NOT FOMAML. In FOMAML, you compute the gradient at theta^K (the endpoint of the inner loop), which means the gradient flows through the scaffold weights AS MODIFIED by the inner loop interaction with the adapter. Here, the scaffold is restored to its pre-inner-loop state before the outer gradient is computed. The gradient therefore measures "how should the scaffold change to reduce loss when this mean-adapter is applied" -- which is a fundamentally different question from "how should the scaffold change to produce better inner-loop adaptation."

This implementation bug means the experiment is testing **direct scaffold optimization for a fixed adapter composition**, not meta-learning. The scaffold is being pushed to minimize loss under the mean-adapter, with no information about how scaffold changes affect inner-loop adaptation dynamics. This explains the scaffold destruction: the optimizer is free to move scaffold weights arbitrarily because it receives no signal about preserving trainability.

### Composition penalty: Gradient-disconnected

The composition penalty (lines 482-500) is computed as a Python float from detached adapter parameters. It contributes to the meta_loss scalar that is printed but is NOT part of the differentiable outer_loss_fn (line 585). The outer gradient receives ZERO signal from the composition penalty. The 0.5 * comp_penalty term exists only in the monitoring metric, not in the optimization.

This is a second critical implementation bug. Even if the FOMAML implementation were correct, the composition penalty would have no effect on scaffold updates.

### Convergence criterion: Reasonable

The 5% meta-loss reduction threshold across 100 steps is a reasonable convergence criterion. The observed 1.2% (first-10 vs last-10 average) clearly fails. Looking at the raw data, meta_loss actually *increased* from 2.7001 (step 1) to 2.8826 (step 100), with the "1.2% reduction" coming only from averaging windows. The meta-loss reduction field in results.json confirms this: -6.8% (i.e., 6.8% WORSE). The paper reports "1.2% reduction" which is misleading -- the paper computes first-10 avg (2.868) vs last-10 avg (2.835), but the initial point-to-point trajectory is actually divergent.

### Ternary quantization analysis: Sound

The observation that FOMAML pushes weights into distributions that quantize 12x worse is well-documented and the mechanism is plausible. Continuous Adam updates without STE produce weight distributions that straddle ternary quantization thresholds.

## Novelty Assessment

### Prior art

The paper correctly identifies three relevant references:
- MAML (Finn et al., 2017): Foundation
- Meta-LoRA (arXiv 2510.11598): Two-stage MAML for LoRA task adaptation (closest)
- "Meta-Learning the Difference" (TACL): Bilevel optimization for base weights with low-rank task reparameterization

The claim "no prior published work exists on meta-learning scaffolds for multi-adapter composition" is approximately correct. Meta-LoRA optimizes for single-task adaptation, not multi-adapter composition. The delta here is the composition objective.

### References check

REFERENCES.yml contains no entry for this specific experiment's references (Meta-LoRA, "Meta-Learning the Difference"). These should have been added before the experiment.

## Experimental Design

### Does it test the hypothesis?

**No**, due to the implementation bugs identified above. The experiment tests "does direct scaffold optimization under mean-adapter composition improve composition quality?" -- not "does MAML-style bilevel optimization produce scaffolds that support better adapter training and composition."

However, this distinction may be moot: even the simpler question (direct optimization) failed, which is informative. If you cannot even directly optimize a scaffold for a fixed adapter composition, the harder MAML problem is unlikely to succeed without significant additional machinery.

### Controls

- Standard pretrained scaffold as baseline: adequate
- GaLore scaffold as secondary comparison: good experimental design
- Both scaffolds undergo identical ternary quantization and adapter training: fair comparison
- Single seed: acknowledged limitation, acceptable for a killed experiment

### Adapter resilience finding

The most interesting finding (3% adapter PPL degradation on 12x worse scaffold) is well-supported by the data and represents genuine new information. However, the explanation needs qualification: the adapters are LoRA with scale=4.0 and rank=16 on d=256, giving an adapter-to-base parameter ratio of ~5.4% (344K / 6.4M). At this ratio, the adapter has enough capacity to partially compensate for scaffold degradation. This ratio is much higher than production (where adapters are typically <1% of base params at scale). The resilience finding may not transfer to realistic adapter/base ratios.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node (exp_bitnet_meta_scaffold). Both kill criteria are correctly evaluated:
- K1: meta comp ratio 1.172 > GaLore 1.155 -- FAIL (correctly assessed)
- K2: 1.2% reduction < 5% threshold -- FAIL (correctly assessed)

The overall KILLED verdict is correct. The evidence entry is accurate.

Note: The GaLore baseline comparison uses the single-seed GaLore result (comp ratio 1.155). The 3-seed GaLore evidence shows comp ratio 1.045 +/- 0.058. The paper should clarify which GaLore result is being compared against.

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. However, for the record:

1. The adapter resilience finding should be re-tested at realistic adapter/base ratios (r=16 on d=2560+ gives <0.5% ratio) before being cited as evidence for base-free directions.
2. Any future meta-scaffold attempt must include a scaffold preservation term (KL divergence from pretrained weights, or explicit PPL constraint) -- the paper correctly identifies this.
3. STE-aware outer loop is necessary if ternary quantization is applied post-meta-training.

## Verdict

**KILL CONFIRMED**

The kill is correct but for stronger reasons than the paper states. Beyond the two kill criteria triggering:

1. **The FOMAML implementation is incorrect.** The outer gradient is computed at the restored scaffold + mean-adapter, not at the inner-loop endpoint. This means the experiment never actually tested FOMAML -- it tested direct scaffold optimization under a fixed adapter.

2. **The composition penalty is gradient-disconnected.** The lambda * comp_penalty term never flows into the outer gradient computation. The scaffold receives zero optimization pressure for composition quality.

3. **The meta-loss "1.2% reduction" is misleading.** The actual initial-to-final trajectory is -6.8% (worse). The 1.2% figure comes from averaging windows that smooth out the divergence.

These bugs mean the experiment neither validates nor invalidates the MAML-for-scaffold idea. It invalidates "naive direct scaffold optimization without preservation constraints" -- which is still useful information. The adapter resilience finding (3% degradation on 12x worse scaffold) is the primary positive output, though it needs the adapter/base ratio caveat.

The paper's "What We Learned" section is largely correct despite the implementation issues, because the qualitative conclusions (unconstrained updates destroy scaffolds, ternary quantization amplifies damage) hold regardless of whether the gradient is FOMAML or direct.

No revision is warranted -- the kill is correct and the findings are properly recorded. The implementation bugs should be noted in FINDINGS.md if the adapter resilience finding is cited in future experiments.

# Peer Review: parallel_pure_linear_combined

## NotebookLM Findings

NotebookLM review was not performed due to the experiment being straightforward enough for direct mathematical and experimental verification. The 2x2 factorial design, the core mechanism, and the claims are all amenable to manual verification against the raw results.json data.

## Mathematical Soundness

### Kill Criterion (CORRECT)

The kill criterion in MATH.md is:

```
Kill: (L_composed(par_pure_linear) - L_composed(seq_hybrid)) / L_composed(seq_hybrid) > 5%
```

Verified against results.json:
- baseline_comp_mean (seq_hybrid) = 0.50467
- test_comp_mean (par_pure_linear) = 0.51214
- degradation = (0.51214 - 0.50467) / 0.50467 = +1.479%

This is arithmetically correct and well under the 5% threshold. No issue here.

### Factorial Analysis (CORRECT but interpretation overstated)

The factorial decomposition is standard and correctly computed:

```
parallel_effect = (par_hybrid_comp - seq_hybrid_comp) / seq_hybrid_comp = +0.066%
linear_effect   = (seq_linear_comp - seq_hybrid_comp) / seq_hybrid_comp = +1.103%
predicted       = 0.066 + 1.103 = +1.169%
actual          = (par_linear_comp - seq_hybrid_comp) / seq_hybrid_comp = +1.479%
interaction     = 1.479 - 1.169 = +0.310%
```

All verified against results.json. The math is sound.

**However**, calling this "approximately additive" is a stretch given the noise level. The interaction term (+0.31%) must be evaluated against the uncertainty in the component effects. The parallel effect is +0.066% with only 5 seeds -- the standard error of this estimate is comparable to the effect itself. When one of the additive components is essentially zero with high relative uncertainty, the additivity test has very low power. More precisely:

- parallel_effect_hybrid = +0.066% (from 5 seeds, gap_std of hybrid conditions ~0.7-1.3%)
- Standard error of the composed loss mean for each condition is roughly stdev/sqrt(5)

The per-seed composed losses for seq_hybrid have std ~0.007, giving SE ~0.003, or about 0.6% when normalized. The interaction term (+0.31%) is well within this noise band. This means we cannot distinguish between "perfectly additive" and "moderate interaction" -- the test simply lacks power.

**Verdict on factorial analysis**: Mathematically correct, but the claim of "approximately additive effects" is weakly supported. A more honest statement is: "the interaction term is indistinguishable from zero given our sample size." This is fine for micro -- it means there is no evidence of destructive interaction.

### Composition Gap Metric (SUBTLE CONFLATION)

The PAPER's Table 1 reports "Gap mean" as (composed - joint) / joint per seed, then averages. This is the composition gap -- how much composition degrades versus joint training of the same architecture.

The kill criterion, by contrast, compares composed losses ACROSS architectures (par_pure_linear composed vs seq_hybrid composed). These are different questions:

1. **Composition gap**: Does this architecture compose as well as it trains jointly?
2. **Kill criterion**: Is this architecture's composed model as good as the baseline's composed model?

The paper correctly separates these but the framing occasionally blurs the distinction. For example, "degradation +1.48%" is the cross-architecture composed loss comparison, but the PAPER's Table 1 shows per-condition composition gaps that tell a different story:

- seq_hybrid gap: -0.50% (composed is BETTER than joint)
- par_pure_linear gap: +0.96% (composed is slightly worse than joint)

The gap difference is +1.46pp, almost identical to the 1.48% degradation. But this is partly coincidental -- it works because the joint training baselines are nearly identical across conditions (0.5072 vs 0.5073). If joint training quality differed more, these metrics would diverge.

This is not a flaw in this experiment, but it is worth flagging: the near-identical joint baselines make the kill criterion robust here. At macro scale where architectural choices affect joint training quality more, the kill criterion should be re-examined.

## Novelty Assessment

### Prior Art

This is explicitly a **combination experiment**, not a novelty claim. It combines:
1. Parallel blocks (proven in micro/models/parallel_block_capsules/, citing Tiny Aya / Cohere 2026)
2. Pure-linear attention (proven in micro/models/pure_linear_composition/, building on full_gdn_stack)

The 2x2 factorial design is standard experimental methodology, appropriately applied here to test interaction effects.

### What Is New

The only novel contribution is the empirical finding that these two modifications do not interact destructively. This is a useful composition-compatibility result. No overclaim of novelty detected.

### Reference Check

No relevant prior art in `references/` that would have already answered this question. The Tiny Aya reference is cited. The GatedDeltaNet lineage through full_gdn_stack is properly credited. The code imports from parent experiments (full_gdn_stack, capsule_moe, hybrid_attention), building on rather than reinventing.

## Experimental Design

### Strengths

1. **2x2 factorial is the right design.** Testing all four combinations of {sequential, parallel} x {hybrid, pure-linear} is the textbook approach for interaction testing. The experiment actually tests what it claims.

2. **Protocol consistency.** The composition protocol (pretrain 300 steps, fine-tune 300 steps per domain, compose, calibrate 100 steps) is identical to the parent experiments. This makes the results comparable.

3. **Same seeds across conditions.** Seeds 0-4 are used for all four conditions, enabling direct per-seed comparison and reducing between-condition variance.

4. **Catastrophic failure check.** Explicitly testing for gap > 20% across all 20 runs is good practice.

### Weaknesses

1. **No statistical significance tests.** The paper reports means and standard deviations but no confidence intervals, t-tests, or bootstrap intervals. For the key claim (1.48% < 5%), a one-sided confidence interval would strengthen the result. With n=5 and the observed variance, a 95% CI upper bound on the degradation would be roughly:

   ```
   1.48% + t(0.95, df=4) * SE_difference
   ```

   The SE of the difference between two means of n=5 with pooled variance... the per-seed composed losses have std ~0.007 for each condition. SE of difference = sqrt(2 * 0.007^2 / 5) = 0.0044, or ~0.87% normalized. With t(0.95, 4) = 2.132, the upper bound is approximately 1.48% + 2.132 * 0.87% = 3.33%. This is still well under 5%, so the conclusion holds even with a proper statistical test.

2. **5 seeds is adequate but not generous.** The parent experiments used 7 seeds (pure_linear_composition) and 3 seeds (parallel_block_capsules). Using 5 here is reasonable. The confidence interval calculation above confirms this is sufficient to distinguish the observed effect from the 5% threshold.

3. **Seed 1 of par_pure_linear is an outlier.** It shows +2.14% composition gap, roughly 2 standard deviations above the condition mean (0.96%). This single seed accounts for much of the variance. The paper does not discuss this. However, +2.14% is still well under 5%, so this is not concerning for the kill criterion.

4. **No re-seeding of the joint baseline.** Each condition trains its own joint baseline with the same seed. This is correct -- the joint baseline should match the architecture of the condition. But it means the "gap" metric (composed vs joint) has correlated noise: if a seed produces a better-than-average joint model, the gap for that seed will look worse. This is a minor concern and standard in this experimental framework.

### Could a Simpler Mechanism Explain the Result?

The key question: could the combined architecture pass simply because neither modification matters much at this scale?

Looking at the effects:
- Parallel blocks: +0.066% effect (essentially zero)
- Pure-linear: +1.10% effect (small but real)
- Combined: +1.48% (dominated by the linear attention effect)

The parallel block modification is a no-op for composition quality at micro scale. The combined result is essentially the pure-linear result with noise. This is consistent with the parallel_block_capsules finding (-0.39pp, essentially zero). The "simplest composition-safe architecture" claim rests almost entirely on the pure-linear finding, not on any interaction between the two modifications.

This is not a flaw -- it is an accurate characterization of the results. But the framing in the paper could be more explicit: the combined result passes because (a) parallel blocks are neutral and (b) pure-linear is mildly worse but within tolerance. There is no evidence of a beneficial interaction that makes the combination better than either alone.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_parallel_pure_linear_combined` correctly:
- Lists dependencies on `exp_parallel_block_capsules` and `exp_pure_linear_composition`
- States the kill criterion matching MATH.md (>5% degradation)
- Records the evidence with the +1.48% result
- Sets status to "proven"

No inconsistencies found.

## Macro-Scale Risks (advisory)

1. **Gradient flow without inter-branch normalization.** In sequential blocks, the second norm (norm2) between attention and MLP acts as a gradient flow regulator. In parallel blocks, both branches share one norm, and their gradients sum directly into the residual. At 24+ layers, this could cause activation magnitude divergence. The paper acknowledges this in its "What Would Kill This" section.

2. **GatedDeltaNet state capacity at d_h=256.** Pure-linear means no full attention layer anywhere to serve as a "memory reset." At macro scale with d_h=256, the 256x256 recurrent state may saturate. The paper acknowledges this.

3. **The 1.48% gap may not be stable at N=5+ domains.** This experiment only tests N=2 domain composition. The interaction between parallel blocks and pure-linear attention might become more pronounced with more domains competing for capsule routing. The paper acknowledges needing N=5 testing.

4. **The speed benefit is the real macro value, not quality.** The quality story is "not worse by more than 5%." The speed story (parallel execution + O(T) attention) is where the macro value lies. This should be the primary macro evaluation criterion, not the quality gap.

## Verdict

**PROCEED**

The experiment is well-designed, the math is correct, the code implements what it claims, and the results clearly pass the kill criterion with margin to spare (1.48% vs 5% threshold, upper 95% CI ~3.3%). The 2x2 factorial is the right approach for testing interaction effects.

Minor issues that do not block proceeding:

1. The "approximately additive" claim is weakly supported due to low statistical power on the interaction term. The honest statement is "no evidence of destructive interaction." The paper should note this distinction, but it does not change the conclusion.

2. No formal statistical tests are reported. The confidence interval analysis above confirms the conclusion holds, but reporting CIs would improve rigor.

3. The parallel block contribution to composition quality is essentially zero at micro scale. The combined result is dominated by the pure-linear effect. This is fine -- it means parallel blocks are neutral for composition, which is the desired outcome. But the framing should be clear that the "simplest composition-safe block" claim rests on two independent findings (parallel is neutral, pure-linear is within tolerance), not on a synergistic combination.

These are presentation refinements, not fundamental concerns. The mechanism works in principle, the experimental evidence supports the claims within stated tolerances, and the hypothesis graph is consistent.

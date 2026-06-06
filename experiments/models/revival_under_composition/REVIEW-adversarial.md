# Peer Review: Revival Under Composition

## NotebookLM Findings

Skipped -- the experiment's MATH.md and PAPER.md are sufficiently clear for direct review. The hypothesis is straightforward (does composition change revival rate?), the experimental design has three well-motivated conditions, and the results are unambiguous in direction.

## Mathematical Soundness

### Derivations

The MATH.md is primarily a setup document -- it frames the gradient competition vs. input diversity hypotheses qualitatively rather than proving theorems. This is appropriate; the experiment is empirical, not theoretical. No mathematical claims are made that require formal proof.

The gradient competition argument (Section 3.1) is sound as an intuition:

```
||Delta W_composed|| <= ||Delta W_single||    (potential cancellation)
```

This is stated as a "potential" inequality, not a proven bound. The paper correctly treats it as a hypothesis to be tested empirically, not a mathematical fact. Acceptable.

### Revival rate computation

The revival rate formula is correctly implemented. `transition_counts` from Exp 18 is reused. The code (line 268-271) correctly computes:

```
revival_a = trans_a["da"] / (trans_a["dd"] + trans_a["da"])
```

This is the fraction of anchor-dead capsules that are now alive, which matches the definition in MATH.md Section 5. Verified correct.

### Domain-half splitting

The `split_composed_mask_by_domain` function (lines 93-113) correctly extracts the first P and second P capsules per layer from the composed model's flat mask. Unit tests verify this. No issues.

### Statistical claim error (minor)

The paper states the 8.6 pp gap is "~1.9 sigma" (PAPER.md, Limitations, item 6). This appears to be computed as 8.6 / 4.6 = 1.87, dividing the gap by the single-domain standard deviation alone. This is not a valid statistical test. The proper calculation using pooled standard error (n=6 single, n=3 composed) yields t approximately 3.7, which is substantially more significant than reported. The paper undersells its own result, which is the safer direction of error. Not blocking, but the claim should be corrected.

## Novelty Assessment

### Prior art

This experiment fills a genuine gap in the project's knowledge base. Experiments 16, 18, and 20 established: (a) the same capsules die in composed vs. single-domain models, (b) revival exists and is driven by inter-layer coupling, and (c) the mechanism is upstream weight updates shifting downstream input distributions. None measured whether revival RATE changes under composition. This is a natural and necessary follow-up.

No external prior art directly addresses "revival rate of dead ReLU neurons under model composition by weight concatenation." The Gurbuzbalaban et al. reference is appropriately cited for context (transient revival).

### Delta over existing work

The key new finding -- that the suppression is 87% structural (7.6 pp from structure alone vs. 8.6 pp total) -- is genuinely informative. It means the effect is NOT about cross-domain gradient cancellation (the primary hypothesis in MATH.md Section 3.1) but about the dimensionality of the composed weight space. This is a surprising and useful result that the paper correctly highlights in Finding 2.

## Experimental Design

### Does the experiment test what it claims?

Yes. Three conditions cleanly isolate the components:
- **Condition A** (single-domain): baseline revival rate
- **Condition B** (composed + joint): revival under composition with cross-domain data
- **Condition C** (composed + own-domain): revival under composition WITHOUT cross-domain data

Condition C is the critical control. The fact that C shows 7.6 pp suppression vs. A's baseline (almost as much as B's 8.6 pp) definitively pins the effect on structure rather than data. This is well-designed.

### Confound: profiling data asymmetry

Single-domain models are profiled on own-domain validation data. Composed models are profiled on joint validation data (line 187: `val_ds_joint`). This means composed models see "out-of-distribution" inputs during profiling that single-domain models do not.

Could this explain the lower revival rate? If joint profiling classifies more capsules as dead (because cross-domain inputs don't activate in-domain capsules), the anchor dead set would be larger and include capsules that are "only dead on cross-domain inputs" -- these would be harder to revive because they were never truly dead on their own domain.

The paper cites Exp 16 (Jaccard=0.895) to argue the profiling difference is small. This is reasonable but incomplete: Jaccard=0.895 means the dead SETS are similar, but it doesn't address whether the ~10% of non-overlapping capsules have different revival propensities. However, since Condition C (composed + own-domain training) still shows 7.6 pp suppression despite training only on the capsules' native domain data, the profiling confound cannot explain the full effect. The structural argument holds.

### Confound: model capacity

The composed model has 2x the capsules per layer. During continued training, gradients are spread across twice as many parameters. This is not "gradient competition from cross-domain data" -- it is a pure capacity effect. The paper correctly identifies this in Finding 2 ("wider weight matrices mean inter-layer coupling must shift a higher-dimensional space") but does not cleanly separate it from a gradient-per-parameter dilution effect.

A fourth condition would strengthen the paper: a single-domain model with 2x capsules (256 per layer) trained on only one domain. This would isolate whether the suppression comes from (a) having more capsules to update, or (b) having capsules from a different domain initialization. This is a "nice to have" rather than blocking, since the practical conclusion (revival is suppressed in composed models) is the same either way.

### Could a simpler mechanism explain the results?

Yes, partially. The 2x parameter count alone could explain reduced per-capsule gradient magnitude and thus slower revival. The paper's narrative attributes this to "gradient competition" and "cross-domain cancellation," but the Condition C results show it is mostly a structural/capacity effect. The paper does acknowledge this (Finding 2) but the framing in the abstract and opening could be clearer that the suppression mechanism is dimensionality, not cross-domain interference.

### Seed count and statistical power

Three seeds is thin. The paper reports standard deviations at S=+3200 of 4.6% (single) and 2.4% (composed). With n=6 and n=3 respectively, a two-sample t-test gives t approximately 3.7 (df approximately 5), p approximately 0.01 -- statistically significant at the 0.05 level but not overwhelmingly so. The 95% confidence interval for the difference includes values from roughly 3 pp to 14 pp. The effect is real but its magnitude is uncertain. Acceptable for micro-scale.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_revival_under_composition` has kill criterion: "composition changes revival rate by <5pp vs single-domain." The experiment measures exactly this. The 8.6 pp result exceeds the threshold. Status "proven" is appropriate.

The node has no dependencies listed and blocks nothing. This is consistent -- it is an informational finding that refines the practical protocol, not a gate for other experiments.

## Macro-Scale Risks (advisory)

1. **Practical calibration regime dominates.** The paper's own data shows that at S=+100 (the practical calibration window), the suppression is only 2.2 pp -- below the 5 pp threshold. The "proven" verdict relies on S=+3200, which is 16x longer than practical calibration. The paper correctly flags this (Section "What Would Kill This"), but the HYPOTHESES.yml evidence claim emphasizes the 8.6 pp number without adequate context about the 2.2 pp at practical scale.

2. **Warmup + cosine decay changes baseline.** Exp 19 showed that standard macro training schedules reduce death from ~47% to ~20% and dramatically change revival dynamics. The composition suppression effect measured here (under constant LR) may not transfer. At macro scale, if baseline revival is already low (~5%), a further 50% suppression would be only ~2.5 pp -- below threshold.

3. **SiLU base models have 0% dead capsules.** The macro model (Qwen2.5-Coder-0.5B) uses SiLU, which has an activation floor preventing dead neuron formation (Exp 15). This entire line of revival research is specific to ReLU architectures. If the macro path uses SiLU, revival dynamics are moot. This is not a flaw in the experiment but limits its applicability.

## Verdict

**PROCEED**

The experiment is well-designed, the code is correct, the results are clear, and the practical conclusion (pruning timing is less critical in composed models) is useful. The findings strengthen the pre-composition pruning recommendation from Exp 16.

Minor issues that do not block PROCEED:

1. **Fix the statistical claim.** The "~1.9 sigma" in Limitations item 6 is incorrectly computed (divides by single-domain stdev instead of pooled standard error). The actual significance is higher (~3.7 on a two-sample test). Replace with either a proper t-statistic or remove the specific number and say "directionally strong with 3 seeds."

2. **Clarify the mechanism.** The framing suggests "gradient competition" (cross-domain cancellation) is the primary mechanism, but Condition C shows 7.6/8.6 = 88% of the effect is structural. The abstract/opening should lead with "dimensionality dilution" rather than "gradient competition."

3. **Note the practical regime caveat more prominently.** The 8.6 pp result is at S=+3200. At the practical calibration length of S=+100, suppression is only 2.2 pp. The HYPOTHESES.yml evidence should mention both numbers.

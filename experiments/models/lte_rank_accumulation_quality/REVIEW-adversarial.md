# Peer Review: LTE Rank Accumulation Quality (d=256)

## NotebookLM Findings

Skipped -- the experiment is sufficiently self-contained and the math tractable enough for direct review without external tooling.

## Mathematical Soundness

### Rank capacity analysis (MATH.md Section 2): Correct with caveats

The per-interval rank bound is correct:
- Parallel: rank(dW_parallel) <= min(K*r, d) = min(32, 256) = 32
- Sequential: rank(dW_sequential) <= r = 8

However, the "coverage" metaphor (32/256 = 12.5%) is misleading. Rank is not coverage of "dimension space." A rank-32 matrix lives in a 32-dimensional subspace of R^256, but this does not mean it "covers 12.5% of the space." The Grassmannian Gr(32, 256) has dimension 32 * (256 - 32) = 7168, so a single rank-32 subspace is a zero-measure point in that space. The ratio r/d is a useful proxy for capacity constraint severity, but calling it "coverage" implies a filling metaphor that is mathematically incorrect.

### Saturation argument (Section 2.3): Partially flawed

The claim that at d=64 "parallel is already SATURATED (rank capacity = d)" is the central motivation for scaling to d=256. Let me verify:

- Parallel at d=64: 2 intervals * 32 = 64 capacity. The paper claims this equals d, hence saturation.
- But rank accumulation across merge intervals is NOT additive in general. After merging dW_1 into the base, the next dW_2 is trained on the updated base. The total rank of W_base + dW_1 + dW_2 is at most rank(W_base) + 16 + 16 = rank(W_base) + 32, not rank(W_base) + 64. The intervals are NOT independent rank contributions.

The saturation argument holds only if the merged deltas span orthogonal subspaces, which is not guaranteed and likely false when training on the same distribution. This weakens the motivation but does not invalidate the experiment since the experiment is empirical.

### Merge implementation (code line 159-171): Correct

The parallel merge correctly:
1. Computes delta = scaling * (A @ B).T with alpha/r scaling (line 165)
2. Averages K deltas (line 167)
3. Adds to base weight (line 168)
4. Resets LoRA to zero (lines 169-170)

The .T transpose is needed because LoRALinear stores delta as A @ B with shape (in_dim, out_dim) but linear.weight has shape (out_dim, in_dim). This matches the forward pass correctly.

### Sequential merge (code line 227-239): Correct

Uses get_delta() which applies alpha/r scaling, then transposes for weight addition. Consistent with parallel.

### Magnitude dilution claim (Section 4.2): Correct and important

The 1/K averaging is real. Each parallel head trains for T=25 steps, then the average of K=4 deltas is merged. The per-direction magnitude is 1/4 of what a single head would contribute. Sequential trains for 4T=100 steps in 8 directions with full magnitude. This is the most compelling explanation for why parallel shows no advantage.

### Kill criteria logic (code line 646-672): Has a logic error

Seed 42 is classified as "PROVEN (parallel shows quality advantage)" because `parallel_better` is True when `pvc < 0.99` (par_vs_seq_cos = 0.71, meaning parallel has LOWER cosine = MORE orthogonal). But this is the ONLY seed where parallel wins on cosine, and it wins precisely because seed 42's sequential cosine (0.034) is anomalously high. The aggregate correctly classifies this as INCONCLUSIVE.

The kill criterion K1 ("quality difference <1%") is poorly specified because it uses only base loss difference, not a composite. At d=256:
- Base loss diff: 2.3% (>1%, not killed by K1)
- Expert loss diff: 0.7% (<1%, would be killed)
- Cosine diff: 106% (>1%, not killed -- but in the WRONG direction for parallel)

The PAPER.md verdict of "INCONCLUSIVE (2 of 3 seeds)" is more honest than the code's automatic classification, but the correct aggregate verdict should be KILLED for the hypothesis that "parallel is better," since parallel is consistently worse or equal on all metrics except one seed's cosine.

## Novelty Assessment

### Prior art

The LTE paper (Hyeon-Woo et al., 2024, arXiv:2402.16828) is the primary reference and is correctly cited. The experiment is a direct scale-up test of LTE's rank accumulation claim, which is a valid and useful contribution to the SOLE architecture decision.

### Delta over existing work

The finding that rank accumulation does NOT help is a useful negative result. The explanation (1/K dilution + data homogeneity) is a genuine insight, though it is somewhat obvious in hindsight: if K heads train on the same distribution, they will find similar subspaces, and averaging them dilutes rather than enriches.

### Bonus finding: LoRA-merge advantage scales with d

The observation that LoRA-merge vs conventional advantage grows from 2% at d=64 to 33% at d=256 is the most valuable finding. However, this needs scrutiny.

**The continued conventional baseline is suspicious at d=256.** Seed 42: cont_val = 0.76 (reasonable). Seed 123: cont_val = 1.03 (worse than pretrained 0.70). Seed 7: cont_val = 1.32 (far worse than pretrained 0.76). The conventional baseline appears to be diverging or overfitting at d=256, which inflates the LoRA-merge advantage.

This is likely a learning rate issue: lr=3e-3 may be too high for conventional full-model training at d=256 (more parameters = smaller optimal LR), while LoRA training is inherently regularized by the low-rank constraint. The "33% advantage" may be an artifact of a poorly-tuned conventional baseline rather than an intrinsic advantage of LoRA-merge. The paper does not discuss this confound.

## Experimental Design

### Does it test the hypothesis? Partially.

The hypothesis is that parallel rank accumulation produces better composition substrates at d=256. The experiment correctly measures base quality, expert quality, and expert orthogonality. The design is a clean scale-up of the parent experiment.

### Controls: Adequate but incomplete

1. Three seeds -- adequate for micro.
2. Compute-fair comparison (same total gradient steps) -- correct.
3. Same data, same hyperparameters -- correct.
4. Missing control: the conventional baseline learning rate should have been tuned separately. At d=256, the optimal LR for full-model training is almost certainly lower than 3e-3. This makes the LoRA-merge vs conventional comparison unfair.

### Confounds

1. **Data homogeneity.** All heads train on different random batches from the same distribution. LTE's claim is about parallel heads on different DATA SHARDS (domains), not random batches. The experiment does not faithfully reproduce LTE's setup. However, PAPER.md acknowledges this.

2. **FFN-only LoRA.** The experiment applies LoRA only to MLP layers (fc1, fc2), not attention layers. This is inconsistent with the project decision to use all-modules adapters. At d=256, attention cosines are known to be higher; testing only FFN may miss interference effects.

3. **Conventional baseline tuning.** As noted above, the diverging conventional baseline at seeds 123 and 7 inflates the LoRA-merge advantage claim.

### Hypothesis graph consistency

The kill criteria in HYPOTHESES.yml match the experiment:
- K1: "K parallel rank-r branches merged show no quality advantage over K sequential rank-r merges at d>=256" -- tested, parallel shows no advantage.
- K2: "quality difference <1% at d=256" -- base loss difference is 2.3%, so technically not killed by K2. But the difference favors sequential, not parallel.

The experiment is classified as "killed" in HYPOTHESES.yml with the correct reasoning. The aggregate verdict in the code says INCONCLUSIVE, which is less clean than the HYPOTHESES.yml classification of KILLED. The FINDINGS.md correctly says "KILLED: par/seq base=1.023, cos=2.06."

## Macro-Scale Risks (advisory)

1. **The LoRA-merge vs conventional scaling claim should NOT be carried forward without LR tuning.** At macro scale (d=4096), lr=3e-3 would almost certainly cause catastrophic divergence in conventional training. This confound must be resolved before claiming LoRA-merge advantage scales with d.

2. **LTE's no-reset variant was not tested.** The LTE paper's primary claim is about no-reset mode with correction terms, not reset-after-merge. The experiment only tests reset-after-merge. This is noted in limitations.

3. **Domain-diverse shards.** LTE's advantage may only manifest with genuinely different data distributions per head (code/medical/legal), not random batches from the same distribution. This is the correct test for macro.

## Verdict

**PROCEED**

The experiment is killed (correctly -- parallel rank accumulation provides no quality advantage over sequential), and the kill is well-supported by the data. The experiment should be considered complete.

However, there are 3 non-blocking issues that should be noted in FINDINGS.md:

1. **Conventional baseline confound.** The "LoRA-merge advantage grows with d" claim (2% at d=64 to 33% at d=256) is contaminated by an untuned conventional baseline that diverges at seeds 123 and 7. This bonus finding should be flagged with a caveat that the conventional LR was not tuned for d=256. Do not carry this claim forward to macro without a properly-tuned baseline.

2. **Kill criteria K1 is ambiguous.** The kill criterion says "quality difference <1%," but base loss difference is 2.3% while expert loss difference is 0.7%. The paper should be explicit: parallel is killed because it is WORSE (or at best equal), not because the difference is small. The kill is a directional kill (parallel provides no advantage), not a magnitude kill (differences are small).

3. **FFN-only LoRA inconsistency.** The experiment uses FFN-only LoRA, which is inconsistent with the project's locked decision to use all-modules adapters. This does not invalidate the kill (the fundamental dilution argument holds regardless of which modules have LoRA), but should be noted for completeness.

The core conclusion -- use ReLoRA (sequential) for single-device SOLE base construction, LTE (parallel) only when multi-GPU parallelism justifies the engineering complexity -- is sound and well-supported.

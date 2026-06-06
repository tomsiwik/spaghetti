# Peer Review: Composition Health via KL Divergence

## NotebookLM Findings

Skipped -- the experiment is already executed with clear numerical results on all three kill criteria. The failure modes are transparent and the analysis in PAPER.md is thorough. A deep review would not surface additional issues beyond what systematic attack reveals below.

## Mathematical Soundness

### What holds

1. **KL definition and computation are correct.** The code properly uses `F.kl_div(log_p_base, log_p_comp, log_target=True, reduction="sum")`, which computes KL(P_composed || P_base). The direction convention is consistent between MATH.md and implementation.

2. **Leave-one-out decomposition is correctly defined.** DeltaKL_i = KL(P_N || P_base) - KL(P_{N-i} || P_base) is a valid marginal contribution measure.

3. **The Taylor expansion analysis (MATH.md lines 107-123) is correct in form.** For small perturbations, KL does decompose into a quadratic form weighted by Fisher information, and cross-terms vanish under orthogonality.

### What does not hold

4. **The "small perturbation" assumption is violated.** MATH.md Section "Theoretical Analysis" derives KL ~ (1/2) h^T (sum delta_i)^T F (sum delta_i) h under a Taylor expansion. But empirically, KL_5 ~ 10.4 nats. This is not a small perturbation -- 10 nats of KL divergence means the composed and base distributions are dramatically different. The linear/quadratic regime analysis is inapplicable at this scale. The entire theoretical framework predicting "superlinear growth signals interference" rests on a regime the system never operates in.

5. **The predicted KL magnitudes in MATH.md are wrong by orders of magnitude.** The worked example (line 314) predicted KL_5 "should be small (< 1 nat)" but observed KL_5 ~ 10.4 nats. This 10x miss indicates the theoretical model fundamentally misestimates the perturbation scale of PEFT composition at the 7B parameter count.

6. **Statistical power assessment is honest but the experiment proceeded anyway.** MATH.md Assumption 5 correctly notes that rho=0.3 at N=10 has p~0.4 (not significant), yet the kill criterion was set at rho >= 0.3. In practice, N=5 made this even worse -- Spearman rho at N=5 has essentially no statistical power. The experiment acknowledged this limitation but did not adjust the protocol.

## Novelty Assessment

### Prior art

The idea of using KL divergence from a reference model to monitor fine-tuned model quality is not novel. Relevant precedents:

- **Rafailov et al. (2023), DPO:** Uses KL(pi || pi_ref) as a regularization constraint to prevent policy drift during alignment. The observation that KL from reference measures "impact magnitude, not quality" is well-established in the RLHF literature -- this is exactly why DPO and PPO use KL as a constraint, not as a quality signal.
- **Knowledge distillation literature (Hinton et al., 2015):** KL divergence is a training signal measuring distribution match, not quality match.
- **Model merging (TIES, DARE):** These methods address interference through parameter-space techniques, not distributional monitoring.

### Delta over existing work

The experiment's actual contribution is **negative**: it demonstrates that KL(composed || base) anti-correlates with quality in the SOLE composition regime. This is a useful finding that aligns with and extends the earlier micro-scale result (cosine gating, r=-0.46). The emerging pattern -- that unsupervised distributional metrics systematically anti-correlate with quality -- is the real finding worth recording.

## Experimental Design

### Severe confound: N=5 instead of N=50

The experiment was designed for 50 pilot adapters but only 5 were available on the GPU instance. This is an operational failure, not a design flaw, but it dramatically weakens every conclusion:

- **K1:** Spearman correlation at N=5 is meaningless (p=0.19). The rho=-0.7 has wide confidence intervals spanning [-1.0, +0.5]. This cannot distinguish a true anti-correlation from noise.
- **K2:** With only 5 adapters, the leave-one-out z-score distribution has 5 points. The standard deviation estimate is unreliable.
- **N-scaling:** Only one data point (N=5). The superlinear growth hypothesis was the primary theoretical prediction and went completely untested.

### The synthetic harm construction is flawed

Negating lora_B of an adapter that is already part of the composition creates an anti-expert that cancels the original adapter's contribution. This is not "harmful" in the sense the hypothesis requires. A harmful expert should push predictions in a novel wrong direction. Instead, the negated adapter acts as a subtraction, which necessarily reduces KL (pulling composed distribution toward base by canceling one adapter's contribution). This failure was foreseeable from the math: if expert_i contributes +delta_i, then the negated version contributes -delta_i, and the net effect on the composition is approximately zero contribution from that domain.

A better construction would have been: (a) a randomly initialized adapter (pure noise in a novel direction), or (b) an adapter trained on deliberately wrong labels, or (c) an adapter from a completely unrelated domain with anomalously large weight norms.

### Calibration text domain overlap

4 of the 20 calibration texts are code snippets (Python, SQL, bash). Three of the 5 available adapters are code-related (bash, python, sql). This means 60% of the experts have domain overlap with 20% of the calibration set. The calibration set is not domain-agnostic for this adapter population. This could bias the KL measurement toward reflecting code-expert quality disproportionately.

### K3 is an engineering failure, not a mechanism failure

The 45s/composition time comes from reloading the full 7B base model for each leave-one-out composition. The SPEC explicitly noted this could be solved with merge_and_unload() or incremental adapter subtraction. The theoretical estimate of 8s was based on reusing the base model. K3 failing does not say anything about whether KL divergence works as a health metric -- it says the implementation was inefficient. This should not count toward the kill verdict on the mechanism itself.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry correctly records the kill with all three criteria failing. The evidence chain from exp_quality_degradation_detection (cosine gating killed, canary queries proven) to this experiment is logical. The recorded pattern -- unsupervised distributional metrics anti-correlate with quality -- is a valuable meta-finding.

However, the kill is recorded with more confidence than the evidence warrants given N=5.

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. But for the record:

1. At N=50, the KL_N ~ 10.4 * 10 (linear extrapolation) ~ 100 nats would put the system deep into the non-perturbative regime where the theoretical analysis completely breaks down.
2. The anti-correlation finding, if robust at N=50, would definitively rule out all KL-based composition monitoring for SOLE.

## Verdict

**KILL** -- but with caveats that weaken the confidence of the kill.

The experiment correctly identifies a real problem: KL divergence measures expert impact magnitude, not quality. This finding is consistent with the prior cosine-gating result (r=-0.46) and strengthens the meta-conclusion that unsupervised distributional metrics do not work for SOLE composition health monitoring. The kill is directionally correct.

However, the evidence is weaker than the PAPER.md presents:

1. **N=5 is insufficient for any correlation conclusion.** The rho=-0.7 at p=0.19 is not statistically significant. The true correlation could plausibly be positive.
2. **The synthetic harm test is flawed by construction.** B-matrix negation necessarily reduces KL by canceling the original expert's contribution. This does not test whether KL can detect genuinely harmful experts.
3. **K3 is an implementation artifact, not a mechanism failure.** The base model reload bottleneck is solvable.

If this were a micro experiment, I would say REVISE with fixes. But this is a macro experiment that consumed ~$0.19 of GPU time and the directional signal (anti-correlation) matches the prior micro result. The probability that running with all 50 adapters would flip the anti-correlation to rho >= +0.3 is low (though nonzero). The cost of re-running with all 50 adapters properly staged would be ~$0.50-1.00.

**Recommendation:** Accept the kill, but record the following caveats in FINDINGS.md:

1. The kill is based on N=5, not the designed N=50. Statistical power is insufficient for confident conclusion.
2. The synthetic harm construction (B-negation) is invalid -- it tests cancellation, not harm. If this mechanism is ever revisited, use random-noise adapters or adversarially trained adapters.
3. The anti-correlation pattern (KL captures magnitude not quality) is the real finding, consistent with cosine-gating kill at micro. This pattern should be treated as supported, not proven, given the sample size.
4. K3 (speed) should be disregarded as mechanism evidence -- it is an engineering implementation issue.

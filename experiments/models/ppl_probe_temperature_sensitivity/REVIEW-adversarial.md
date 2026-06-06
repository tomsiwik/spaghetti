# Peer Review: PPL-Probe Temperature Sensitivity

## Mathematical Soundness

**Correct elements:**

- The softmax formulation, entropy analysis, and oracle construction via least squares are all standard and correctly implemented.
- The gap improvement metric is well-defined and meaningful.
- ANOVA and Kruskal-Wallis tests are appropriate for comparing across tau groups.
- Numerical stability (subtracting max before exp) is handled properly in code.

**Issues:**

1. **K2 kill criterion is inconsistently stated vs. implemented.** MATH.md says K2 triggers when `|log2(tau*/1.0)| > 1`, i.e., optimal tau differs from 1.0 by more than 2x. The code checks `tau < 0.5 or tau > 2.0` (strict inequality). For K=2 and K=3, optimal tau is 0.5, which is exactly 2x from 1.0 -- this sits right on the boundary. The code reports K2 violations as `[0.1]` (only the K=5 case), yet PAPER.md claims "All differ from 1.0 by >=2x." This is sloppy: tau=0.5 is exactly 2x, not >2x. The kill still holds due to K1 alone and due to K=5's tau=0.1, but the paper overstates the K2 evidence.

2. **The monotonic tau-gap relationship is mathematically trivial.** Given that losses are Frobenius distances with non-degenerate spread, lower tau produces more concentrated weights on the best expert. Since the best expert (lowest distance) is by construction closest to the target, concentrating weight on it trivially reduces distance to target vs. equal weighting. The experiment essentially measures a known property of softmax temperature -- it does not reveal anything surprising. The MATH.md partially acknowledges this ("critical scale is tau ~ std(s)") but the paper does not emphasize how unsurprising the result is.

3. **Oracle allows negative weights, PPL-probe does not.** MATH.md acknowledges this in Limitations but does not analyze its impact. The oracle gap metric compares softmax-constrained weights against an unconstrained oracle. This means the "oracle gap" numbers are not meaningful as a measure of how close PPL-probe gets to the best possible softmax weighting. A constrained oracle (softmax with optimal tau) would be a more relevant baseline. This does not invalidate the gap-vs-equal-weight metric, which is the primary one used.

## Novelty Assessment

**Low novelty, but appropriate for its purpose.** This is a parameter sensitivity study, not a novel mechanism. It tests whether a hyperparameter choice matters -- a necessary engineering validation step.

There is no prior art to cite because this is specific to the SOLE PPL-probe weighting scheme. The finding that softmax temperature affects weight concentration is textbook knowledge. The value added is quantifying the effect in the SOLE context and recommending tau=0.5.

## Experimental Design

**The fundamental flaw: synthetic losses do not predict real PPL loss scales.**

This is the central weakness and the paper correctly flags it in Limitations, but then proceeds to make a concrete recommendation ("change default tau from 1.0 to 0.5") that cannot be justified from synthetic data alone. Here is why:

- The synthetic "losses" are Frobenius distances between random d=32 matrices. These have a specific scale and distribution (chi-like, with mean and variance determined by d, r, and the initialization scale of 0.1).
- Real PPL losses on probe inputs have a completely different scale. For a language model, per-token cross-entropy losses might range from 1.0 to 8.0, with expert-to-expert differences on the order of 0.1-2.0 depending on domain relevance.
- The optimal tau is entirely determined by the ratio of tau to the loss spread (as MATH.md correctly derives). Since synthetic loss spread bears no relationship to real PPL loss spread, the specific value tau=0.5 is an artifact of the synthetic setup, not a transferable recommendation.

**What the experiment does show (validly):**

- Temperature IS a sensitive hyperparameter (K1 holds regardless of loss scale).
- The qualitative pattern (lower tau = more discriminating = higher improvement but higher variance) will hold at any scale.
- tau should be calibrated to the loss spread, not fixed at an arbitrary value.

**What it does NOT show:**

- That tau=0.5 is the right value for real PPL losses. It could be 0.01 or 10.0 depending on actual loss scales.
- That 60% more improvement is achievable by changing tau. This number is entirely scale-dependent.

**Controls are adequate** for the synthetic setting. Equal-weight baseline is the right comparison. 50 seeds provide sufficient statistical power.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node. Kill criteria K1 and K2 are the ones actually tested. The kill is justified by K1 alone (max std 11.76pp vs 5pp threshold). The status change to "killed" is correct.

However, the evidence field in HYPOTHESES.yml includes the actionable recommendation "change PPL-probe default from tau=1.0 to tau=0.5" which, per the analysis above, is not supported by this experiment. The valid recommendation is: "tau should be calibrated to the actual loss spread; tau=1.0 is not necessarily optimal."

## Macro-Scale Risks (advisory)

1. **Loss scale mismatch is the primary risk.** At macro scale with real PPL losses, the entire tau landscape shifts. The first macro experiment using PPL-probe should include a tau sweep calibrated to the observed loss spread, not assume tau=0.5.

2. **Adaptive tau is the right direction.** Setting tau = c * std(losses) for some constant c is more principled than any fixed tau. The experiment's spread sensitivity analysis (lines 273-305 in code) supports this -- tau sensitivity varies with spread scale. This adaptive approach should be tested at macro.

3. **K=5 optimal at tau=0.1 vs K=2,3 at tau=0.5 suggests K-dependent tau.** This makes sense: more experts means the weight distribution needs to be more concentrated to avoid dilution. At macro with potentially K=10+, tau may need to be even lower.

## Verdict

**PROCEED** (as a killed experiment with caveats on the actionable recommendation)

The kill decision is correct and well-supported: temperature IS sensitive, and tau=1.0 is not special. The experimental design, statistics, and code are sound for what they measure. The kill criteria were met cleanly.

However, the actionable recommendation needs revision:

1. **Revise the recommendation in HYPOTHESES.yml evidence and PAPER.md.** Replace "change default tau from 1.0 to 0.5" with "tau must be calibrated to the actual loss spread; a macro-scale tau sweep is required before setting a default." The value 0.5 is an artifact of synthetic Frobenius distance scales.

2. **Add to PAPER.md's "What This Means for SOLE" section:** The principled approach is adaptive temperature: tau = c * std(losses), where c is determined empirically at macro scale. This is the real finding, not a specific tau value.

3. **Note in HYPOTHESES.yml:** The macro PPL-probe experiments should include a tau sweep as a standard step, calibrated to observed loss distributions.

These are documentation fixes, not re-runs. The experiment itself is complete and its kill is valid.

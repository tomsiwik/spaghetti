# Peer Review: Cross-Adapter Knowledge Transfer

## NotebookLM Findings

Skipped -- experiment is a clean negative result with straightforward math. Deep review not warranted for a KILL confirmation.

## Mathematical Soundness

### Transfer Coefficient Definition (correct but problematic)

The transfer coefficient T(i,j) = (PPL_native - PPL_composed) / PPL_native is a standard relative improvement metric. The math is correct.

**However, the composition formula has a critical confound:**

```python
W_composed = W_base + alpha * Delta_foreign + (1 - alpha) * Delta_native
```

This is a *convex interpolation*, not an *additive composition*. When alpha=0.1, the native adapter is downweighted to 0.9 of its original strength. The experiment simultaneously tests two things:
1. Does adding foreign knowledge help?
2. Does diluting the native adapter hurt?

These effects cancel. A transfer of +0.24% means the foreign adapter at 10% weight contributed slightly more than the 10% native signal that was removed. The correct test for "does foreign knowledge help" would be:

```
W_additive = W_base + beta * Delta_foreign + 1.0 * Delta_native
```

where beta is searched over small values. This isolates the additive contribution without diluting the native signal.

**Impact on verdict:** This confound makes the null result STRONGER, not weaker. If convex interpolation (which dilutes the native adapter) still shows near-zero transfer, then the adapters are truly independent. But the paper's quantitative claim ("max 0.24% transfer") is not a measure of knowledge transfer -- it is a measure of how much replacing 10% of native signal with foreign signal costs vs. helps. The distinction matters for the interpretation.

### Kill Criteria (reasonable)

K1 (>2% improvement on at least 1 pair) is a reasonable threshold for a 5-domain matrix. At 20 pairs, finding zero is strong evidence.

K2 (matrix structure) uses variance > 0.5 OR structure gap > 1% OR symmetry > 0.3. These thresholds are somewhat arbitrary but the actual values are so far below (variance 0.004, gap 0.02%, symmetry -0.24) that threshold choice does not matter.

### The "best_alpha=0.0" Problem

For 15/20 pairs, best_alpha=0.0, meaning ALL tested alpha values made things worse. The code correctly handles this -- `best_ppl` starts at `native_ppl` and `best_alpha` starts at 0.0, so if no alpha improves PPL, the transfer coefficient is exactly 0.0. This is correct implementation.

But it means 15/20 pairs showed strictly destructive interference at all tested blending weights. This is a stronger result than "no transfer" -- it is "any blending hurts." The paper underplays this.

### Row/Column Means Are Informative

The column means reveal that python is the only domain that benefits at all (col_mean = 0.113%), while math/legal/creative receive exactly 0.0% from all foreign adapters. This is a real structural pattern (python receives from all, nothing else receives from anything) but the effect size is negligible. The paper's explanation (python code contains elements of all domains) is reasonable.

## Novelty Assessment

### Prior Art

This experiment tests a natural question that arises from the Grassmannian orthogonality framework already proven in this project. It is not claiming novelty -- it is testing whether a specific mechanism (pairwise cross-domain transfer) exists.

The result directly addresses a claim in `FINDINGS.md` (macro/pilot50_composition_quality caveats): "composed PPL substantially better than naive 1/N dilution prediction... suggesting constructive cross-domain transfer." This experiment correctly identifies that the prior "constructive transfer" observation was likely regularization from 1/N scaling, not genuine knowledge sharing. This is a valuable clarification.

### Related Work

The concept of measuring adapter transfer matrices has appeared in the multi-task learning literature (e.g., Fifty et al., "Efficiently Identifying Task Groupings for Multi-Task Learning," NeurIPS 2021), though not specifically for orthogonally-initialized LoRA adapters on ternary bases. The experiment does not claim novelty here, so absence of citation is not a problem for a micro experiment.

## Experimental Design

### Strengths

1. **Clean negative result.** 0/20 pairs exceeding threshold is unambiguous.
2. **Alpha sweep is adequate.** Testing {0.1, 0.2, 0.3, 0.5} covers the reasonable range. Very small alphas (0.01) would just add noise.
3. **Individual adapters all work.** 19-35% improvement per adapter proves the adapters are functional -- the null transfer result is not from broken adapters.
4. **Reuses data from prior experiment.** Consistent methodology.

### Weaknesses

1. **Convex interpolation confound** (detailed above). Not fatal but weakens the quantitative interpretation.

2. **200 training iterations is short.** The paper acknowledges this. At 200 steps with batch=1 and seq_len=256, each adapter sees ~51K tokens. For a 2B-parameter model, adapters may not have developed rich enough representations to exhibit cross-domain features. However, they do achieve 19-35% PPL improvement, suggesting they have learned something meaningful.

3. **No additive composition control.** Testing `W_base + alpha * Delta_foreign + Delta_native` (no native downweighting) would isolate the question more cleanly.

4. **5 domains only, manually chosen.** The paper acknowledges this. With python/javascript or medical/biology, transfer might emerge between highly related domains.

5. **Not registered in HYPOTHESES.yml.** This experiment has no node in the hypothesis graph. While the kill criteria are stated in the paper, the experiment was not pre-registered in the project's standard tracking system.

### Does This Test What It Claims?

Partially. It tests whether convex blending of two Grassmannian-orthogonal adapters improves native domain PPL. It does NOT cleanly test whether one adapter's knowledge helps another domain, because the blending simultaneously dilutes the native adapter. The conclusion ("adapters are independent modules") is directionally correct but the evidence pathway has a confound.

### Could a Simpler Explanation Exist?

Yes. The null result may be entirely explained by the convex interpolation design: replacing 10-50% of a domain-specialized adapter with a foreign adapter trivially hurts performance because you are removing domain-relevant signal. This does not require invoking Grassmannian orthogonality at all -- the same result would hold for random LoRA adapters on an FP16 base.

**This is the most important critique.** The paper attributes the null result to "Grassmannian orthogonality works too well" (Section 1 of Key Insights). But the experiment does not include a control with NON-orthogonal adapters to test this explanation. If non-orthogonal adapters also show zero pairwise transfer under convex blending, the orthogonality explanation is wrong. Without this control, the causal claim is unsupported.

## Hypothesis Graph Consistency

The experiment is **not registered in HYPOTHESES.yml**. This means:
- No pre-registered kill criteria in the graph
- No dependency tracking
- No blocking relationships

The experiment references prior work (Grassmannian framework, OSRM diagnostic, per-token routing) but its relationship to the hypothesis graph is informal.

## Integration Risk

Low. This is a killed experiment with no downstream dependencies. The conclusion (routing, not blending, is the correct composition mechanism) is consistent with the existing architecture in VISION.md and with the per-token routing results.

## Macro-Scale Risks (advisory)

Not applicable -- experiment was killed.

However, the interpretation that "1/N composition benefit comes from regularization, not transfer" has implications for macro routing design. If true, it suggests the router should use hard top-1 selection rather than soft blending. The per-token routing experiment (which found top-2 beats top-1 and uniform) partially contradicts this, but that experiment used sequence-level majority voting, not weight-space blending. The distinction between output-space routing and weight-space composition remains unresolved.

## Verdict

**PROCEED** (accept the kill)

The experiment correctly identifies that pairwise weight-space blending of Grassmannian-orthogonal adapters does not produce cross-domain transfer. The kill is clean and the 0/20 result is unambiguous. The experiment advances the project's understanding by disambiguating "constructive transfer" (from prior composition results) as regularization.

**Two caveats the paper should note but that do not change the verdict:**

1. The convex interpolation design conflates "adding foreign signal" with "diluting native signal." An additive composition test (native at full strength + small foreign contribution) would be the proper test. The null result under convex blending is necessary but not sufficient to prove adapters are "truly independent" -- it could simply be that removing 10% of domain signal costs more than any foreign contribution provides.

2. The causal attribution to Grassmannian orthogonality ("works too well") is unsupported without a non-orthogonal control. The same null result might hold for any adapter pair under convex blending. This weakens Section 1 of Key Insights but does not change the kill.

The kill is valid. The interpretation needs qualification.

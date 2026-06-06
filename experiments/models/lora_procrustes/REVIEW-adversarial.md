# Peer Review: LoRA Procrustes Linear Decomposition

## NotebookLM Findings

Unable to complete automated NotebookLM deep review due to authentication
requirements. The following review is conducted manually with equivalent rigor,
informed by close reading of MATH.md, PAPER.md, the implementation
(`lora_procrustes.py`, `test_lora_procrustes.py`), the model definition,
HYPOTHESES.yml, REFERENCES.yml, FINDINGS.md, and VISION.md.

---

## Mathematical Soundness

### What holds

**The linearity argument is correct and trivially so.** The core claim in
MATH.md Section 2 is:

```
(W_base + dW_shared + dW_unique_k) @ x = (W_base + dW_k) @ x
```

This follows directly from `dW_shared + dW_unique_k = dW_k` by construction
(mean plus residual). Matrix addition distributes over multiplication. There is
nothing to prove here -- it is an identity, not a theorem. The paper correctly
identifies this.

**The N=2 algebraic identity is correctly derived.** Section 7 of MATH.md shows
that for routing weights (w_A, w_B) summing to 1:

```
shared + w_A * unique_A + w_B * unique_B = w_A * dW_A + w_B * dW_B
```

This is verified step-by-step and correct. The paper draws the correct
conclusion: decomposition adds zero information at N=2.

**The norm decomposition in Section 5 is correct.** The cross-term cancellation
when summing over domains (sum_k <shared, unique_k> = 0) follows from
sum_k unique_k = 0, which is true by construction.

**The ReLU subtlety (Section 6) is correctly identified.** The paper correctly
notes that while dW decomposition is exact, the full MLP has ReLU between fc1
and fc2, so routing operates on full expert MLP outputs, not on decomposed
weight additions. Both RoutedDeltaGPT and DecomposedDeltaGPT correctly
implement per-expert MLP forward passes (lines 268-274, 390-393).

### What does not hold (or is misleading)

**The shared fraction metric is ill-defined for its stated purpose.** The
shared fraction is computed as:

```
f_shared = ||dW_shared||_total / (||dW_shared||_total + ||dW_unique||_total)
```

This uses L2 norms (not squared norms) in a ratio. This is not a variance
decomposition. For a proper Pythagorean decomposition, you would need squared
norms:

```
sum_k ||dW_k||^2 = N * ||dW_shared||^2 + sum_k ||dW_unique_k||^2
```

The paper derives this in Section 5, but then uses the unsquared ratio for the
kill criterion. The 50% figure is therefore not "half the variance is shared"
but rather a ratio of norms that is harder to interpret. With N=2 and
near-orthogonal deltas of roughly equal magnitude, both the squared and
unsquared versions will give approximately 50%, so the result is not wrong in
practice, but the metric choice is imprecise.

**The "50% shared fraction is expected at N=2" admission undermines the kill
criterion.** The paper itself explains (Analysis Section 5) that with
near-orthogonal deltas at N=2, the shared fraction approaches 50% geometrically.
This means the >10% kill criterion is trivially satisfied and provides no
evidence that there is "meaningful shared structure." A random pair of
orthogonal vectors also has a mean with 50% of their combined norm. The test
is not discriminating.

---

## Novelty Assessment

### Prior art

**Task arithmetic (Ilharco et al., ICLR 2023)** already defines task vectors as
fine-tuned_weights - pretrained_weights and composes them by addition. The
"shared delta = mean of deltas" in this experiment is literally task arithmetic
with lambda = 1/N. The paper acknowledges this at N=2 (Section 3 of Analysis).

**TIES-Merging and DARE** are referenced but not compared against. The paper
references them correctly but only as "complementary" techniques. At N=2 with
near-zero cosine similarity, TIES and DARE would reduce to simple averaging
anyway, so this is acceptable for micro scale.

**LoRAHub (Huang et al., 2023)** composes LoRA adapters with learned
coefficients, which is functionally similar to the routing approach here. Not
referenced.

### Delta over existing work

The genuine contribution is the explicit contrast with the killed Procrustes
experiment (exp3): ReLU breaks weight-space decomposition, but LoRA deltas
avoid this because the delta path has no activation. This is a valid
architectural insight, even though the math is trivial. The experiment confirms
that the mechanism failed in exp3 for the right reason (nonlinearity), not
because decomposition is fundamentally flawed.

---

## Experimental Design

### Critical flaw: N=2 makes the experiment tautological

The paper itself identifies this as a limitation and the core of the analysis,
so this is not a surprise finding. But it must be stated plainly: **the
experiment does not test the hypothesis it claims to test.**

The hypothesis is: "LoRA deltas can be decomposed into shared + unique
components." At N=2, the decomposition is an algebraic identity. Decomposed
routing is mathematically identical to concatenated routing for any pair of
routing weights summing to 1. The kill criterion "decomposed >3% worse than
concatenated" cannot fail at N=2 because the two models are the same model
expressed in different coordinates.

The fact that empirical results show +0.0% gap is not evidence that
decomposition works -- it is evidence that the implementation is not buggy.

### Controls are adequate for what they test

The experiment correctly includes:
- Joint training baseline
- Task arithmetic (confirms it equals shared-only at N=2)
- Concatenated with calibrated and uniform routing
- Decomposed with calibrated and uniform routing
- Linearity verification (numerical exactness check)
- 3-seed aggregation

These are well-designed controls. The problem is not the controls but the
N=2 setting in which they operate.

### The linearity verification is the real contribution

The max output diff of <1e-05 between (base + shared + unique_k) and
(base + delta_k) numerically confirms that the decomposition is exact in
function space. This is the one result that could not be predicted from the
math alone (floating point issues could theoretically matter). It validates
the implementation.

---

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_lora_procrustes_linear` lists:
- Kill criteria: decomposed >3% worse than concat; shared <10% of norm
- Status: validated
- Evidence includes the N=2 triviality caveat

The status "validated" is generous. The kill criteria were tested and passed,
but as argued above, they could not have failed at N=2. A more honest status
would be "mechanism confirmed, not yet tested" since the non-trivial test
(N >= 3) has not been run.

The `blocks: [exp_lora_merging_bakeoff]` relationship is correct -- you should
confirm decomposition works before comparing merging strategies.

---

## Macro-Scale Risks (advisory)

1. **Shared fraction will likely decrease with N and domain diversity.** With
   5+ domains from distinct distributions, the mean delta could be small
   relative to individual deltas (the "committee of experts" problem). The
   10% threshold becomes a real test.

2. **Sign conflicts emerge at scale.** The near-zero cosine at micro scale
   (character-level, similar distributions) may not hold at macro scale with
   BPE tokens and diverse domains. TIES-style conflict resolution may become
   necessary before mean decomposition.

3. **Routing unique deltas is no cheaper than routing full deltas.** Both
   RoutedDeltaGPT and DecomposedDeltaGPT run N expert MLP forward passes
   per layer. The decomposition saves zero compute at inference. The only
   benefit is that shared knowledge is always active (robustness to routing
   errors), which is untested.

4. **The always-on shared component could hurt.** If the shared mean is
   non-trivial but points in a suboptimal direction for some tokens, baking
   it into the base weights introduces a bias that per-token routing cannot
   undo.

---

## Verdict

**PROCEED** -- with the explicit understanding that this experiment validated
the mechanism (linearity of LoRA delta decomposition) but did not test the
hypothesis (that decomposition provides value over concatenation). The N >= 3
follow-up is the real experiment.

The math is sound (trivially), the code is correct, the controls are adequate,
and the paper is unusually honest about the N=2 limitation. The experiment
achieves what it set out to do: confirm that the failure mode of exp3 (ReLU
breaking decomposition) is resolved by operating on LoRA deltas instead of
capsule groups.

### Required before claiming this line of research "works":

1. Run N >= 3 (ideally N=5 quintary split) decomposition and re-test both
   kill criteria. This is when the shared fraction and decomposed-vs-concat
   gap become meaningful.

2. When running N >= 3, also test whether decomposed+uniform routing is more
   robust than concat+uniform. The hypothesized benefit of always-on shared
   knowledge is only testable when routing errors matter (N > 2).

3. Consider replacing the shared fraction metric with a squared-norm ratio
   (variance decomposition) for clearer interpretation.

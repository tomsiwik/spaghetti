# Peer Review: adapter_specialization_emergence

## NotebookLM Findings

Skipped -- the experiment is a clean negative result with straightforward math. A deep review would not surface issues beyond what follows.

## Mathematical Soundness

### Gradient decorrelation derivation (MATH.md Section 2): Correct but misleading

The derivation that `E[dB_i^T dB_j] = A_i^T Sigma_x A_j * C` is correct under the factorization assumption that `E[(dL/dy)^2]` separates from `E[x x^T]`. This is only exact when the loss gradient and input are independent, which they are not in general. However, the directional conclusion holds: orthogonal A matrices reduce cross-correlation of B gradients.

The critical error is in the *interpretation*. The paper argues decorrelated gradients should produce different B matrices. This is wrong. Decorrelated gradients in orthogonal subspaces produce *equivalent functions*. The key identity:

```
delta_W_i = B_i @ A_i^T
```

Even though B_i and B_j are different matrices (they live in rotated subspaces), the *effect* on the output `delta_W_i @ x` can be identical in the loss-relevant directions. The paper's own root cause analysis (PAPER.md, point 2) correctly identifies this: "the optimization finds the same basin rotated into each adapter's subspace." But MATH.md does not derive this -- it should have.

### Missing analysis: subspace equivalence

MATH.md should have included the following argument that predicts the null result:

Given any optimal `B* @ A_0^T`, there exists a `B_i = B* @ A_0^T @ A_i / ||A_i||^2` (pseudo-inverse) such that `B_i @ A_i^T = B* @ A_0^T` when `A_i` has full column rank. More precisely, since all A_i span r-dimensional subspaces of the same d-dimensional space, the rank-r update `delta_W = B @ A^T` can represent the same rank-r perturbation regardless of which orthonormal A is used, because B is free to compensate. The optimizer finds this equivalent solution.

This was foreseeable from the math. The hypothesis required that different A projections *constrain* B to learn different functions, but B has `r * d_out` free parameters and the target perturbation is rank-r, so B can always rotate to match.

### JL lemma invocation: Correct but irrelevant

The JL lemma argument (that random projections approximately preserve inner products) is technically correct but does not support the specialization hypothesis. JL guarantees that `A_i^T x` preserves distance structure of the input -- meaning each adapter sees an approximately faithful low-dimensional view. But *faithful* means *equivalent*, not *different*. If all views are faithful, all adapters learn the same function.

FlyLoRA used JL to argue that frozen-A matches trained-A *quality*, not that different frozen-A produce different *specializations*. The paper correctly notes this distinction in LEARNINGS.md but the MATH.md framing is misleading.

### Silhouette computation: Correct

The silhouette implementation is standard (euclidean distance on z-scored PPL profiles, manual computation matching sklearn). The degenerate case (all adapters in one cluster) correctly returns 0.0.

### Kill criteria: Appropriate

K1 (silhouette >= 0.2) is a reasonable threshold. A silhouette of 0.2 would indicate weak but non-random clustering. The result of exactly 0.0 is maximally decisive -- no ambiguity.

## Novelty Assessment

### Prior art that already implies this result

1. **MoE self-specialization literature (Shazeer 2017, Fedus 2022):** The MoE community established that gating gradients are the mechanism for expert specialization. The paper acknowledges this but frames the experiment as testing whether projection geometry alone suffices. The answer was predictable: no gating = no specialization.

2. **SOLE's own findings (VISION.md, "Killed" section):** The EigenLoRAx experiment already showed "Grassmannian A-matrices prevent shared subspace. Orthogonality enables composition but prevents cross-adapter transfer." This finding directly implies that orthogonality isolates rather than differentiates.

3. **exp_cross_adapter_knowledge_transfer (KILLED):** 0/20 pairwise transfers >2% already showed adapters trained on different domains have no cross-adapter signal. If domain-trained adapters cannot share signal, mixed-trained adapters certainly cannot develop different signals.

The hypothesis was reasonable to test explicitly given the FlyLoRA framing, but the result was strongly predicted by 2-3 existing findings in the project's own evidence base.

### Delta over existing work

The experiment cleanly demonstrates that FlyLoRA's "implicit feature selection" does not extend to "implicit specialization." This is a useful clarification of the FlyLoRA mechanism. The quantitative evidence (10x10 PPL matrix with CV 0.1-0.4%) is thorough and the negative result is maximally clean.

## Experimental Design

### Does it test the hypothesis? Yes, cleanly.

The experiment isolates exactly the variable claimed: same data, same hyperparameters, different Grassmannian A matrices. The 10x10 PPL matrix is the correct measurement. The controls are:

- **Base model PPL** -- establishes that adapters do learn (mean -30% improvement)
- **Domain-trained adapters** -- establishes the achievable specialization level
- **Multiple metrics** (silhouette, entropy, CV, unique best domains) -- redundant confirmation

### Could a simpler mechanism explain the null result?

The paper's root cause analysis identifies four contributing factors. The dominant one is clear: without gating gradients, there is no competition mechanism. The other three (identical data ordering, short training, low capacity) are secondary.

One additional control would have strengthened the paper: **train 10 adapters with the same A matrix on different domain subsets.** This would confirm that specialization requires different data, not different A matrices. The domain-trained baselines partially serve this role but use the full 24-adapter skeleton rather than a single repeated A.

### Data ordering concern

All 10 adapters see the same 500 samples in the same order. The paper acknowledges this but dismisses it. This is actually important: even in MoE with gating, identical data ordering would reduce specialization variance. However, given that the loss trajectories differ by < 0.01 across all 200 steps, different shuffles would produce at most epsilon differences in B matrices. The paper's dismissal is justified by the data.

### Scale concern (within micro constraints)

200 steps on 500 samples is low, but the loss convergence data (identical trajectories) makes a compelling argument that more steps would not help. The paper correctly notes this in Limitations. This is an acceptable micro-scale choice.

## Hypothesis Graph Consistency

The experiment correctly tests K1 (silhouette >= 0.2) and the result is unambiguous. The kill decision is correct. The LEARNINGS.md correctly updates the hypothesis graph: Grassmannian A = interference prevention only.

This finding is consistent with and reinforces multiple prior killed experiments (EigenLoRAx, cross-adapter transfer, binary routing collapse). The project's evidence base now strongly converges on: orthogonality is for composition safety, not for specialization.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment was killed. No macro follow-up needed.

If someone revisited this with competitive gating (as suggested in Alternative Approaches), the macro risk would be: gating during training creates a chicken-and-egg problem where the router must specialize before adapters do, which is exactly the MoE training instability problem that load-balancing losses address. This is a well-studied problem with known solutions, not novel research.

## Verdict

**PROCEED** (accept the kill)

The experiment is a well-executed negative result. The kill decision is correct and maximally clean (silhouette = 0.0, not borderline). The root cause analysis is sound. The implications for SOLE are correctly drawn.

**Minor revisions (non-blocking):**

1. **MATH.md should derive the null prediction.** The subspace equivalence argument (B compensates for any orthonormal A to produce the same rank-r delta_W) should appear in Section 3 ("What Breaks It") with explicit notation. Currently, the mathematical framework predicts specialization but the experiment refutes it -- the math should have been able to predict the null.

2. **Clarify FlyLoRA distinction.** MATH.md Section 2 frames FlyLoRA as evidence FOR the hypothesis. The LEARNINGS.md correctly notes that FlyLoRA's claim is about quality equivalence, not specialization. MATH.md should make this distinction upfront rather than in the post-mortem.

3. **Missing control:** A single A matrix trained on 10 different domain subsets would have cleanly separated "A matters" from "data matters" in one experiment. This is a minor design improvement, not a flaw -- the domain-trained baselines serve a similar role.

These are documentation improvements. The experimental result, kill decision, and implications are all correct. The finding that Grassmannian A matrices are purely a composition safety mechanism (not a specialization inducer) is now well-established by convergent evidence from multiple experiments.

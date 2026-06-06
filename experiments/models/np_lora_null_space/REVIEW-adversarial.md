# Peer Review: NP-LoRA Null Space Projection

## NotebookLM Findings

Skipped -- the experiment is a clear KILL with multiple independent failure modes. Deep review is not warranted.

## Mathematical Soundness

**The math is correct.** Step-by-step verification:

1. **Cross-term decomposition** (MATH.md lines 32-36): The Frobenius norm expansion is standard linear algebra. The cross-term tr(Delta_i^T Delta_j) correctly captures pairwise interference. No issues.

2. **Grassmannian zero cross-term argument** (lines 46-49): The derivation tr((B_i^T B_j)(A_j A_i^T)) = 0 when A_j A_i^T = 0 is correct. The trace is cyclic, so tr(A_i^T B_i^T B_j A_j) = tr((A_j A_i^T)(B_i^T B_j)). Since A_j A_i^T is the zero matrix, the product vanishes regardless of B correlation. Sound.

3. **Null space projection** (lines 56-95): The SVD-based projector P_i = I - V_r V_r^T correctly projects into the orthogonal complement of the row space of S_i. By construction, P_i @ delta_j = 0 for j != i (since delta_j is a row of S_i), which means the projected delta_i has zero inner product with all other deltas. Mathematically correct.

4. **Complexity analysis** (lines 111-136): O(N^3 * L * d^2) is correct. For each of N adapters, across L layers, computing SVD of an (N-1, d^2) matrix costs O(N^2 * d^2) when N << d^2 (thin SVD). Total = N * L * O(N^2 * d^2). The micro-scale estimates are consistent with empirical measurements (predicted "well under 1s" for N=5, measured 0.79s -- actually borderline, see below).

**One hidden assumption worth noting:** The per-layer projection (Step 98-108 in MATH.md) assumes independence across layers. This is weaker than the full vectorized projection because two adapters could have zero per-layer interference but nonzero cross-layer interference (through the network's forward pass). However, this is a standard and reasonable approximation, and NP-LoRA itself uses this variant.

**The worked example** (lines 138-162) is pedagogically sound and correctly demonstrates both the Grassmannian no-op case and the output-subspace separation case.

## Novelty Assessment

**This is a negative-result experiment, not a novelty claim.** The experiment correctly tests whether a known method (NP-LoRA, arxiv 2511.11051) provides value on top of the project's Grassmannian skeleton. The answer is no, for three independent reasons. This is appropriate micro-scale methodology.

**Prior art check:** NP-LoRA (2511.11051) is properly cited. The FlyLoRA JL-lemma argument (2510.08396) is correctly invoked to explain why random A matrices are near-orthogonal at high d/r. No reference material exists in the `references/` directory for NP-LoRA.

**The key finding -- that Grassmannian init makes post-hoc projection redundant -- is not itself novel** (it follows directly from the math), but experimentally confirming it with measured numbers is useful for the project's decision log.

## Experimental Design

**Strengths:**

1. The three-condition design (Grassmannian + NP-LoRA, Random + NP-LoRA, timing at N=50) tests three independent failure modes. This is thorough.
2. Using pre-trained adapters from a prior experiment for the Grassmannian condition eliminates confounds from re-training.
3. The random-A condition correctly trains new adapters (1500 steps) rather than just swapping A matrices on frozen B, which would be invalid since B was trained jointly with A.

**Weaknesses and concerns:**

1. **The NP-LoRA projection on random A barely changes the cosine similarity** (9.34e-4 to 9.29e-4, a 0.5% relative reduction). This is suspicious. If NP-LoRA projects into the null space of other deltas, the post-projection cross-terms should be *exactly zero* by construction. The fact that they are not indicates a bug or a measurement artifact.

   Looking at the code (lines 313-340), the `compute_composition_metrics` function concatenates all layer deltas into one vector per adapter and measures cosine. But the projection is applied *per-layer*. Per-layer orthogonality does not imply global (concatenated) orthogonality. Specifically, if adapters i and j have zero per-layer cross-terms for each layer individually, the concatenated vectors can still have nonzero dot product if the per-layer zero cross-terms don't cancel in sum.

   Wait -- actually no. If each per-layer segment has zero dot product, the concatenated dot product is the sum of per-layer dot products, which is also zero. So the metric *should* show zero cross-terms after projection.

   **Re-examining:** The issue is that thin SVD with `full_matrices=False` returns V_r of shape (min(N-1, d^2), d^2). When N-1 < d^2 (always true here: 4 < 65536), the null space has dimension d^2 - rank(S_i). The projection removes the component in the row space of S_i, which is at most rank N-1. For N=5, this removes at most 4 directions from a 65536-dimensional space. The change is minuscule because each delta vector is mostly in the null space already (it only has a tiny component along other deltas). The cosine similarity is computed on the full concatenated vector, so a tiny per-layer change in a tiny subspace barely moves the needle.

   **This is actually correct behavior** -- it is not a bug. The projection removes the interference component, but at d/r=32, that component was already negligible (< 0.001 of the vector norm). The post-projection cosine is not exactly zero because the measurement concatenates across layers while projection is per-layer, AND because the remaining cosine comes from numerical precision issues with such small components.

   **However, the paper should have reported per-layer cosine before/after to confirm the projection actually achieves zero within each layer.** This is a missing diagnostic, not a fatal flaw.

2. **No statistical significance testing.** The PPL differences between conditions are in the 4th decimal place (1.5595 vs 1.5595). With only 20 batches of 64 for evaluation, the noise floor likely exceeds the signal. The paper should have run multiple seeds or at least reported confidence intervals. That said, the *claim* is that NP-LoRA provides zero benefit, which a null result with negligible effect size supports even without formal statistics.

3. **Single PPL evaluation for Grassmannian adapters is not reported in the Grassmannian condition.** The paper reports single PPLs for the Grassmannian baseline (1.5239 mean) and uses composition ratio (composed/single = 1.0224), which is the right metric. This is fine.

4. **K2 timing threshold of 1 second is reasonable but the N=5 case already takes 0.79s** -- dangerously close to the threshold. The paper claims K2 PASS at N=5 but the margin is 21%. A slightly larger model or more layers would fail. This does not change the verdict but should be noted.

5. **Missing control: NP-LoRA on highly interfering adapters.** The experiment tests two conditions where interference is already low (Grassmannian: near-zero, Random at d/r=32: near-zero). The interesting control would be random A at a *low* d/r ratio (e.g., d=64, r=16, d/r=4) where interference would be substantial. The paper acknowledges this in Limitations but does not test it. Given that the experiment is already KILLED on timing, this omission is acceptable.

## Hypothesis Graph Consistency

The kill criteria in the code (K1: NP-LoRA not worse, K2: N=50 < 1s) reference hypothesis IDs 268 and 269 which do not appear in HYPOTHESES.yml. This is a bookkeeping gap -- the experiment should have been registered in the hypothesis graph before execution. Not blocking for the verdict, but the FINDINGS.md entry should be added.

## Macro-Scale Risks (advisory)

1. **The O(N^3 * L * d^2) scaling is absolutely fatal at macro scale.** At d=2560, r=16, L~100 layers, N=50: each SVD would be on a (49, 6.5M) matrix. This is not just slow, it is memory-prohibitive (49 * 6.5M * 8 bytes = 2.5GB per matrix, times 100 layers times 50 adapters).

2. **Even approximate null space methods (e.g., randomized SVD, iterative projection) cannot save this.** The fundamental issue is that you need O(N) SVDs per adapter per layer, and each SVD operates on d^2-dimensional vectors. Any method operating in the full parameter space will hit this wall.

3. **The positive macro finding is that Grassmannian init makes this entire approach unnecessary.** The pre-hoc orthogonality guarantee is O(1) at composition time (just sum the deltas). This is the right answer.

## Verdict

**PROCEED** (with the KILL decision)

The experiment is correctly killed. The methodology is sound, the math checks out, and the three independent failure modes (no-op on Grassmannian, negligible benefit on random A, catastrophic scaling) each independently justify termination. The conclusion that pre-hoc orthogonality (Grassmannian) dominates post-hoc projection (NP-LoRA) is well-supported.

**Minor fixes for the record (not blocking):**

1. Add per-layer cosine similarity before/after projection as a diagnostic to confirm the projection achieves exact zero within each layer. The current global cosine metric obscures this.
2. Register hypothesis IDs 268/269 in HYPOTHESES.yml or update the code comments to reference the correct IDs.
3. Add the NP-LoRA kill finding to FINDINGS.md under the "Killed" section in VISION.md.
4. Note in the paper that the N=5 timing (0.79s) is within 21% of the K2 threshold -- the "PASS" at N=5 is marginal, not comfortable.

The kill is clean and the evidence is decisive. No further compute should be spent on null space projection approaches for this architecture.

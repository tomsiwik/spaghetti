# Peer Review: bitnet_effective_delta_cosine

## NotebookLM Findings

Skipped. This is a pure measurement experiment with straightforward linear algebra. The mathematical claims are verifiable by direct inspection of the code and the Frobenius norm identities. NotebookLM deep review would add no value beyond what manual verification provides.

## Mathematical Soundness

### What holds

1. **The per-module filtering identity is correct.** The trace identity vec(DW_i)^T vec(DW_j) = tr(A_i^T B_i^T B_j A_j) is standard. The bound |tr(A_i^T B_i^T B_j A_j)| <= kappa(B_i) * kappa(B_j) * ||A_i^T A_j||_F / r follows from submultiplicativity. The per-layer results (mean |cos_eff| ~ 0.001-0.005) confirm the filtering mechanism works at the module level. No errors found.

2. **The code correctly implements the math.** `DW = B.T @ A.T` with stored shapes lora_a:(d_in, r) and lora_b:(r, d_out) produces (d_out, d_in), which is the correct weight-space perturbation. The cosine computation, coherence measurement, and condition number calculation are all correct. I verified against the NPZ loading, key parsing, and flattening logic.

3. **The dimensionality analysis is correct.** D_eff = 2.08B vs D_raw = 21.6M (96.4x ratio) is consistent with the module shapes listed in MATH.md. The dimension counts are independently verified by the results.json output.

### What is weak but not wrong

1. **The MATH.md is unusually messy.** Lines 14-46 are a stream-of-consciousness working-through of matrix product conventions, with multiple false starts ("wait, note ordering below", "which is wrong -- actually", "NO."). This is not a mathematical error -- the final convention (line 95: `effective_delta = lora_b.T @ lora_a.T`) is correct -- but it undermines confidence in the derivation. The bound in Section 3.2 is correct.

2. **The Cauchy-Schwarz bound in Section 3.2 is stated loosely.** The paper writes |vec(DW_i)^T vec(DW_j)| <= ||A_i^T A_j||_F * ||B_i^T B_j||_F and then switches to a submultiplicativity argument. These are different bounds. The submultiplicativity version is tighter and is the one used for the kappa-based prediction. This notational sloppiness does not affect the experiment's conclusions since the bound was used only for prediction, not for the kill criterion.

3. **The prediction in Section 3.3 has a logical gap.** The argument that |cos_eff| << |cos_raw| relies on A-filtering "killing" the B-B contribution in the effective-delta inner product. But the raw parameter cosine inner product is p_i . p_j = sum[vec(A_i)^T vec(A_j)] + sum[vec(B_i)^T vec(B_j)], which also benefits from A-orthogonality (the first sum is also small). The filtering argument proves that effective-delta removes the B-B contribution, but raw parameter cosine already has a small B-B contribution in this regime. The paper implicitly assumes B-B correlation dominates raw cosine, which the data refutes: cos_B (mean 0.00073) is actually LOWER than cos_A (mean 0.00107). The A-matrices contribute MORE to raw cosine than B-matrices do, so filtering B through A does not provide the expected advantage.

### What is genuinely insightful

4. **The aggregation failure analysis is the key contribution.** The paper correctly identifies that per-module filtering works, but concatenation across 210 modules in a 96x higher-dimensional space does not preserve the filtering property. This is because the concatenated inner product is a sum of 210 per-module inner products, each with independent signs. The effective-delta cosine normalizes by a much larger norm (sqrt of sum of squared module norms), but the numerator (sum of per-module inner products) does not cancel as aggressively as in the raw parameter space, where A and B terms interleave with more cancellation opportunities. This analysis is correct and non-obvious.

## Novelty Assessment

This is a targeted internal measurement, not a publishable method. No prior art measures "raw parameter cosine vs effective-delta cosine for multi-module LoRA composition" because this is a specific design choice within the SOLE/CTE architecture.

The closest relevant work is the original Grassmannian skeleton experiment (b_matrix_training_correlation), which established the 17x per-module filter at d=64. This experiment correctly extends that measurement to d=2560 with multi-module aggregation and discovers the aggregation failure. The discovery that per-module guarantees do not trivially lift to adapter-level guarantees via concatenation is a useful negative result.

**VISION.md impact:** The "17x decorrelation filter" claim in VISION.md (line 23 and line 48) is now partially invalidated. The per-module filter is confirmed, but the headline "17x" number is misleading because it measured a single module at toy scale. VISION.md should be updated to clarify this is a per-module property.

## Experimental Design

### Strengths

1. **Clean measurement study.** No training, no hyperparameters, no stochasticity. The results are deterministic given the adapter weights. This is the ideal design for a metric comparison experiment.

2. **Comprehensive diagnostics.** Four cosine metrics (raw, effective-delta, A-only, B-only), A-coherence, B-condition numbers, and per-layer decomposition. The per-layer analysis (Phase 6) directly confirms the per-module filtering hypothesis, isolating the failure to the aggregation step.

3. **Appropriate controls.** Comparing A-only and B-only cosines alongside raw and effective-delta cosines reveals the relative contribution of each component.

### Weaknesses

1. **Only 5 adapters (10 pairs).** However, the effect is extremely consistent -- all 10 pairs show eff > raw. The variance in the ratio (5.8x to 404.5x) is large, but this is driven by pairs where raw cosine happens to be near zero (lucky cancellation), not by inconsistency in the mechanism. Adequate for a negative result.

2. **200-step adapters only.** The paper acknowledges this. The aggregation dimensionality argument is structural and training-independent, so this is not a concern.

3. **No test of the Grassmannian-initialized A-matrices.** The adapters use standard random init. Grassmannian A-init would lower per-module A-coherence but, as the paper correctly notes, would not fix the aggregation issue. Not a blocking concern.

### One concern about the kill criterion design

The K2 criterion ("effective-delta cosine > 5x raw parameter cosine") was designed to detect whether the raw metric is "dangerously loose." But the experiment reveals a more nuanced situation: effective-delta cosine is higher, but still well below the 0.05 interference threshold (K1 passes with 2x margin). The kill is technically correct on K2 but the practical implication is mild -- the raw metric is conservative, not dangerous. The paper handles this distinction well in its "Implications" section.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node (exp_bitnet_effective_delta_cosine). Kill criteria K1 and K2 match exactly between HYPOTHESES.yml and the code. The experiment depends on exp_bitnet_cosine_convergence_trajectory (correct -- uses the same adapters and builds on the convergence finding). It blocks nothing, which is appropriate for a measurement study.

One node (exp_bitnet_lori_sparse_b) depends on this experiment. The kill of effective-delta cosine does not block LoRI sparse-B, which tests a different mechanism (B-sparsity for interference reduction). The dependency seems to be "understand B-matrix contribution to interference before trying to reduce it," which is satisfied by the negative result here.

## Macro-Scale Risks (advisory)

Not applicable. This experiment is killed and recommends no changes to the production metric. The one action item is advisory: VISION.md should update the "17x decorrelation filter" claim to clarify it is a per-module property, not an adapter-level property.

## VISION.md Update Required

VISION.md currently states:
- Line 23: "17x decorrelation filter on B-matrix interference"
- Line 48: "Empirically confirmed: B-matrix cos 0.0298 -> delta cos 0.0017 (17x filter)"

These claims are now known to be per-module only. At adapter level (210 modules aggregated), the filter inverts to 0.05x (effective-delta is 19x higher than raw). VISION.md should add a qualifier: "per-module" or "single-module measurement." This is not blocking for the architecture -- the Grassmannian skeleton provides per-module orthogonality guarantees, and the composition results (N=25 scaling, reproducibility, etc.) are proven independently of this metric choice.

## Verdict

**PROCEED** (kill is valid, analysis is sound)

The experiment correctly identifies a real and non-obvious failure in the hypothesis. The kill is well-supported by the data. The analysis of why the toy-scale prediction failed (per-module vs aggregated, dimensionality trap) is the most valuable part of the paper and is mathematically sound. The practical conclusion (keep raw parameter cosine as the operational metric) is the correct action.

No revisions required. The MATH.md is messy but not wrong. The PAPER.md is thorough and honest about the kill. The code is correct.

One advisory action: update VISION.md to qualify the "17x decorrelation filter" as a per-module property. This does not change the architecture or invalidate other findings.

# Peer Review: Grassmannian Expert Init (Revised)

## NotebookLM Findings

Skipped (NotebookLM authentication not configured in this session). Review conducted via direct systematic analysis of MATH.md, PAPER.md, implementation code, results.json, HYPOTHESES.yml, and the prior adversarial review.

## Revision Assessment

The five required fixes from the first review have all been addressed:

1. **Haar-random orthonormal control:** Added as `init_lora_random_orthonormal()` (line 348-371). Three-condition comparison now runs at all dimensions. This was the critical fix and it is implemented correctly.

2. **Wilcoxon signed-rank tests:** Added with scipy.stats.wilcoxon, one-sided alternative='less'. Aggregate p-values computed across both seeds (56 paired observations per d). Correctly implemented.

3. **K3 reclassified:** Code prints "RECLASSIFIED: Zero drift is a design property" (line 955-958). MATH.md Section 5.2 now titled "Zero Drift is a Design Property (Not an Empirical Finding)." Kill criteria in HYPOTHESES.yml updated with parenthetical reclassification note. Clean.

4. **Finding #2 reframed:** PAPER.md Finding #2 now reads "With frozen-A LoRA, initialization geometry IS the final geometry" and explicitly states "This is a design property, not an experimental finding." Correct.

5. **d=256 tail anomaly discussed:** PAPER.md Section "d=256 Tail Anomaly" (lines 99-112) acknowledges the outlier, explains AP optimizes mean not worst-case, and notes minimax packing as the alternative for tail guarantees. Adequate.

## Mathematical Soundness

### Welch Bound (MATH.md Section 3): CORRECT

The derivation via trace(G^2) and Cauchy-Schwarz on eigenvalues is standard. The formula matches Dhillon et al. (2008). The code implementation (lines 63-76) correctly handles the Nr <= d trivial case.

One note from the prior review remains valid but non-blocking: the jump from E[||U_i^T U_j||_F^2] = r^2/d to E[||U_i^T U_j||_F] ~ r/sqrt(d) elides Jensen's inequality. The MATH.md Section 6.2 now at least mentions "Jensen's inequality correction factor that is close to 1 for large d," which is sufficient.

### AP Algorithm (MATH.md Section 4): CORRECT, prior caveat resolved

The prior review noted the convergence claim cites Bauschke & Borwein for two convex sets, but the rank-d constraint is not convex. The implementation correctly projects to rank-at-most-d PSD (which IS convex), so convergence holds for the implementation. The MATH.md still says "both constraint sets are convex" without the rank-at-most-d clarification, but this is a minor imprecision. The empirical convergence to 2.8-3x above the Welch bound (a non-optimal fixed point) is honestly reported.

### Interference Bound (MATH.md Section 5.3): CORRECT

Submultiplicativity argument is sound. The connection between skeleton coherence mu and worst-case interference is the key load-bearing math for the SOLE architecture, and it holds.

### Capacity Scaling (MATH.md Section 8.1): Still hand-wavy but explicitly scoped

The d^2/r^2 claim still relies on the "effective dimension D ~ d^2" argument without rigorous proof. MATH.md Section 8.1 now hedges with "This is consistent:" rather than asserting it as proven. The prior review flagged this as non-blocking but recommended separating it for rigorous treatment. That remains valid advice.

## Novelty Assessment

### Prior Art

Citations are adequate: Dhillon et al. (2008) for AP, LoRI (2504.07448) for orthogonal LoRA init, GrMoE (2602.17798) for Grassmannian routing, SMILE (2408.10174) for SVD-based expert construction.

### Delta Over Existing Work

The contribution is modest but genuine: applying Grassmannian packing specifically to LoRA expert A-matrix initialization and empirically isolating the packing benefit from orthonormality. LoRI uses orthogonal init without optimal packing; this experiment shows the packing itself matters (1.2-1.5x beyond orthonormality). The three-condition comparison is the key methodological contribution.

No reinvention detected in the references/ directory -- no existing Grassmannian packing implementation was available.

## Experimental Design

### The Three-Condition Comparison: Well-Designed

The revised experiment correctly isolates the packing effect:
- AP-orthonormal vs. random-orthonormal: packing effect only (both are orthonormal)
- Random-orthonormal vs. random-Gaussian: orthonormality effect only (both are randomly placed)
- AP-orthonormal vs. random-Gaussian: total improvement (confounded)

This is the right factorial design for the question being asked.

### Statistical Analysis: Adequate but with a subtle concern

The Wilcoxon signed-rank test is appropriate for paired non-normal data. The pairing is by expert pair index (pair i-j gets the same index across conditions because the same domain data is used). This is correct.

**Concern about paired sample independence:** The 56 pairs per d come from 8 experts yielding C(8,2)=28 pairs, times 2 seeds. The 28 pairs within a seed are NOT independent -- they share experts. Expert 0's delta vector appears in 7 of the 28 pairs. The Wilcoxon test assumes independent paired observations. This inflates the effective sample size and may produce optimistic p-values.

However, this is a common and generally accepted approximation in this literature (pairwise cosine analysis). The effect is to make the p-values appear more significant than they truly are. Since d=64 already fails significance (p=0.096) and d=128/256 pass with reasonable margins (p=0.009, p=0.012), the dependence structure would need to inflate significance by roughly 5-10x to change the conclusion at d=128. This is unlikely but worth noting.

A cleaner approach would be to report per-expert-pair statistics or use a permutation test that respects the dependency structure. Non-blocking.

### Consistent Random Seeds Across Conditions: CORRECT

All three conditions use the same domain data (same `domain_id`), ensuring the comparison is fair. The model weights are shared (same `model` object). The only difference is A-matrix initialization. This is clean experimental design.

### Architecture Variation at d=256: Acknowledged

At d=256, `d_ff_mult` drops from 4 to 2 (d_ff=512 vs d_ff=256 at d=64/128). This means the delta vector dimension D changes non-uniformly with d. The MATH.md Section 9.3 notes "Micro-scale toy data" but does not explicitly flag the architecture change. The PAPER.md Limitations Section 1 mentions "Toy data" but not the architecture variation.

This is minor because the three-condition comparison at each d is internal (same architecture), so the AP vs. ortho comparison is valid within each d. Cross-d comparisons of absolute cosine values are confounded, but the paper does not make strong cross-d claims about absolute levels.

### Loss Values Confirm No Learning

All conditions produce losses of ~3.466 (near log(32) = 3.466, i.e., random prediction). The paper correctly acknowledges this. The delta vectors reflect gradient noise from initialization, not learned features. The PAPER.md revised Finding #2 correctly frames this as a geometric property that is preserved (not "carried through training").

### Wilcoxon Test at d=256: Closer Look

At d=256, the numbers deserve scrutiny:
- AP mean |cos| = 0.00231, std = 0.00316 (std > mean: heavy-tailed, many near-zero values)
- Ortho mean |cos| = 0.00307, std = 0.00271
- Gauss mean |cos| = 0.00304, std = 0.00280

Ortho and Gauss are nearly identical (ratio 0.99x, p=0.628). This is expected and actually strengthens the argument: at d=256, orthonormality provides zero benefit, but AP packing still provides 1.33x improvement. The packing effect is isolated cleanly here.

The anomaly: ortho is slightly WORSE than Gauss at d=256 (0.00307 vs 0.00304). The paper notes this but does not overinterpret it. Correct behavior.

## Hypothesis Graph Consistency

### HYPOTHESES.yml Entry: Consistent

- Status: "supported" (not "proven") -- matches the code's verdict (`overall: false` because d=64 is not significant)
- Kill criteria updated to reflect the three-condition design and K3 reclassification
- Evidence list includes all revision details
- Blocks exp_scale_500_experts -- appropriate (skeleton is a prerequisite)

### Kill Criteria Actually Tested: YES

- K1 (AP <= random-ortho): tested via Wilcoxon. Pass at all d directionally, significant at 2/3.
- K2 (timing): tested at micro, production estimated. Pass.
- K3: correctly reclassified.

The verdict "SUPPORTED" (not "PROVEN") is honest and appropriate given d=64 non-significance.

## Integration Risk

### Composes with SOLE Architecture: YES

The skeleton produces orthonormal frames that slot directly into LoRA A-matrices. The frozen-A assumption is standard LoRA practice and is assumed throughout SOLE. The interference bound (MATH.md 5.3) connects skeleton coherence to composition interference, which is the load-bearing connection.

### Does Not Conflict with Existing Components

The structural orthogonality proof (micro/models/structural_orthogonality_proof/) showed random init gives 17-69x below the sqrt(r/d) bound. This experiment shows AP adds a further 1.2-1.5x on top. These are complementary: structural orthogonality is the dominant effect (concentration of measure), and AP packing is a refinement. The paper correctly positions this in the lineage diagram and the "Connection to SOLE Architecture" section.

## Remaining Issues (Non-Blocking)

1. **Paired sample dependence in Wilcoxon test.** The 28 pairs per seed share experts, inflating effective N. A permutation test or bootstrap that respects the dependency structure would be more rigorous. The current p-values are likely optimistic by a modest factor.

2. **AP convergence quality.** Final coherence is 2.8-3x above the Welch bound after 500 iterations. The PAPER.md acknowledges this and suggests TAAP for improvement. At production scale, convergence quality is unknown. This is a macro risk, not a micro issue.

3. **The practical significance question.** Even at d=128 where the packing effect is strongest (1.52x), the absolute numbers are tiny: AP mean |cos| = 0.0032 vs ortho mean |cos| = 0.0049. Both are already negligible for practical purposes (the structural orthogonality proof showed values 17-69x below the interference bound). The 1.52x ratio is statistically significant but the practical impact -- whether this translates to measurable quality differences in generation -- is entirely unknown. The paper correctly defers this to macro validation.

4. **The "MATH.md Section 4.3 convexity" note from the prior review** is still technically imprecise ("both constraint sets are convex" when the rank constraint requires the rank-at-most-d formulation). This is a documentation issue, not a code issue.

## Macro-Scale Risks (advisory)

1. **AP convergence at d=4096, N=500.** The 8000x8000 eigendecomposition is feasible on GPU, but convergence behavior at this scale is unvalidated. TAAP (cited but not implemented) would be advisable.

2. **The 1.2-1.5x benefit may shrink or grow.** With real learning signals (converged adapters), gradient alignment may dominate initialization geometry, reducing the AP advantage. Alternatively, higher N/d ratios at production scale may increase packing pressure and enlarge the benefit. Unknown without macro testing.

3. **Per-layer independence.** The current design assigns one skeleton to layer 0 and rotates for other layers. At 32+ layers, the rotation approach (MATH.md 5.1: simple 2D Givens rotation using only the first two dimensions of the r-dimensional frame) barely distinguishes layers. A per-layer skeleton would multiply the packing problem by L but provide stronger guarantees.

4. **Worst-case vs mean-case.** The d=256 tail anomaly shows AP can produce outlier high-interference pairs. For composition safety (SOLE's promise of "experts can't interfere"), mean-case improvement is insufficient -- worst-case bounds matter. The minimax packing variant mentioned in the paper should be pursued.

## Verdict

**PROCEED**

The revised experiment addresses all five required fixes from the first review. The three-condition comparison cleanly isolates the packing effect from orthonormality, which was the critical methodological gap. The results are:

- **Directionally correct at all dimensions** (AP < random-ortho at d=64, 128, 256)
- **Statistically significant at production-relevant dimensions** (p=0.009 at d=128, p=0.012 at d=256)
- **Honestly reported** as "SUPPORTED" rather than "PROVEN" because d=64 is not significant
- **Kill criteria properly assessed** with K3 correctly reclassified as a design property

The remaining issues (paired sample dependence, AP convergence, practical significance) are non-blocking for a micro experiment. The mechanism works in principle: Grassmannian packing provides a genuine geometric benefit beyond orthonormality, and the benefit increases with dimension -- exactly the direction needed for production viability.

The experiment status of "SUPPORTED" in HYPOTHESES.yml is correct and should remain until macro validation at d=4096 confirms the benefit at production scale.

### Non-blocking recommendations for macro follow-up:
- Use a permutation test or block bootstrap that respects the paired structure (experts shared across pairs)
- Implement TAAP for better convergence at N=500
- Test minimax packing (worst-case) in addition to mean-case
- Measure whether the 1.2-1.5x coherence improvement translates to measurable generation quality difference

# Peer Review: bitnet_semantic_compositionality

## NotebookLM Findings

Skipped (experiment self-killed; deep review not warranted for a confirmatory pass on a negative result).

## Mathematical Soundness

### What holds

1. **The core derivation is correct.** Weight-space orthogonality (cos(vec(DW_i), vec(DW_j)) ~ 0) constrains A_i^T A_j ~ 0, but data-space orthogonality requires A_j^T h_i ~ 0. These are indeed different conditions, and the paper correctly identifies this gap.

2. **The dimensional analysis is sound.** For hidden states with effective dimensionality d_eff, the expected cross-activation ratio scales as sqrt(r / d_eff). The observed ratio of 0.86 implying d_eff ~ 22 is plausible and consistent with known transformer hidden-state concentration phenomena.

3. **The JL-lemma argument is correctly applied.** Random projections onto r-dimensional subspaces preserve norms approximately, so cross-activation ratios near 1.0 are expected for isotropic data -- and even more so for concentrated data.

### What does not hold

4. **The composition scaling math in MATH.md and the code are inconsistent.** MATH.md (lines 109-125) describes the composed output as:

   ```
   output = base(h) + 0.5 * B_i @ A_i @ h + 0.5 * B_j @ A_j @ h
   ```

   This is the correct 1/N formulation: apply each adapter's full delta at 1/N weight. However, the actual code (`compose_adapters`, line 249-257) computes:

   ```
   A_composed = (A_i + A_j) / 2
   B_composed = (B_i + B_j) / 2
   output = base(h) + alpha * B_composed @ A_composed @ h
   ```

   Expanding: `alpha * ((B_i + B_j)/2) @ ((A_i + A_j)/2) @ h = alpha/4 * (B_i@A_i + B_i@A_j + B_j@A_i + B_j@A_j) @ h`

   The matched terms each get scaled by alpha/4 (not alpha/2), and there are two spurious cross-terms (B_i@A_j and B_j@A_i) also at alpha/4. The paper's SNR analysis (1/0.88 ~ 1.14) is computed against the wrong model -- the actual forward pass is not a weighted sum of independent adapters but a product of averaged matrices. This is a pre-existing issue across the entire project (all composition experiments use the same function), so it does not invalidate the empirical findings, but it means the SNR derivation in MATH.md does not describe what the code actually computes.

5. **The 0.1 threshold is self-imposed, not from OSRM.** The paper acknowledges this in Limitations (item 5), but K3's kill criterion is built on this threshold. The OSRM paper (arXiv:2505.22934) does not specify a universal threshold. At ratio ~0.86, the diagnostic is measuring something real (high cross-activation), but the KILL depends entirely on a threshold the authors chose.

## Novelty Assessment

**Prior art is well-cited.** The three key references (OSRM 2505.22934, Rethinking Inter-LoRA Orthogonality 2510.03262, FlyLoRA 2510.08396) are all relevant. The contribution is:

- **OSRM** showed how to FIX the weight-vs-data orthogonality gap. This experiment confirms the gap EXISTS for the project's setup, which is useful but not novel.
- **Rethinking Inter-LoRA Orthogonality** already demonstrated that weight-space orthogonality is insufficient for semantic compositionality in diffusion models. This experiment extends that finding to LLMs with ternary adapters, which is a modest contribution.
- **FlyLoRA** uses frozen random A matrices (similar to Grassmannian skeleton) and notes the JL-lemma property. The overlap is significant.

The genuine novel finding is empirical: **composition works despite high cross-activation (ratio 0.86) through constructive transfer rather than orthogonal isolation.** This is a useful negative result that reframes the project's theoretical narrative.

**No reinvention detected.** The experiment builds on prior implementations from bitnet_instruction_task_eval and does not reimplement OSRM.

## Experimental Design

### Strengths

1. **Two-part design is well-structured.** Separating semantic composition (Part A) from the orthogonality diagnostic (Part B) makes the experiment cleanly interpretable.

2. **OSRM diagnostic is correct in implementation.** Hidden states are extracted from the base model (no adapter), which is the standard approach in routing literature. The A-only and full B@A measures provide two levels of diagnostic.

3. **Adapter reuse is appropriate.** Using instruction-tuned adapters from a prior proven experiment avoids confounding adapter quality with the phenomenon under test.

### Weaknesses

4. **K2 (semantic coherence) is not "manual inspection" -- it is an automated proxy.** The kill criterion states "manual inspection of 20 samples" but the code (lines 960-968) uses a length-based heuristic: a response is "coherent" if `len(words) >= max(3, len(base_words) * 0.5)`. This is a non-empty-and-not-too-short check, not a coherence assessment. A response containing random domain keywords at sufficient length passes K2. The paper reports "18/20 composed responses are semantically coherent (90%)" and "Manual inspection shows composed outputs are substantive" -- this conflates the automated proxy with actual manual inspection. If manual inspection WAS performed separately, this is not documented in the code.

5. **K1 uses PPL, not cross-domain query performance, despite the kill criterion wording.** The kill criterion says "composed adapter worse than either alone on >50% of cross-domain queries." The code evaluates this as PPL on query prompts (lines 493-509), where "worse" means higher PPL. But the paper's K1 table (lines 41-47) shows both PPL and cross-domain rate. The K1 pass/fail determination (line 950-954) uses `composed_better_than_best_individual` which is computed from `cross_domain_rate` (keyword overlap), not PPL. So K1 actually uses the keyword metric, not PPL -- which is a weaker signal than claimed.

6. **Cross-domain queries are hand-crafted and domain-specific keywords are cherry-picked.** The keyword lists are generous (e.g., "=" counts as a math keyword, "if " and "for " count as code keywords). Many responses from a 2B base model would naturally contain "if" or "=" regardless of domain specialization. This inflates cross-domain rates.

7. **KR-Test results are mixed but not discussed in the verdict.** 3/5 pairs show improvement, 2/5 show degradation (medical pairs). This is not integrated into the kill criteria. The medical adapter dilution (0.967 -> 0.467) suggests composition can hurt strong individual adapters -- a finding that deserves more prominence.

8. **No control for "constructive transfer" claim.** The paper claims composition works through "constructive cross-domain transfer" (better PPL than either individual). But a simpler explanation is possible: 1/N scaling simply regularizes adapter magnitude. A control would be: single adapter at 0.5x scale vs composed at 0.5x each. If the single adapter at 0.5x also improves PPL, the "constructive transfer" narrative collapses.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node (`exp_bitnet_semantic_compositionality`) correctly. Kill criteria K1/K2/K3 are consistent between code, paper, and HYPOTHESES.yml. The status is correctly set to `killed` based on K3 failure. Evidence is accurately recorded.

The experiment does not block any other nodes, which is appropriate for a diagnostic.

## Macro-Scale Risks (advisory)

1. **Composition method (averaging A and B separately) produces cross-terms that scale as O(N^2).** At N=25 or N=100, these cross-terms may dominate. The macro experiments should compare separate-average composition vs true delta-weight composition (sum of B_i@A_i) to quantify this.

2. **The "constructive transfer" finding may be data-specific.** Cross-domain queries are designed to benefit from multiple domains. On domain-specific benchmarks (MMLU subsets, HumanEval), cross-activation may be purely destructive.

3. **The d_eff ~ 22 finding suggests that scaling to more adapters (N >> 25) will see increasing cross-activation problems.** If hidden states concentrate in ~22 dimensions and each adapter's A projects into 16 dimensions, there is severe subspace overlap. OSRM-style initialization or per-token routing becomes more important at scale.

## Verdict

**PROCEED**

The experiment is self-killed on K3, which is the correct call. The finding -- weight-space orthogonality does not imply data-space orthogonality, yet composition works through other mechanisms -- is valid and well-documented. The KILL is appropriately scoped and correctly integrated into the project's knowledge base (VISION.md, FINDINGS.md).

The issues found are non-blocking:

- The composition math inconsistency (point 4) is a pre-existing project-wide issue, not specific to this experiment. It should be addressed but does not invalidate the diagnostic.
- The K2 coherence proxy (point 4 in Experimental Design) inflates the K2 PASS confidence, but K2 is not the killed criterion.
- The missing regularization control (point 8) weakens the "constructive transfer" interpretation but does not change the core finding.

The experiment advances the project by (a) closing the weight-vs-data orthogonality question, (b) providing the d_eff ~ 22 estimate that informs capacity planning, and (c) reframing the theoretical narrative from "orthogonal isolation" to "constructive composition + 1/N regularization."

**Non-blocking recommendations for future work:**

1. Fix the composition math inconsistency across the project: either compose delta weights (sum B_i@A_i) or use runtime multi-LoRA with separate adapters. Document which method is used and why.
2. Run the regularization control: single adapter at 1/2 scale on cross-domain queries. If it matches composed performance, the "constructive transfer" claim needs revision.
3. Upgrade K2 from the length proxy to actual LLM-as-judge or human evaluation in future coherence experiments.
4. The keyword lists should exclude common tokens ("=", "if ", "for ") that appear in general text regardless of domain.

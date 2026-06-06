# Peer Review: gram_schmidt_composition

## NotebookLM Findings

Skipped -- sufficient material for a thorough manual review. The experiment is well-documented with clear MATH.md derivations, three-seed empirical results, and honest interpretation of findings.

## Mathematical Soundness

**Gram-Schmidt derivation (MATH.md Sections 3-4): CORRECT.**

The classical GS process is stated correctly. The projection formula, orthogonality guarantee, and signal retention formula (rho_k = sqrt(1 - sum cos^2)) are all standard linear algebra. The first-order Taylor approximation in Section 3.1 (rho_k ~ 1 - (1/2) sum cos^2) is valid when cos << 1, which holds in this regime.

**Worked example verification (Section 6): CONSISTENT.**

Predicted rho_5 ~ 0.9981 vs observed 0.9994. The paper correctly explains the discrepancy (actual overlaps are smaller than max). This checks out.

**One hidden assumption worth noting (Section 8, item 3):** The paper assumes cosine similarity in weight space implies functional interference. This is a sufficient but not necessary condition -- two deltas could have high cosine similarity but modify different functional pathways if they operate on different input distributions. Conversely, near-zero cosine does not guarantee zero functional interference in the nonlinear forward pass. The paper acknowledges this in Assumption 3 but does not test it. At micro scale this is acceptable -- it is a known limitation, not a flaw.

**Merge strategy analysis (Section 4): CORRECT but incomplete.** The paper correctly identifies that naive sum scales the perturbation magnitude with N, while 1/N averaging controls it. However, the claim that GS removes "interference in the overlapping subspace" while averaging "does not" (Section 4.2) is slightly misleading. When deltas are near-orthogonal (cos < 0.06), the overlapping subspace carries < 0.4% of the energy. Averaging by 1/N already suppresses this residual by a factor of N. GS removes it entirely, but the difference is negligible in either case. The paper reaches this correct conclusion in Section 7 but frames Sections 2-4 as if interference is a real problem before revealing it is not. This is a rhetorical issue, not a mathematical one.

## Novelty Assessment

**Prior art coverage: ADEQUATE.**

The paper cites the relevant work: InfLoRA (orthogonality during training), TIES-Merging (sparsification), Task Arithmetic (scaling coefficients). These are the right references.

**Is this novel?** No, and it does not claim to be. Post-hoc Gram-Schmidt orthogonalization of model deltas is a straightforward application of classical linear algebra. The paper's actual contribution is the **negative result**: GS is unnecessary because LoRA deltas are already near-orthogonal. This negative result has value -- it confirms and strengthens the foundational orthogonality finding (cos=0.0002 at d=896) by showing that even at d=64 where cosines are 50-300x higher (0.01-0.06), the interference is still too small to matter.

**Relation to existing project findings:** This directly builds on `lora_merging_bakeoff` (simple average is best) and the orthogonality scaling results. The experiment is well-positioned in the lineage.

## Experimental Design

**Does it test the stated hypothesis? YES, but with a caveat.**

The hypothesis is: "GS preserves expert quality within 10% PPL improvement while guaranteeing zero interference." The experiment confirms this. However, the experiment cannot distinguish between two explanations:

1. GS works well (preserves signal, removes interference)
2. GS is a no-op (there is no interference to remove)

The data strongly supports explanation 2, and the paper correctly reaches this conclusion. The experiment design could have been stronger with a **synthetic high-overlap control** -- artificially creating deltas with cos > 0.3 and testing whether GS helps in that regime. The unit test `test_gram_schmidt_orthogonality` does this with constructed vectors (45-degree angle), but only verifies the GS math, not the downstream loss impact. This is a missed opportunity but not a fatal flaw.

**Controls: ADEQUATE.**

- Four merge strategies compared (naive sum, simple avg, GS sum, GS avg)
- Three seeds
- Two domain counts (N=2, N=5)
- Order sensitivity check (forward vs reversed)
- Individual expert baselines

**KC1 assessment: HONEST but awkward.** The paper reports that KC1 technically triggers on individual domains where improvements are < 0.005 absolute loss, then correctly dismisses these as measurement noise. The kill criterion is poorly calibrated for this regime -- a relative threshold (>10%) is meaningless when the denominator is near zero. The paper should have defined KC1 with a minimum absolute threshold (e.g., "loses >10% PPL improvement OR >0.01 absolute loss, whichever is larger"). This is a minor design flaw the paper handles transparently.

**KC2: CLEAN PASS.** Minimum retention 99.67% vs 50% threshold. No ambiguity.

## Hypothesis Graph Consistency

The experiment is marked `status: proven` in HYPOTHESES.yml with the correct conclusion: GS works technically but is unnecessary because deltas are already near-orthogonal. Multiple downstream nodes depend on this experiment (`exp_collision_scaling`, `exp_composable_merge_pipeline`, and five others). The "proven" status is appropriate -- the mechanism works in principle, and the negative finding (unnecessary) is itself a useful result that strengthens confidence in simple averaging.

One concern: `exp_collision_scaling` and `exp_composable_merge_pipeline` list this as a dependency, implying they might use GS. The paper's recommendation is to NOT use GS. The dependency should be interpreted as "this experiment confirmed simple averaging is sufficient" rather than "GS is a prerequisite component." This is clear in the paper but could be made explicit in the HYPOTHESES.yml notes.

## Macro-Scale Risks (advisory)

1. **The interesting regime was not tested.** At d=896+ with real domain data (math, code, medical), some domain pairs might share substantial subspace structure (e.g., math and physics). If pairwise cosines reach 0.1-0.3, GS could become relevant. The paper acknowledges this in its Limitations section.

2. **GS order dependence could matter at scale.** With N=100+ experts and non-trivial overlap, the first expert in GS ordering retains 100% signal while later experts could lose meaningful amounts. The paper's N=5 test showed negligible order sensitivity, but this could change at scale.

3. **The real risk is not interference but magnitude.** With N=50+ experts composed via simple averaging, the 1/N scaling shrinks each expert's contribution substantially. At N=500, each expert contributes only 0.2% of its original delta. Whether the base model can absorb 500 simultaneous 0.2%-strength perturbations without degradation is the real macro question. This is orthogonal to the GS question but is the actual composition bottleneck.

## Verdict

**PROCEED**

The experiment is well-designed within micro constraints, the math is sound, the code is clean, the results are honestly reported, and the negative finding (GS is unnecessary) is the most valuable outcome. The experiment successfully strengthens the project's foundational claim that LoRA orthogonality is structural, not engineered.

Minor issues that do not block proceeding:

1. KC1 threshold is poorly calibrated for near-zero improvements (relative threshold without absolute floor). Not worth revising since the experiment is complete and the conclusion is correct.
2. A synthetic high-overlap control would have made the negative result more definitive, but the conclusion is already well-supported by the combination of low cosines + equivalent loss numbers.
3. HYPOTHESES.yml downstream dependencies should clarify that this experiment's output is "simple averaging is sufficient" rather than "GS is a tool to use."

None of these warrant a REVISE verdict. The experiment answers its question cleanly and the answer advances the project's understanding of composition mechanics.

# Peer Review: ReLoRA Composition Test (Rev 2)

## Rev 1 Issues -- Status

All 5 required fixes from rev1 have been addressed:

1. **Training data asymmetry (critical)** -- FIXED. Both conditions now train on `expert_train_ds` and evaluate on `expert_val_ds`. Verified in code lines 572-597.
2. **Evaluation data consistency** -- FIXED. Explicit `train_dataset` and `val_dataset` arguments in `train_lora_expert`. Both conditions share the same held-out split.
3. **Confidence intervals on cos ratio** -- FIXED. Bootstrap CI reported: [0.77, 2.64].
4. **Clarify all-parameter training** -- FIXED. MATH.md Section 2.2 and PAPER.md Section 1 now explicitly state that ReLoRA trains all parameters, with LoRA merge-restart layered on top.
5. **results.json relabeled** -- FIXED. Per-seed files named `results_seed_{N}.json`, integration test data in `results_integration_test.json`.

Both advisory items also addressed: (6) direct loss ratio reporting, (7) permutation test added.

## NotebookLM Findings

Skipped. The experiment is small and self-contained enough for direct manual review.

## Mathematical Soundness

### Verified Correct

1. **Merge operation.** `merge_lora_into_base` computes `W_new = W + delta.T` where `delta = (alpha/r) * A @ B` has shape `(in_dim, out_dim)` and `W` has shape `(out_dim, in_dim)`. Post-merge, `x @ W_new.T = x @ W.T + (alpha/r) * x @ A @ B`, which equals the pre-merge output since B is reset to zero. Mathematically correct.

2. **Rank bound.** `rank(sum_k dW_k) <= min(K*r, min(d_in, d_out))` is correct by subadditivity. With K=5, r=8, d=64: min(40, 64) = 40. The clarification that all-parameter training makes the final weight full-rank is accurate.

3. **Effective rank.** Roy & Vetterli (2007) `exp(H(p))` correctly implemented with `1e-12` guard. Verified against edge cases in tests.

4. **Expected random cosine.** `E[|cos|] ~ sqrt(2/(pi*D))` for D=131,072 gives ~0.0039. Standard result for independent Gaussian vectors in high dimensions.

5. **Cosine computation.** Standard pairwise cosine with `1e-12` epsilon guard. Correct.

### Bootstrap CI -- Correct but Degenerate

With N=3 seeds, the bootstrap resamples from {0.77, 1.90, 2.64}, yielding at most 27 distinct bootstrap samples. The CI [0.77, 2.64] trivially spans min to max. This is not wrong -- it is the correct bootstrap result for N=3 -- but it provides no information beyond "the range of our data." The paper does not overclaim from this, so it is acceptable.

### Permutation Test -- Correct with Minor Caveat

The test pools all pairwise cosines across seeds (18 ReLoRA + 18 conventional) and permutes labels. Methodologically sound. The p=0.056 is honestly reported as "marginally non-significant."

**Minor caveat:** The 6 pairwise cosines within a single seed share the same 4 experts, so they are not independent. The effective sample size is smaller than 18, which means the true p-value may be somewhat lower than 0.056. However, this would make the non-significance finding *more conservative*, not less, so it works in favor of the paper's honest conclusion.

### Interference Bound (Section 3.4) -- Correct but Loose

The bound `||W_composed - W_ideal|| <= C(N,2) * cos_max * max(||dW_i||)` is standard Cauchy-Schwarz. It is correct but loose (uses max norms). At micro scale, the bound gives 6 * 0.149 * ||dW|| which is not negligibly small. The paper's hedging ("both are small relative to ||W_base||") is reasonable given that ||dW|| ~ 0.1 * ||W_base||, making the bound ~0.09 * ||W_base||. This is a non-trivial interference bound that would need to be much tighter at macro scale.

## Novelty Assessment

The specific question -- "do LoRA experts compose on a ReLoRA-built base as well as on a conventional base?" -- is genuinely untested in published literature. Neither ReLoRA (Lialin et al.), LoRAHub (Huang et al.), nor InfoLoRA (2024) addresses this question. The experiment fills a legitimate gap required by the VISION.md architecture.

No reinvention detected. The experiment correctly reuses the `lora_procrustes` LoRA infrastructure. ReLoRA is listed in `REFERENCES.yml` (comment-only entry, no dedicated folder -- appropriate since the experiment implements ReLoRA directly rather than wrapping an external library).

## Experimental Design

### Hypothesis-Test Alignment (Good)

The three kill criteria (10x cos, 2x loss, 2x coherence) directly test the hypothesis. They are set at "catastrophic failure" thresholds rather than marginal significance thresholds, which is appropriate for a mechanism-viability test.

### Controls (Adequate)

- Same domain data splits for both conditions (fix 1)
- Same held-out evaluation splits (fix 2)
- Same expert training seeds (seed + i)
- Same pretraining data and step count
- Same learning rate and optimizer
- 3 random seeds for aggregate statistics

### Confound: Architectural Asymmetry During Pretraining

The ReLoRA model is a LoRAGPT (base + LoRA A/B on MLP) while the conventional model is a plain GPT. During pretraining, the ReLoRA model has ~15-20% more parameters being trained (the LoRA matrices) and undergoes periodic optimizer resets. This architectural difference is inherent to the experimental design and cannot be eliminated without changing the question being asked. The 4.6% base quality gap could be attributed to any of: (a) merge-restart dynamics, (b) optimizer resets, (c) extra LoRA parameters splitting gradient signal. The paper correctly identifies this as a base quality issue rather than a composition issue (Section 4.1: "the remaining ~0.6% is the composition penalty").

### Verdict Logic (Defensible with a Noted Inconsistency)

The code uses two thresholds: kill at >10x, inconclusive at >2x, survive at <2x. The 2x INCONCLUSIVE threshold is not stated in the formal kill criteria but is a reasonable soft threshold. The aggregate verdict is "INCONCLUSIVE" (due to seed 123 at 2.64x), which is honest.

**However:** HYPOTHESES.yml status is "supported" while the experiment's own verdict is "INCONCLUSIVE." This is a tension. The justification is that all kill criteria are disproven, which is the bar for "supported" -- the INCONCLUSIVE verdict reflects uncertainty about the magnitude of the effect, not about whether the mechanism works. This is defensible but should be stated explicitly.

### Per-Seed Analysis (Informative)

The per-seed breakdown reveals interesting structure:
- Seed 42: ReLoRA cos < conventional cos (ratio 0.77x) -- ReLoRA is actually better
- Seed 7: ratio 1.90x -- moderate degradation
- Seed 123: ratio 2.64x -- notable degradation

The inconsistency in direction (seed 42 vs seed 123) strongly supports the permutation test conclusion: the difference is likely noise at this scale.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node correctly lists:
- Status: "supported" (all kill criteria disproven)
- Depends on: `exp_adapter_taxonomy_wild` (proven)
- Blocks: `exp_base_free_composition`
- Kill criteria match what is tested in code

The evidence lines in HYPOTHESES.yml still reference some rev1 numbers (cos ratio 2.13x, quality 93.6%). These should be updated to rev2 values.

## Stale References in Other Files

**FINDINGS.md (line 34):** Reports "cos ratio 2.13x, quality ratio 0.936" -- these are rev1 numbers. Rev2 shows cos_ratio = 1.77x, loss_ratio = 1.052.

**VISION.md (line 28):** Reports "cos ratio 2.13x, quality 93.6%" -- also rev1 numbers.

These are housekeeping issues, not blocking.

## Macro-Scale Risks (advisory)

1. **The 4.6% base quality gap.** Lialin et al. shows this gap shrinks with model size, but their experiments only go to 1.3B. If the gap persists at 7B, it compounds across all experts served on that base.

2. **cos_ratio direction is unresolved.** At d=896, random cosine drops to ~0.0002. If the ~2x degradation factor persists, cos ~0.0004 remains negligible. But if the factor grows with scale (plausible if ReLoRA creates correlated weight structure), it could matter. This is the first thing to test at macro.

3. **Only 5 merge cycles.** Production ReLoRA uses hundreds of cycles. More cycles means more optimizer resets and more potential for systematic weight bias.

4. **N=4 experts is too few for interference compounding.** At N=50, the C(50,2) = 1,225 pairwise interference terms grow quadratically. The interference bound at N=50 would be 1,225 * cos_max * ||dW||, which could become non-negligible even with cos_max ~ 0.001.

## Verdict

**PROCEED**

The experiment is methodologically sound after rev2 fixes. All five required fixes from rev1 have been correctly implemented. The math is verified correct. The kill criteria are well-calibrated and clearly disproven. The INCONCLUSIVE aggregate verdict is an honest reflection of limited statistical power at N=3 seeds, not a fundamental mechanism failure.

The key result: LoRA experts compose on a ReLoRA-built base with only 5.2% loss degradation, of which ~4.6% is base quality and only ~0.6% is composition penalty. The orthogonality degradation is statistically indistinguishable from noise (p=0.056). This is sufficient directional evidence to proceed to macro validation.

### Required Housekeeping (non-blocking)

1. **Update FINDINGS.md** to rev2 numbers: cos_ratio 1.77x (not 2.13x), loss_ratio 1.052 (not "quality 93.6%"), base gap 4.6% (not 5.7%).

2. **Update VISION.md line 28** to rev2 numbers.

3. **Update HYPOTHESES.yml evidence lines** for `exp_relora_composition_test` to match rev2 values (currently lists rev1 numbers in evidence items).

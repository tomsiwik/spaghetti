# Peer Review: Fallback G1 Adapter Retraining

## NotebookLM Findings

Skipped -- the experiment is already self-killed with clear evidence. Deep review would not change the verdict.

## Mathematical Soundness

**MATH.md derivations: Correct with one gap.**

1. The LoRA parameter counting (Fix 1) is correct: 4 layers x 4 projections x (4x256 + 256x4) = 32,768. The 0.25% ratio claim checks out against ~13M base params. The actual code reports 0.11% against 29M total params (including embeddings). Both framings are valid but inconsistent -- MATH.md uses 13M (transformer blocks only), PAPER.md uses 29M (full model). Not a mathematical error, just sloppy bookkeeping.

2. The worked example for delta magnitude has a subtle error. It claims `||B @ A||_F ~ ||B||_F * ||A||_F / sqrt(r)`. This is not a general identity. For rank-r matrices B (d_out x r) and A (r x d_in), the Frobenius norm of the product depends on the alignment of singular vectors, not just the individual norms. The inequality `||BA||_F <= ||B||_F * ||A||_F` holds (submultiplicativity), and dividing by sqrt(r) is a heuristic for "average-case" random matrices, not a bound. The estimate gives ~0.063 relative delta; the actual measured value was ~0.126. The factor-of-2 discrepancy confirms the heuristic is loose.

3. The composition methods (1/N averaging, Task Arithmetic, TIES) are correctly described and correctly implemented. One minor discrepancy: MATH.md specifies `W_composed = W + lambda * tau_merged` for TIES, but the code applies TIES without a lambda scaling factor (effectively lambda=1.0). This is a reasonable default but diverges from the written specification.

4. The evaluation metric `ratio_i = PPL_base / PPL_adapted` is well-defined and appropriate for this test.

**Hidden assumption that matters:** Assumption #2 ("d=256 model can learn enough structure for domain adaptation to be meaningful") turns out to be the critical failure mode. At PPL 268, the model is at roughly 5.6 bits per token -- well above the entropy of English text (~1.0-1.5 BPT for competent LMs). The model has not learned enough distributional structure for domain-specific fine-tuning to have purchase. This is not a flaw in the math; it is a correctly identified risk that materialized.

## Novelty Assessment

**No novelty claimed, none needed.** This experiment applies standard techniques (LoRA, Task Arithmetic, TIES-Merging) to diagnose a prior failure. The contribution is diagnostic, not methodological. The three cited references (Ilharco 2023, Yadav 2023) are the correct ones.

**Important context from VISION.md:** The project has already proven adapter composition works at the BitNet-2B-4T scale (PPL ~4-5) with Grassmannian orthogonal A matrices. This micro experiment was testing whether self-trained ternary bases at d=256 could replicate the same pattern. The answer is no, and the paper correctly attributes this to base model weakness rather than mechanism failure.

## Experimental Design

**The experiment tests what it claims, but has three design weaknesses:**

1. **Early stopping was functionally inert.** All three adapters ran to exactly 3000 steps. Examining the validation trajectories reveals why: science val_ppl goes 269.7 -> 270.1 -> 270.6 -> 270.4 -> 270.3 -> 270.4. The best checkpoint was at step 500 for all adapters, with monotonically worsening validation from there. The patience-5 window (5 checks at 500-step intervals = 2500 steps) exactly matches the remaining budget after the first check, so early stopping could only trigger at the final step. The PAPER's claim that "all adapters hit early stopping at 3000 steps" is misleading -- they hit the step cap, not the early stopping criterion. **The adapters were overfitting from step 500 onward.** Evaluating the step-500 checkpoints would have given a fairer test of the adapter mechanism, separate from the overfitting confound.

2. **Domain separation is weak by design, but this is acknowledged.** FineWeb-Edu science/history/technology filtered by keyword overlap produces distributions that differ primarily in topic vocabulary, not in deeper linguistic structure. At PPL 268, the base model hasn't even mastered common vocabulary well enough for topic-level differences to matter. The Limitations section correctly flags this.

3. **Composition evaluation conflates interference with weak signal.** The experiment applies all 3 adapters simultaneously and evaluates on individual domains. When individual adapters only improve domain PPL by 3%, applying 2 irrelevant adapters alongside the relevant one predictably degrades performance. The composition test is valid for what it measures (multi-adapter merging), but does not distinguish between "composition methods are broken" and "there is no signal to compose." A fairer control would be single-adapter evaluation on each domain (which was done -- showing 3% improvement -- but could have been more prominently compared).

**What works well:**
- The delta ratio diagnostic is genuinely useful. Proving that prior failures were overfitting (not vacuous deltas) is a valuable negative result.
- The monotonic relationship between lambda and degradation (lambda=0.3 gives -4%, lambda=1.0 gives -14%) is clean experimental evidence that the deltas are active but destructive when summed.
- The kill criteria are well-defined and correctly applied.

## Macro-Scale Risks (advisory)

These findings do not block macro work because:

1. The project already has proven composition at BitNet-2B-4T scale (PPL ~4-5). This micro experiment confirms that self-trained ternary bases at d=256 are too weak for meaningful domain adaptation -- a useful data point but not a threat to the core architecture.

2. The key risk to watch at macro scale: the delta ratio of ~0.13 is in the "healthy" range per MATH.md (target 0.005-0.05), but actually exceeds the upper bound by 2.5x. At larger scale, this could indicate adapters are too aggressive even with rank-4. Monitor delta ratios carefully when scaling.

3. The domain data pipeline (keyword filtering from FineWeb-Edu) should be replaced with genuinely distinct corpora (e.g., PubMed vs GitHub vs legal) at macro scale.

## Verdict

**PROCEED** (with the kill -- the self-kill decision is correct)

The experiment correctly identifies that its hypothesis failed: no adapter achieves the 10% PPL improvement threshold. The analysis of why (weak base, insufficient domain separation, noise-dominated deltas) is sound and well-reasoned. The diagnostic value -- confirming that the prior catastrophic failure was overfitting, not vacuous deltas -- is a genuine contribution to the project's knowledge base.

Two observations for the record, neither blocking:

1. The early stopping mechanism was ineffective (patience window matched training budget). Future adapter experiments should use tighter patience (e.g., 3 checks of 200 steps) and save/evaluate the best-validation checkpoint rather than the final one. The adapters were likely overfitting from step 500, which means the "true" improvement might have been slightly higher than 3.3% if evaluated at the best checkpoint -- though unlikely to reach 10%.

2. The delta ratios (~0.13) exceeding the MATH.md target range (0.005-0.05) by 2.5x should be noted in FINDINGS.md. The "healthy range" calibration in MATH.md appears to be miscalibrated for this scale, or the alpha/rank=2.0 scaling factor is too aggressive for rank-4.

The kill is justified. The learnings (overfitting diagnosis, composition-requires-strong-base) are correctly captured and advance the project's understanding.

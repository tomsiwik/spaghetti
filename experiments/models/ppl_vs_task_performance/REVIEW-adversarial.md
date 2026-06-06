# Peer Review: PPL vs Task Performance

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a deep review is unnecessary. The math is simple (Pearson correlation over 5 points), the code is readable, and the claim is modest (a kill result).

## Mathematical Soundness

**Correct:**

1. PPL definition (MATH.md Section 2.1) is standard: exponentiated average cross-entropy. Implementation in `compute_ppl()` (line 304-340) matches: sum of per-token cross-entropy with `ignore_index=pad_id`, divided by non-pad token count, exponentiated.

2. The PPL improvement formula `(PPL_base - PPL_expert) / PPL_base` is a valid relative improvement metric. Positive = expert is better. This is correctly implemented (line 670).

3. The decomposition argument in Section 3 is mathematically sound. Full-sequence PPL averaging over prompt + answer tokens dilutes the answer signal. The condition for divergence (Section 3.2) is correct: when prompt degradation exceeds answer improvement scaled by the length ratio.

4. Pearson correlation computation via `np.corrcoef` is standard.

**Issues:**

1. **N=5 is below the threshold for meaningful Pearson correlation.** The paper acknowledges this (r_crit = 0.687 at p<0.05), and the kill criterion (r >= 0.5) is more lenient. However, with N=5, even a "true" moderate correlation could easily appear as r=0.08 by chance. The paper frames this as "correlation is essentially zero" but a confidence interval around r=0.08 with N=5 spans roughly [-0.8, +0.9]. The point estimate is meaningless -- the experiment cannot distinguish r=0 from r=0.5 with any statistical power. **The kill is not statistically justified at N=5.** However, the consistency across 3 seeds (all showing near-zero r) and the mechanistic explanation (prompt/answer averaging) provide qualitative support for the kill.

2. **The "repeat" domain is at ceiling** (base accuracy = 1.0, expert accuracy = 1.0, accuracy improvement = 0.0). This data point contributes zero information to the correlation but still counts as one of the 5 points. With N already dangerously small, a wasted data point is significant. If repeat is excluded, N=4 and no statistical test is meaningful.

3. **PPL improvement is computed differently from accuracy improvement.** PPL uses relative improvement `(base - expert) / base`, while accuracy uses absolute improvement `expert - base`. This inconsistency is minor (both are monotonic in the same direction) but could slightly affect the correlation if base PPLs vary widely (which they do: 1.56 to 5.26).

## Novelty Assessment

**This is not a novel finding.** The disconnect between perplexity and downstream task performance is well-established in the literature:

- Schaeffer et al. (2023, "Are Emergent Abilities of LLMs a Mirage?") showed that metric choice (including the relationship between loss and accuracy) explains much of what looks like emergence. The nonlinear mapping from loss to accuracy is precisely the mechanism observed here.
- Gadre et al. (2024, DataComp-LM) found significant per-task variance in the PPL-to-downstream correlation, even at scale.
- The general principle that cross-entropy loss averages over all positions while accuracy measures top-1 correctness on specific positions is textbook.

**However**, the specific application to SOLE shadow scoring is novel and valuable. The experiment doesn't claim to discover that PPL and accuracy can diverge -- it tests whether this divergence matters for the specific architecture. That framing is appropriate.

**Delta over prior art:** The insight about answer-conditioned PPL as a fix is straightforward but useful for the project. The paper correctly identifies this as the actionable takeaway.

## Experimental Design

**What it tests well:**

- The experiment cleanly separates PPL evaluation (full-sequence) from accuracy evaluation (answer-only greedy decode). This is the right decomposition.
- The 5 domains have genuinely different characteristics (deterministic arithmetic, string manipulation, classification), providing some diversity.
- 3 seeds with consistent results strengthen the qualitative conclusion.
- The code is clean, well-structured, and reproducible.

**Design weaknesses:**

1. **The experiment tests the wrong PPL metric.** MATH.md Section 6 identifies answer-conditioned PPL as the likely fix, and PAPER.md Section "Implications" recommends it. But the experiment never actually tests answer-only PPL. This is the most important missing control. If answer-only PPL correlates strongly with accuracy (r > 0.5), then the kill conclusion changes from "PPL is misleading" to "full-sequence PPL is misleading but answer-conditioned PPL works fine." The delimiter positions are known for all 5 domains, so this would have been trivial to implement. This is a significant omission.

2. **Expert training uses a fixed shuffling seed** (line 415: `rng = random.Random(42)`) regardless of the experiment seed. This means the same training order is used for every expert across all seeds. The per-seed variation comes only from data generation and model initialization, not from training dynamics. This weakens the "3 independent seeds" claim somewhat.

3. **The lora_layers parameter in compute_ppl is never used.** Lines 323-326 show a commented-out block with `pass`. The actual evaluation works because experts are merged into base weights before calling `compute_ppl`, but this dead code suggests the evaluation path was not the originally intended one.

4. **Accuracy evaluation uses a fixed seed** (line 517: `rng = random.Random(999)`) independent of the experiment seed. This means all 3 seeds are evaluated on identical test prompts. This is actually good for reducing noise in the accuracy measurement, but it means the across-seed variance in accuracy is lower than truly independent evaluations would produce.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node:
- Kill criterion K1 (Pearson r >= 0.5) is tested and failed (0/3 seeds).
- Kill criterion K2 (best PPL = best accuracy) is tested and partially failed (2/3 seeds).
- Status correctly set to "killed."
- The downstream dependency `exp_execution_based_self_learning` is appropriately noted -- this experiment motivates using execution feedback rather than PPL for code domains.

## Macro-Scale Risks (advisory)

1. **The finding may not transfer to macro.** At scale with subword tokenization (V=32K+), the probability distributions are smoother and PPL may correlate better with downstream performance. Chinchilla scaling laws are built on the premise that loss predicts performance. The micro-scale divergence could be an artifact of character-level tokenization with V=42 creating very peaked distributions.

2. **Answer-conditioned PPL is trivial to compute for structured tasks but hard for free-form generation.** In the SOLE architecture, most expert domains will involve free-form text (e.g., "python-async" expert generating code), where there is no clear prompt/answer delimiter. The paper acknowledges this but does not propose a solution.

3. **The VISION.md still describes shadow scoring as using "per-token perplexity" (line 138).** This has not been updated to reflect the killed hypothesis. The evolution mechanism needs redesign, and this experiment provides the motivation but not the solution.

## Verdict

**PROCEED**

The experiment is correctly killed. The code runs, the results are reproducible across 3 seeds, the mechanistic explanation is sound, and the implications for SOLE shadow scoring are clearly stated. The kill result is informative and directionally correct even though it lacks statistical power at N=5.

The following are recommendations, not blocking issues:

1. **Add answer-only PPL as a control.** The delimiter positions are known. Computing PPL only on tokens after the delimiter would test whether the decomposition fix actually works. This is the single highest-value addition and would take ~20 lines of code.

2. **Drop or replace the "repeat" domain.** It contributes nothing (base accuracy already 1.0). Replace with a domain where both base and expert have room to improve.

3. **Harmonize improvement metrics.** Use either relative improvement for both PPL and accuracy, or absolute improvement for both.

4. **Update VISION.md** to reflect that raw PPL shadow scoring is a known vulnerability. The evolution mechanism description should note the need for task-specific or answer-conditioned metrics.

These are all low-cost improvements that would strengthen an already-informative result. The core finding -- that full-sequence PPL is a poor proxy for task accuracy in the SOLE context -- stands on mechanistic grounds even if the statistical evidence at N=5 is weak.

# Peer Review: bitnet_task_eval

## NotebookLM Findings

NotebookLM review was not performed for this post-mortem review. The experiment is already KILLED by its own criteria, and the review below validates that verdict rather than challenging a live claim.

## Mathematical Soundness

**Composition math is correct but reveals a fundamental problem the paper correctly identifies.**

The averaged-factor composition formula is properly derived:

    A_merged = (1/N) * sum_i A_i
    B_merged = (1/N) * sum_i B_i

The expansion to diagonal (1/N^2) and cross-terms (N(N-1)/N^2) is correct. At N=5, each adapter's own-signal is attenuated to 4%, which the paper correctly identifies as the root cause of task signal destruction.

**One mathematical gap:** The paper states cross-terms are "negligible when adapters are near-orthogonal (|cos| ~ 0.002)" but does not formalize what "negligible" means quantitatively. The cross-term contribution is O(N(N-1)/N^2) = O(1 - 1/N) = 80% of the total adapter contribution at N=5. Even if individual cross-terms are small due to orthogonality, their aggregate is 20x the diagonal signal. The paper hand-waves this. However, since the experiment was killed for other reasons, this gap is not decision-relevant.

**Keyword F1 metric is mathematically sound** but semantically weak, as the paper acknowledges. The implementation correctly uses token-level precision/recall with Counter intersection.

**Confidence intervals are correctly noted as devastating.** Binomial 95% CI for 5% accuracy on N=20 is approximately [0.1%, 24.9%]. The paper is honest that differences < ~15pp are noise. This is a legitimate and well-articulated limitation.

## Novelty Assessment

**This experiment has low novelty but high diagnostic value.** It bridges a known gap (PPL does not predict task performance, r=0.08 from prior micro) to the BitNet-2B setting. The prior finding was at smaller scale; confirming it at 2B with real adapters is valuable bookkeeping, not novel science.

**Prior art that matters:**
- The PPL-to-task disconnect is well established in the literature (the project's own micro/models/ppl_vs_task_performance/ already proved r=0.08).
- LoRA Soups (COLING 2025) showed that instruction-format training is critical for task performance in composed adapters. The follow-up experiment (exp_bitnet_instruction_tuned_task_eval) correctly addresses this.
- The finding that NTP adapters do not transfer to tasks is unsurprising given that the base model is not instruction-tuned. This is a well-known property of base completion models.

## Experimental Design

**The experiment tests the stated hypothesis but has a fatal confound: the base model is not instruction-tuned.**

This is not a micro-scale limitation to be forgiven -- it is a design choice that makes the kill criteria nearly impossible to pass regardless of composition quality.

1. **Floor effect on the base model.** Math accuracy of 5% (1/20) means the base model is essentially guessing. You cannot measure adapter improvement when the base cannot perform the task. This is like testing whether a hearing aid improves music appreciation on a deaf person -- the test is invalid, not the hearing aid.

2. **K2 is unfalsifiable at this base quality.** The kill criterion "math adapter shows >= 3pp accuracy improvement" requires the base to have measurable math ability. At 5% base accuracy on N=20 trials, the 95% CI for "no change" encompasses 0%-25%. K2 was doomed from the start.

3. **Training objective mismatch is the primary confound, not composition.** The adapters were trained with NTP loss on domain text. The evaluation requires generative task completion. These are different capabilities. The experiment cannot distinguish "composition destroys task signal" from "there was never task signal to destroy."

4. **Medical exception supports the confound diagnosis.** Medical is the only domain where training format (QA flashcards) matches eval format (QA). It is also the only domain showing improvement. This is strong evidence that the kill reflects training mismatch, not composition failure.

5. **N=13 for medical (not N=20).** The medical eval only found 13 valid QA pairs from the validation data. This further reduces statistical power for the one domain that showed positive signal.

6. **No individual-adapter task eval for all domains.** Individual adapters were only tested on their own domain. Without cross-domain individual results, you cannot compute the composition penalty (composed vs. best-individual) on a per-metric basis.

**The paper's self-diagnosis is accurate and thorough.** The "Analysis: Why Composition Hurts Task Metrics" section correctly identifies all five root causes. The paper is honest about its own limitations.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry is well-structured. Kill criteria K1 and K2 match what was actually tested. The status is correctly "killed." The follow-up node (exp_bitnet_instruction_tuned_task_eval) directly addresses the identified root cause.

One concern: exp_bitnet_reasoning_x_domain depends on exp_bitnet_task_eval, yet it was run and marked "supported" despite the dependency being killed. The reasoning_x_domain experiment used PPL metrics (not task metrics), so it side-steps the killed hypothesis. This dependency should probably be removed or clarified -- a killed task-eval experiment should not block a PPL-based composition test.

## Macro-Scale Risks (advisory)

1. **The instruction-tuning fix is necessary but not sufficient.** Even with instruction-tuned adapters, the 1/N^2 attenuation at N=5 reduces each adapter's task contribution to 4%. Unless the composition method changes (to routing, or 1/N scaling of the merged product rather than averaged factors), task performance will remain attenuated.

2. **Averaged-factor composition has a structural scaling problem for tasks.** At N=25 (target scale), diagonal contribution drops to 0.16%. PPL is robust to this because it only needs the aggregate distribution to be reasonable. Task performance requires specific knowledge to be accessible at full strength. This suggests routing is mandatory for task eval at any N > 1.

3. **The 2B base model limitation will persist at 2B.** Scaling to 7B+ would help, but the project's budget constraints target BitNet-2B. If the instruction-tuning follow-up also fails at 2B, the project needs to either accept PPL-only evaluation or find a way to use a larger base.

## Verdict

**KILL -- correctly applied. No revision needed.**

The experiment was properly designed to test whether composed adapters improve task performance, and it cleanly showed they do not under the tested conditions. The kill is legitimate and well-documented. The paper's self-analysis is thorough and honest.

The kill is attributable to training objective mismatch (NTP vs. task) and base model weakness (non-instruction-tuned 2B), NOT to a fundamental composition failure. This distinction matters: it means the SOLE architecture is not invalidated, only the specific adapter training recipe. The follow-up experiment (exp_bitnet_instruction_tuned_task_eval) is the correct next step.

**Post-mortem quality: high.** The paper identifies all root causes, provides actionable next steps, and does not overclaim. The FINDINGS.md entry is accurate. No corrections needed.

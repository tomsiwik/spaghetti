# Peer Review: Distillation Pilot 50 (Re-review)

**Previous verdict:** REVISE (3 required fixes)
**This review:** Verification that fixes were applied, plus any remaining concerns.

## Revision Assessment

### Fix 1: Contamination caveat added to PAPER.md after results tables

**Status: ADEQUATELY APPLIED.**

PAPER.md line 126 now contains a clear paragraph immediately after the bottom-5 results table:

> "Evaluation uses the last 100 examples of each domain's 1000-example training file. These examples were seen during training (~2.4 epochs). The reported 42.2% average PPL improvement reflects performance on memorized training data, not generalization to unseen queries."

This is precisely the language the original review requested. A reader encountering the 42.2% figure in the results tables will see the contamination caveat before reaching the kill criteria assessment. No ambiguity remains about what the metric measures.

### Fix 2: Status downgraded from "proven" to "supported" in HYPOTHESES.yml

**Status: ADEQUATELY APPLIED.**

HYPOTHESES.yml shows `status: supported` for `exp_distillation_pilot_50`. The evidence claim explicitly notes "(contaminated eval -- see PAPER.md caveat)" and "Downstream task evaluation (MMLU/HumanEval) pending." This correctly reflects the gap between what was tested (PPL on training data) and what the kill criteria specify (MMLU subsets or HumanEval).

### Fix 3: Limitations section rewritten to state eval data IS training data

**Status: ADEQUATELY APPLIED.**

PAPER.md Limitations item 2 now reads:

> "Eval data IS training data -- we evaluate on the last 100 of 1000 training examples per domain. The model has memorized these sequences (~2.4 epochs). This measures memorization quality, not generalization."

The previous wording ("eval data is from training distribution") was ambiguous -- it could have been interpreted as same-distribution-different-samples. The revised wording eliminates that ambiguity entirely. The verdict line (line 136) also correctly says "SUPPORTED" with explicit qualification about contaminated eval data and missing MMLU/HumanEval evaluation.

## Remaining Concerns

The three required fixes address the blocking issues from the original review. The following items from the original review remain as non-blocking advisories (unchanged from first review):

1. **SQL failure uninvestigated** -- examining a handful of SQL training examples for quality issues would be cheap and informative. Not blocking.

2. **Cost discrepancy between MATH.md and VISION.md** -- MATH.md uses $0.355/expert for data cost while other documents reference $0.19-0.25/expert. The kill criterion is evaluated at the conservative $0.44 figure, so this is cosmetic.

3. **No per-example PPL variance reported** -- bootstrap CIs over 100 eval examples per domain would strengthen statistical reporting. Not blocking.

4. **PPL-to-task-accuracy gap remains the primary macro risk** -- the project's own `ppl_vs_task_performance` experiment showed Pearson r=0.08 between full-sequence PPL and task accuracy. The 42.2% PPL improvement on memorized data could correspond to minimal downstream improvement. This is correctly deferred to macro validation but should remain front-of-mind for the next experiment in the dependency chain.

## Mathematical Soundness

No changes from original review. All arithmetic in MATH.md verified correct. Cost analysis, LoRA parameter budget, and composition theory recap are sound.

## Verdict

**PROCEED**

All three required fixes from the REVISE verdict have been adequately applied. The contamination is clearly disclosed in both the results section and the limitations section. The status has been appropriately downgraded to "supported" in HYPOTHESES.yml. The evidence claim accurately describes what was measured and what remains pending.

The experiment provides directional evidence that the SOLE distillation pipeline produces adapters at the target cost point ($0.44/expert) with high win rates on domain-specific data. The "supported" status correctly reflects that downstream task evaluation (MMLU/HumanEval) has not yet been performed. The pipeline engineering is solid, the code is reproducible, and the cost analysis is rigorous.

Next steps on the critical path should prioritize either (a) held-out evaluation of a subset of the 50 experts on MMLU/HumanEval to upgrade from "supported" to "proven," or (b) composition testing (pre-merge of N experts) to validate that individual expert quality translates to composed model quality.

# LEARNINGS: Cross-Domain PPL Matrix N=24

## Core Finding

Grassmannian LoRA adapters are simultaneously domain-specific (DDR=1.126, 24/24 diagonal wins) AND general-purpose (every adapter improves all 24 domains by ~35%). The specificity gap is modest (12.6%), meaning wrong-adapter routing costs only ~13% of the total gain. This explains the Finding #200 paradox: 41% routing accuracy achieves oracle-level aggregate PPL because even misrouted adapters provide most of the benefit.

## Why This Happened

The modest DDR (~1.13) results from three compounding factors:

1. **Rank-16 capacity limitation.** With only 16 dimensions of adapter freedom, the B-matrix learns primarily a generic quality improvement (the dominant 35% effect) plus a small domain-specific refinement (the 13% specific effect). Higher-rank adapters would likely show stronger specialization, as the parameter-efficient fine-tuning survey (arXiv:2403.14608) documents sensitivity to rank and capacity.

2. **Shared Grassmannian A-matrices.** The frozen, structurally-designed A-matrices enforce cross-adapter similarity by construction. All adapters project through the same structured subspace, limiting how different the effective perturbations A*B can be. This is a deliberate design choice (interference prevention) that trades specialization for composability.

3. **Domain overlap at the text level.** Our 24 domains are all English text genres, not fundamentally different tasks (NLI vs QA vs translation). Medical text and science text share vocabulary and structure far more than, say, sentiment classification and code generation. The KL divergence between our domains is relatively small.

## Confirming Evidence

- **LoRAuter** (arXiv:2601.21795): Cross-task performance matrix shows a "pronounced diagonal trend" with "blocks of elevated performance among related task families." Confirms both diagonal dominance and near-interchangeability of related tasks. Out-of-domain single-adapter drops from 99.0% to 81.8% on LLaMA2-7B.
- **LoraHub** (arXiv:2307.13269): Demonstrates adapter composability and transferable skills across tasks, supporting the "adapters are simultaneously specific and general" finding.
- **Our own Finding #201**: Correct A-matrix loading produces 34.8% PPL improvement (vs 0.04% with bug), confirming adapters encode real learned structure.

## Contradicting Evidence

The literature shows **catastrophic**, not modest, degradation from wrong routing in other settings:

- **Task-Aware LoRA Composition** (arXiv:2602.21222): Wrong single-adapter routing causes 24-25 percentage point degradation on PIQA (46% vs 71%) and RTE (52% vs 78%). Much worse than our 12.6%.
- **DES-MoE** (arXiv:2509.16882): Blocking correct MoE experts causes 43% MRR drop (Qwen1.5-MoE) and 76% drop (OLMoE).
- **Rethinking Inter-LoRA Orthogonality** (arXiv:2510.03262): Mathematical orthogonality between LoRA modules does NOT guarantee semantic disentanglement. Adapters cannot be safely interchanged even with enforced orthogonality.
- **Our own SOLE experiments**: Equal-weight composition of 5 adapters caused 127% degradation; a single mismatched SQL adapter exploded PPL to trillions.

**Resolution of the tension:** Our modest DDR reflects the specific experimental conditions (rank-16, shared Grassmannian A, overlapping text domains, PPL-only evaluation). The literature's catastrophic drops occur with (a) higher-capacity adapters, (b) truly distinct tasks, and (c) task-level accuracy metrics. PPL may systematically underestimate behavioral specialization because it weights all tokens equally, masking domain-specific knowledge differences (Finding #200: PPL vs task quality r=0.08).

## Alternative Approaches

For measuring adapter specialization beyond PPL:

1. **Answer-Conditioned PPL (Shadow Scoring)**: Isolate target-answer probability, ignoring prompt/format tokens. Improves correlation with task quality from r=0.08 to r=0.811 (our own SOLE experiments).
2. **Execution-Based Evaluation**: For deterministic domains (code, math), unit test pass rates. Our code adapter showed +9.1pp gain on HumanEval.
3. **Teacher Model Judging**: Use larger model as automated judge (~$0.001/query). Captures semantic quality PPL misses.
4. **MMLU Regression Test**: Check if domain adapters degrade general knowledge. Our adapters showed -3.71pp MMLU degradation, proving format-specialization not knowledge-addition.
5. **Cross-Task Performance Matrices** (arXiv:2601.21795): Min-max normalized scoring across tasks, the methodology we partially adopted.
6. **Multi-adapter fusion** rather than single-adapter selection: LoRA-LEGO (arXiv:2409.16167) clusters at rank dimension; TC-LoRA (arXiv:2508.03999) uses tensor decomposition; ICM-Fusion (arXiv:2508.04153) learns fusion weights without training data.

## Implications for Next Experiments

1. **PPL-based routing evaluation is nearly useless for this setup.** The 12.6% specificity gap drowns in the 35% general improvement. Future routing evaluation MUST use behavioral metrics (generation quality, task accuracy, answer-conditioned PPL).

2. **The "routing doesn't matter" conclusion is PPL-specific, not general.** Literature consistently shows 20-76% degradation at the task level. Our adapters likely ARE more specialized than PPL reveals. The existential test (exp_generation_quality_test) is the right next step.

3. **Multi-adapter fusion is more promising than single-adapter routing.** Since all adapters help all domains, combining 2-3 relevant adapters could capture both the generic and specific benefits. LoRA-LEGO's rank-wise clustering (arXiv:2409.16167) and Ortho-LoRA's gradient projection (arXiv:2601.09684) are proven approaches.

4. **The Grassmannian design creates a composability-specialization tradeoff.** Shared structured A-matrices prevent interference (good for composition) but limit specialization (modest DDR). This is a feature, not a bug, IF composition provides the behavioral benefit.

## Recommended Follow-Up

1. **exp_generation_quality_test (P0)**: Does routed composition produce better TEXT than base model or random routing? This is THE existential test. Motivated by: DDR=1.13 at PPL level may correspond to much larger differences at behavioral level (contradicting evidence above), and Finding #200 showing PPL is wrong metric.

2. **exp_task_accuracy_real_benchmarks (P0)**: MMLU/GSM8K/HumanEval with correct N=24 adapter loading. Motivated by: Task-Aware LoRA Composition (arXiv:2602.21222) showing 24-25pp task-level degradation where we see only 12.6% PPL degradation.

3. **exp_multi_adapter_fusion**: Combine top-2 or top-3 adapters per query instead of selecting one. Motivated by: LoRA-LEGO (arXiv:2409.16167), ICM-Fusion (arXiv:2508.04153), and our finding that multiple adapters provide overlapping benefit.

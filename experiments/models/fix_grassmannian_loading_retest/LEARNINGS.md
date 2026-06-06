# LEARNINGS: fix_grassmannian_loading_retest

## Core Finding

The A-matrix loading bug was confirmed as root cause of all 7 prior N=24 routing kills — correct Grassmannian A-matrix loading restores 34.8% PPL improvement (was 0.04%). However, **routed PPL = oracle PPL despite 41% routing accuracy** raises a fundamental question: adapters may provide general improvement rather than domain-specific specialization, which would undermine the "composable domain experts" thesis.

## Why This Happened

### Bug validation (straightforward)
The Grassmannian skeleton stores domain-specific A matrices indexed by training-time domain ordering (medical=0, code=1, ...), not alphabetical. All prior routing experiments used `mlx_lm.LoRALinear` (random A) or alphabetical indexing, producing `output = base + scale * (x @ RANDOM_A) @ TRAINED_B ≈ base + noise`. Theorem 1 in MATH.md correctly proves E[noise] = 0, confirmed by -0.6% PPL "improvement."

### Adapter interchangeability (the real question)
The result that routed PPL (6.32) = oracle PPL (6.32) despite 59% misrouting has two interpretations:

1. **Adapters provide general quality improvement** — the Grassmannian subspaces and trained B weights improve the model regardless of which domain's adapter is applied. This would mean routing is unnecessary.

2. **PPL is the wrong metric** — our own Finding #170 showed PPL has r=0.08 correlation with task quality. Adapters might be domain-specific at the *behavioral* level (task accuracy, knowledge retrieval) while appearing interchangeable at the PPL level. PPL measures fluency, not knowledge.

The literature strongly supports interpretation 2. Li et al. (DES-MoE, arXiv:2509.16882) show that blocking correct experts causes 43-76% performance drops on *task-specific* metrics. Our own SOLE experiments showed domain adapters *degrade* MMLU by 3.71pp — they are format-specialized, not general-purpose.

## Confirming Evidence

- **Our own SOLE experiments**: Python adapter +9.1pp on HumanEval but -3.71pp on MMLU. Adapters ARE domain-specific at the task level, even if PPL doesn't capture this (Finding #170: PPL-quality correlation r=0.08).
- **DES-MoE** (arXiv:2509.16882): Blocking correct domain experts causes 43% (Qwen1.5-MoE) to 76% (OLMoE) performance drops on domain-specific tasks. Routing accuracy matters — but only when measured on domain-specific benchmarks, not PPL.
- **LoRAuter** (arXiv:2601.21795): Achieves 101.2% of oracle performance via task-embedding routing at 1500+ adapter scale. Uses cross-task performance matrix to verify adapter specificity — shows "pronounced diagonal trend" proving peak performance only on training task.
- **DeepSeekMoE**: Emphasizes "ultimate expert specialization" where routing ensures non-overlapping knowledge — required for MoE to match dense model performance.

## Contradicting Evidence

- **This experiment itself**: Routed PPL = oracle PPL despite 59% misrouting. If taken at face value, this contradicts all literature showing routing matters. Resolution: PPL is the wrong metric for detecting domain specificity (r=0.08 with task quality).
- **"What is Wrong with Perplexity"** (arXiv:2410.23771, ICLR 2025): Perplexity confounds stylistic mimicry with factual internalization. A model can achieve low PPL by matching surface patterns without domain knowledge.
- **KR-Test** (arXiv:2601.03505): Proposes Knowledge Retention Test as lightweight alternative to PPL for measuring factual learning vs linguistic mimicry in LoRA fine-tuning. Standard PPL fails to detect knowledge differences.

## Alternative Approaches

1. **24x24 Cross-Domain PPL Matrix** (recommended by reviewer): Apply each adapter to each domain's validation set. If adapters are interchangeable, all rows look the same. If domain-specific, the diagonal dominates. This is the LoRAuter evaluation methodology (arXiv:2601.21795).

2. **Answer-Conditioned PPL** (our Finding #170): Replace standard PPL with shadow scoring (condition on correct answer, measure continuation PPL). Correlation with quality jumps from r=0.08 to r=0.811. This would detect domain specificity that standard PPL misses.

3. **Task-Specific Benchmarks**: Use execution-based evaluation (HumanEval for code, GSM8K for math, etc.) to measure whether routing to the correct adapter improves *task accuracy*, not just PPL. This is already planned as exp_task_accuracy_real_benchmarks on the P0 deployment track.

4. **Adapters Selector** (COLING 2025, Tian et al.): Trains a middleman adapter to select the correct domain adapter. Cross-domain multi-task execution via compact model + multiple LoRA modules.

5. **LoRI** (arXiv:2504.07448): Reduces cross-task interference in multi-task LoRA by orthogonal gradient projection — directly relevant to our Grassmannian approach.

## Implications for Next Experiments

1. **Finding #198 is definitively invalidated.** The "adapters provide 0.04% benefit" was entirely a loading bug. True benefit is 34.8% PPL improvement. All 7 routing kills remain invalidated in their PPL-based conclusions, though routing accuracy numbers (~40%) may still be valid since routing was trained on base hidden states.

2. **The routing accuracy plateau at ~40% is real but may not matter.** If adapters provide general improvement (PPL-level), routing accuracy is irrelevant for PPL. But if adapters are domain-specific at the task level (literature strongly suggests yes), routing accuracy matters for task-specific quality. The 24x24 cross-domain matrix would resolve this.

3. **PPL-based evaluation is insufficient for the "composable domain experts" thesis.** The project proved PPL doesn't predict task quality (r=0.08). Yet we're still using PPL to evaluate adapter specialization. The P0 deployment track (exp_generation_quality_test, exp_task_accuracy_real_benchmarks) is the correct path — it tests behavioral outcomes, not proxy metrics.

4. **The P0 critical path is now unblocked.** With 34.8% oracle PPL improvement confirmed, the adapters work. The question is whether they produce *better text* (exp_generation_quality_test) and *better task accuracy* (exp_task_accuracy_real_benchmarks), not whether they produce lower PPL.

## Recommended Follow-Up

1. **exp_cross_domain_ppl_matrix** — 24x24 matrix applying each adapter to each domain. Motivation: LoRAuter (arXiv:2601.21795) uses this exact methodology to verify adapter specificity. If diagonal dominates → adapters are domain-specific and routing matters. If uniform → adapters are general-purpose and routing is unnecessary. Quick experiment, reuses existing infrastructure.

2. **exp_generation_quality_test** (already on P0 critical path) — Generate text with routed composition vs base model, evaluate with LLM judge or human eval. This is THE existential test for the thesis. Motivation: Finding #170 (PPL r=0.08 with quality) means only behavioral evaluation can validate the architecture.

3. **exp_task_accuracy_real_benchmarks** (already on P0 critical path) — MMLU/GSM8K/HumanEval with vs without composition. Motivation: DES-MoE (arXiv:2509.16882) shows 43-76% task performance drops from wrong routing — but only on task-specific metrics, not PPL.

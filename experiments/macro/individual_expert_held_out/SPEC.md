# Experiment Spec: individual_expert_held_out

## Objective

Diagnose whether the -3.67pp MMLU regression (from N=50 composed model) comes
from distillation quality or composition interference, by testing the top 20
adapters INDIVIDUALLY (no composition) on the full MMLU test set.

Primary metric: mean individual delta vs base ($\bar{\delta}$).

## Model & Data

- **Base model**: Qwen2.5-7B at `/workspace/models/Qwen2.5-7B` (NF4 quantized)
- **Adapters**: Top 20 by training PPL improvement from `/workspace/llm/adapters/`.
  Ranked via `/workspace/llm/results/pilot50_benchmark.json` (field:
  `per_adapter.{name}.ppl_improvement_pct`, descending). Fallback: alphabetical.
- **Eval data**: MMLU test split from `cais/mmlu` on HuggingFace. Use ALL 57
  available subjects (not a random subset). Use full test set per subject
  (no --max-per-subject cap) for maximum statistical power.
- **Evaluation**: 0-shot log-probability scoring (argmax over log P(c|prompt)
  for c in {A,B,C,D}). Same methodology as pilot50_held_out_eval/eval_mmlu.py.

## Procedure

1. **Load base model** (NF4 quantized, device_map="auto").

2. **Evaluate base model** on all 57 MMLU subjects. Save base results
   as checkpoint (in case of crash). Record per-subject accuracy.

3. **For each of the top 20 adapters** (sequentially):
   a. Reload a FRESH base model (do NOT reuse -- PeftModel.from_pretrained
      can modify the base in-place, contaminating subsequent adapters).
   b. Load adapter via `PeftModel.from_pretrained(fresh_base, adapter_path)`.
   c. Evaluate on ALL 57 MMLU subjects using same log-prob methodology.
   d. Compute per-subject delta vs base.
   e. Delete model, clear CUDA cache.
   f. Save intermediate results after each adapter (crash resilience).

4. **Compute aggregate metrics**:
   - Per-adapter overall accuracy (micro-averaged across all subjects)
   - Per-adapter delta vs base ($\delta_i$)
   - Mean delta ($\bar{\delta}$) with bootstrap 95% CI (1000 resamples)
   - Median delta, std delta
   - In-domain vs out-of-domain delta breakdown (using DOMAIN_TO_MMLU mapping
     from pilot50_held_out_eval/eval_mmlu.py)
   - Diagnosis: DISTILLATION_QUALITY / COMPOSITION_INTERFERENCE / INCONCLUSIVE

5. **SMOKE_TEST mode** (SMOKE_TEST=1): Use only 3 adapters, 5 MMLU subjects,
   10 questions per subject. Should complete in <5 min.

## Kill Criteria Assessment

- **K1**: $\bar{\delta} > -1\text{pp}$ -- PASS: problem is composition, not distillation.
  Individuals are roughly neutral, so the -3.67pp came from composing 50.
- **K2**: $\bar{\delta} < -3\text{pp}$ -- PASS: distillation memorized. Adapters
  individually harm generalization. Composition just accumulates intrinsic harm.
- **Inconclusive**: $-3\text{pp} \le \bar{\delta} \le -1\text{pp}$ -- both effects contribute.

## Output

Save results to: `results/individual_expert_held_out/individual_expert_results.json`

Required fields in JSON:
```json
{
  "experiment": "individual_expert_held_out",
  "timestamp": "ISO-8601",
  "base_model": "/workspace/models/Qwen2.5-7B",
  "config": {
    "n_experts": 20,
    "n_subjects": 57,
    "max_per_subject": null,
    "seed": 42,
    "smoke_test": false
  },
  "base": {
    "per_subject": {"subject_name": {"correct": int, "total": int, "accuracy": float}},
    "overall": {"correct": int, "total": int, "accuracy": float}
  },
  "individual_experts": {
    "adapter_name": {
      "eval_results": {"per_subject": {...}, "overall": {...}},
      "accuracy": float,
      "delta_vs_base_pp": float
    }
  },
  "analysis": {
    "avg_delta_pp": float,
    "median_delta_pp": float,
    "std_delta_pp": float,
    "bootstrap_ci_95_pp": [float, float],
    "n_positive": int,
    "n_negative": int,
    "n_neutral": int,
    "n_tested": int,
    "in_domain_avg_delta_pp": float,
    "out_of_domain_avg_delta_pp": float,
    "diagnosis": "DISTILLATION_QUALITY | COMPOSITION_INTERFERENCE | INCONCLUSIVE",
    "detail": "human-readable explanation"
  },
  "kill_criteria": {
    "K1_composition_interference": {"threshold": "> -1pp", "value": float, "pass": bool},
    "K2_distillation_memorization": {"threshold": "< -3pp", "value": float, "pass": bool}
  }
}
```

## Constraints

- **Max runtime**: ~3 hours (budget ~$0.50 on A5000)
- **Expected GPU memory**: ~8GB (NF4 base + rank-16 LoRA adapter)
- **Must support SMOKE_TEST=1**: 3 adapters, 5 subjects, 10 questions/subject
- **Crash resilience**: Save intermediate results after each adapter. On restart,
  load cached base results and skip already-evaluated adapters.
- **Fresh base per adapter**: Critical -- do NOT reuse base model across adapters.
  PeftModel modifies the base model object. Each adapter evaluation must start
  from a freshly-loaded base to avoid cross-contamination.

## Existing Code

A script already exists at `macro/individual_expert_held_out/run_individual_eval.py`.
It implements the core logic but needs these improvements:

1. **Evaluate ALL 57 subjects** instead of random 20 (more statistical power,
   enables proper in-domain vs out-of-domain breakdown).
2. **Remove max_per_subject cap** (currently defaults to 50). Use full test sets.
3. **Add bootstrap CI** on $\bar{\delta}$.
4. **Add in-domain vs out-of-domain analysis** using DOMAIN_TO_MMLU mapping.
5. **Add crash resilience** (save after each adapter, resume on restart).
6. **Add SMOKE_TEST support**.
7. **Add kill_criteria section** to output JSON.

The experiment-programmer should use the existing script as a starting point
and apply these improvements.

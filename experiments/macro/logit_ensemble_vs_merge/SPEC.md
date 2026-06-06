# Experiment Spec: logit_ensemble_vs_merge

## Objective

Diagnose whether the -3.67pp MMLU regression in N=50 pilot composition comes
from **weight-space interference** (LoRA deltas destructively interact when
summed) or **distillation quality** (individual adapters hurt the base on
held-out MCQ). Compares weight-merge (1 forward pass, summed weights) vs
logit-ensemble (N forward passes, averaged logits) vs base (no adapter).

## Model and Data

- **Base model**: Qwen2.5-7B, NF4 quantized (4-bit), /workspace/models/Qwen2.5-7B
- **Adapters**: Pilot-50 LoRA adapters (rank-16, all modules), /workspace/llm/adapters/
  - Selected by training PPL ranking from results/pilot50_benchmark.json
  - Top-N adapters used for each N value
- **Eval data**: MMLU test split (cais/mmlu on HuggingFace), 15 subjects, up to 50 examples each
- **Subjects**: abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge,
  college_biology, college_chemistry, college_computer_science, college_mathematics,
  college_medicine, college_physics, computer_security, conceptual_physics, econometrics,
  electrical_engineering

## Procedure

The existing script at `macro/logit_ensemble_vs_merge/run_logit_ensemble.py`
implements the following protocol. This SPEC documents expected behavior for
verification.

### Phase 1: Base Model Evaluation
1. Load Qwen2.5-7B in NF4 quantization
2. Evaluate on all 15 MMLU subjects using log-probability scoring:
   - For each question: compute logits at the last token position
   - Score each answer choice (A/B/C/D) by log-prob (including space-prefixed variants)
   - Predict the choice with highest log-prob
3. Record per-subject and overall accuracy

### Phase 2: Weight-Merge Evaluation (for each N in {5, 10, 25, 50})
1. Reload fresh base model (PeftModel modifies in place)
2. Load top-N adapters by training PPL ranking
3. Compose via PEFT add_weighted_adapter with combination_type="linear",
   weights=[1/N, 1/N, ..., 1/N]
4. Evaluate on same 15 subjects with same scoring method
5. Record per-subject and overall accuracy, delta vs base
6. Unload model, clear CUDA cache

### Phase 3: Logit-Ensemble Evaluation (for each N in {5, 10, 25, 50})
1. Reload fresh base model
2. Pre-load all MMLU examples (to avoid re-downloading per adapter)
3. For each of the N adapters:
   a. Load adapter via PeftModel.from_pretrained(base_model, adapter_path)
   b. For each (subject, example): compute logits at last position, accumulate
   c. Unload adapter, gc.collect(), cuda.empty_cache()
4. Average accumulated logits by N
5. Score predictions from averaged logits using same A/B/C/D scheme
6. Record per-subject and overall accuracy, delta vs base

### Phase 4: Diagnosis
1. For each N, compute gap = ensemble_delta - merge_delta (pp)
2. Average gap across all N values
3. Classify:
   - avg_gap > 2.0pp: WEIGHT_INTERFERENCE
   - avg_gap < 0.5pp: DISTILLATION_QUALITY
   - otherwise: MIXED

## Kill Criteria Assessment

- **K1: Same regression** -- Logit ensemble shows same regression as weight
  merge (gap < 0.5pp averaged across N values). Diagnosis: DISTILLATION_QUALITY.
  The problem is in the adapters, not the composition. Action: improve training.
  Status: KILLED (problem is distillation, not interference).

- **K2: Overhead > 10x** -- Logit ensemble wall-clock time exceeds 10x weight
  merge time for N=5 (the most practical N). If ensemble at N=5 takes >10x
  merge time, ensembling is impractical even as a diagnostic.
  Note: at N=5, expected ratio is ~5x (well within K2). At N=50, ratio is
  ~50x (violates K2 but irrelevant if N=5 already shows the pattern).

## Output

- Save results to: results/logit_ensemble_vs_merge/logit_ensemble_vs_merge.json
- Required fields in JSON:
  ```json
  {
    "base": {
      "per_subject": {"subject_name": {"correct": int, "total": int, "accuracy": float}},
      "overall": {"correct": int, "total": int, "accuracy": float}
    },
    "weight_merge": {
      "5": {"n_experts": 5, "accuracy": float, "delta_vs_base_pp": float, "elapsed_s": float},
      "10": {...},
      "25": {...},
      "50": {...}
    },
    "logit_ensemble": {
      "5": {"n_experts": 5, "accuracy": float, "delta_vs_base_pp": float, "elapsed_s": float},
      "10": {...},
      "25": {...},
      "50": {...}
    },
    "config": {
      "subjects": [...],
      "max_per_subject": int,
      "n_values": [5, 10, 25, 50],
      "seed": 42
    },
    "diagnosis": {
      "avg_ensemble_vs_merge_gap_pp": float,
      "verdict": "WEIGHT_INTERFERENCE" | "DISTILLATION_QUALITY" | "MIXED",
      "interpretation": string
    }
  }
  ```

## Expected Results (Pre-Registration)

Based on prior evidence (3 adapters showed -3.71pp individually, structural
orthogonality cos=0.0002, safety bound alpha=0.022), the prediction is:

**Most likely outcome: DISTILLATION_QUALITY (K1 KILLED)**

| N | Merge delta (pp) | Ensemble delta (pp) | Gap (pp) |
|---|------------------|--------------------|---------:|
| 5 | -2 to -4 | -2 to -4 | < 0.5 |
| 10 | -3 to -4 | -3 to -4 | < 0.5 |
| 25 | -3 to -4 | -3 to -4 | < 0.5 |
| 50 | -3 to -4 | -3 to -4 | < 0.5 |

**Alternative outcome: WEIGHT_INTERFERENCE**

Would require ensemble delta near 0pp while merge delta is -3.7pp. This
would contradict the individual adapter results and the orthogonality proof.

**Surprise outcome: ensemble BEATS base**

If ensemble delta > 0, individual adapters actually help on MMLU when
ensembled (majority voting smooths errors). This would mean the regression
is entirely from weight-space interference and selective composition (top-k)
should solve it completely.

## Constraints

- **Max runtime**: ~10 hours (dominated by N=50 ensemble: 50 adapters x 750 questions)
- **Expected GPU memory**: ~8GB with NF4 quantization + single adapter loaded
- **Must support SMOKE_TEST=1**: Use --max-per-subject 5 to reduce to ~42 min
- **Sequential adapter loading**: Ensemble loads one adapter at a time to stay
  within GPU memory. Cannot batch multiple adapters.
- **Fresh base per composition**: The script reloads the base model for each N
  to avoid PeftModel state contamination. This adds ~4 base loads (~2 min total).

## Verification Checklist (for experiment-programmer)

The script already exists at `macro/logit_ensemble_vs_merge/run_logit_ensemble.py`.
Verify:

1. [ ] Weight merge uses combination_type="linear" with 1/N weights (not "svd" or "ties")
2. [ ] Logit ensemble accumulates raw logits (not log-probs or softmax probs)
3. [ ] Same 750 questions evaluated under all three conditions (base, merge, ensemble)
4. [ ] Adapter ranking uses pilot50_benchmark.json (top-N by training PPL improvement)
5. [ ] Base model is reloaded fresh before each weight merge (PeftModel modifies in place)
6. [ ] gc.collect() + torch.cuda.empty_cache() between adapter loads in ensemble
7. [ ] SMOKE_TEST support: --max-per-subject flag reduces workload
8. [ ] Results JSON includes elapsed_s for each condition (enables K2 check)
9. [ ] Diagnosis thresholds: 2.0pp for interference, 0.5pp for distillation

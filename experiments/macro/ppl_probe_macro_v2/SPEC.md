# Experiment Spec: ppl_probe_macro_v2

## Objective

Determine whether PPL-probe weighted composition of 5 LoRA adapters improves
MMLU accuracy over equal-weight composition on Qwen2.5-7B. This is the FP16
lifeline experiment: if PPL-probe routing fixes the composition catastrophe,
SOLE works with FP16 adapters. If not, pivot to BitNet.

## Model & Data

- **Base model**: Qwen2.5-7B cached at `/workspace/models/Qwen2.5-7B`
- **Adapters**: 5 adapters from `/workspace/llm/adapters/`:
  - `bash`, `math`, `medical`, `python`, `sql`
  - rank-16, all-modules (q/k/v/o/gate/up/down), trained with Unsloth
- **Eval data**: MMLU test set via `datasets.load_dataset("cais/mmlu", subject, split="test")`
  - All 57 subjects
  - Per subject: first 10 examples for probe, remaining for eval
- **Quantization**: NF4 (4-bit) with float16 compute dtype, double quant

## Procedure

### Phase 0: Setup and Validation

1. Load base model with NF4 quantization
2. Verify all 5 adapters load correctly with PEFT
3. Load all 57 MMLU subjects, split each into probe (10) and eval (remaining)
4. Record total example counts per subject

### Phase 1: Probe Profiling (5 adapters x 57 subjects)

For each adapter (load once, eval all subjects):
1. Load adapter via `PeftModel.from_pretrained(base_model, adapter_path)`
2. For each subject, compute answer-only PPL on 10 probe examples:
   - Format as `{question}\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:`
   - Extract log-prob of correct answer token (A/B/C/D) from last position
   - PPL = exp(mean negative log-prob across 10 examples)
3. Also compute base model PPL on all probe buffers (no adapter)
4. Unload adapter, clear CUDA cache
5. Save probe profile: `{adapter: {subject: ppl}}`

**Timing estimate**: 5 adapters x 57 subjects x 10 examples = 2850 forward passes, ~5 min

### Phase 2: Weight Computation

For each subject and each temperature tau in {0.1, 0.5, 1.0, 2.0}:
1. Collect PPL scores: `ppls = [PPL_adapter_i(subject) for i in range(5)]`
2. Compute scores: `scores = -log(ppls) / tau`
3. Softmax: `weights = softmax(scores - max(scores))`
4. Also compute top-1 selection: `best = argmin(ppls)`
5. Save all weight vectors

### Phase 3: Evaluation (8 conditions x 57 subjects)

Evaluate each condition on the held-out eval examples (everything after first 10):

| Condition | Weight Vector | Description |
|-----------|--------------|-------------|
| C0: base | no adapters | Frozen Qwen2.5-7B |
| C1a: equal_scaled | [0.2, 0.2, 0.2, 0.2, 0.2] | Equal-weight, scaled by 1/N |
| C1b: equal_unscaled | [1.0, 1.0, 1.0, 1.0, 1.0] | Full addition (reproduces catastrophe) |
| C2a: ppl_probe_t0.1 | softmax(-log(ppl)/0.1) | Near-one-hot routing |
| C2b: ppl_probe_t0.5 | softmax(-log(ppl)/0.5) | Recommended from micro |
| C2c: ppl_probe_t1.0 | softmax(-log(ppl)/1.0) | Standard softmax |
| C2d: ppl_probe_t2.0 | softmax(-log(ppl)/2.0) | Smoother weighting |
| C3: top1_probe | one-hot on argmin(ppl) | Best single adapter by probe |

For each condition:
1. Compose model using `PeftModel.add_weighted_adapter(combination_type="linear")`
2. For each subject, evaluate 0-shot MMLU:
   - Format question same as probe
   - Compare log-probs of A/B/C/D tokens, pick argmax
   - Count correct/total
3. Record per-subject accuracy and overall accuracy
4. Unload composed adapter before next condition

**Timing estimate**: 8 conditions x 57 subjects x ~40 examples = ~18,240 forward passes, ~30 min

**Efficiency optimization**: For C0 (base), no adapter needed. For all composition
conditions (C1a, C1b, C2a-d, C3), the model reload is the same -- only the weight
vector changes. Use `model.delete_adapter("composed")` between conditions to avoid
full reload. Each PEFT weighted merge takes <1 second.

**IMPORTANT**: For C1b (unscaled), PEFT's `add_weighted_adapter` with
`combination_type="linear"` and weights=[1,1,1,1,1] should work. Verify by checking
if PPL matches the known catastrophic value (~trillions). If PEFT normalizes weights
internally, use `combination_type="cat"` or manual weight injection.

### Phase 4: Latency Measurement (K2)

After evaluation, measure per-query serving latency:
1. Time a single PEFT weighted merge (compose 5 adapters with given weights)
2. Time a single forward pass on a 512-token input
3. Report total = merge_time + inference_time
4. Do 100 iterations, report mean and std

### Phase 5: Analysis

1. Compute per-condition overall accuracy and delta vs base
2. For each PPL-probe temperature, compute:
   - Overall accuracy
   - Per-subject accuracy
   - Per-subject weight distribution (entropy, max weight, which adapter gets highest)
3. Compute probe-oracle correlation:
   - Oracle = which single adapter gives best accuracy on each subject (from C3 or from
     per-adapter individual eval if available)
   - Probe = which adapter gets highest weight from PPL-probe
   - Pearson r between weight vectors
4. Identify whether sql adapter consistently gets lowest weight (validating the
   dropout-robustness finding that sql is harmful)
5. For subjects without a relevant adapter (e.g., philosophy, history): check if
   PPL-probe produces near-uniform weights (correct behavior)

## Kill Criteria Assessment

- **K1**: For each of the 5 adapter home domains (bash -> computer_security,
  math -> college_mathematics/high_school_mathematics, medical -> college_medicine/
  clinical_knowledge, python -> college_computer_science, sql -> (no direct MMLU match)):
  compute composed PPL on the probe buffer. If composed PPL > 2x single-adapter PPL
  on >50% of home domains: KILL.

- **K2**: Total per-query latency (weight lookup + merge + forward pass) > 100ms: KILL.
  Note: weight lookup is cached (<1ms), merge is <100ms for 5 adapters, forward pass
  is ~20ms. Expected PASS.

- **K3**: best PPL-probe condition accuracy - equal_scaled accuracy < 2pp: KILL.
  This is the critical metric. At micro scale PPL-probe gave +9.34pp over equal-weight.
  If even +2pp at macro: sufficient to justify the routing overhead.

## Output

- Save results to: `/workspace/llm/results/ppl_probe_macro_v2/results.json`
- Required fields in JSON:
  ```json
  {
    "config": {
      "base_model": "Qwen2.5-7B",
      "adapters": ["bash", "math", "medical", "python", "sql"],
      "n_probe": 10,
      "n_subjects": 57,
      "temperatures": [0.1, 0.5, 1.0, 2.0],
      "quantization": "nf4",
      "seed": 42
    },
    "probe_profiles": {
      "base_ppls": {"subject": "ppl_value"},
      "adapter_ppls": {"adapter_name": {"subject": "ppl_value"}},
      "probe_time_s": 0
    },
    "conditions": {
      "base": {
        "per_subject": {"subject": {"correct": 0, "total": 0, "accuracy": 0.0}},
        "overall": {"correct": 0, "total": 0, "accuracy": 0.0}
      },
      "equal_scaled": {"...same structure..."},
      "equal_unscaled": {"..."},
      "ppl_probe_t0.1": {"..."},
      "ppl_probe_t0.5": {"..."},
      "ppl_probe_t1.0": {"..."},
      "ppl_probe_t2.0": {"..."},
      "top1_probe": {"..."}
    },
    "weight_distributions": {
      "ppl_probe_t0.5": {
        "subject": {
          "weights": {"adapter": 0.0},
          "entropy": 0.0,
          "max_weight": 0.0,
          "best_adapter": "name"
        }
      }
    },
    "latency": {
      "merge_ms_mean": 0.0,
      "merge_ms_std": 0.0,
      "forward_ms_mean": 0.0,
      "forward_ms_std": 0.0,
      "total_ms_mean": 0.0,
      "total_ms_std": 0.0
    },
    "kill_criteria": {
      "K1_domains_exceeding_2x": 0,
      "K1_total_domains": 5,
      "K1_pass": true,
      "K2_total_ms": 0.0,
      "K2_pass": true,
      "K3_best_probe_acc": 0.0,
      "K3_equal_scaled_acc": 0.0,
      "K3_improvement_pp": 0.0,
      "K3_pass": true
    },
    "analysis": {
      "probe_oracle_top1_agreement": 0.0,
      "probe_oracle_pearson_r": 0.0,
      "sql_mean_weight": 0.0,
      "unmatched_subjects_weight_entropy": 0.0
    }
  }
  ```

## Constraints

- **Max runtime**: 3 hours (generous; expected ~2 hours)
- **Expected GPU memory**: ~10GB with NF4 4-bit quantization
- **Must support SMOKE_TEST=1**: When set, use 3 subjects, 5 probe examples, 10 eval examples
- **Adapter loading**: Load each adapter ONCE for probe phase. For composition eval,
  use PEFT's weighted adapter mechanism -- do NOT manually merge weight tensors.
- **CRITICAL: Do NOT load all 50 adapters.** Only load the 5 specified adapters.
- **Random seed**: 42 for all shuffling (probe/eval split, subject ordering)
- **Answer tokens**: Use both `"A"` and `" A"` token IDs for robustness (Qwen tokenizer
  may encode the space differently). Take max log-prob across both.

## Differences from v1 (macro/ppl_probe_macro_composition/)

| Aspect | v1 | v2 (this) |
|--------|-----|-----------|
| Adapters | All 50 | Only 5 (bash, math, medical, python, sql) |
| Equal-weight | 1/N only | Both 1/N and unscaled |
| Temperatures | Single (1.0) | Sweep {0.1, 0.5, 1.0, 2.0} |
| Top-k conditions | top-5, top-10 | top-1 only |
| Latency measurement | No | Yes (K2 kill criterion) |
| Weight distribution analysis | Minimal | Full entropy/distribution tracking |
| MMLU subjects | Random 20 | All 57 |
| Known issue | N=50 catastrophe confounds routing with dilution | N=5 isolates routing quality |

## Prior Art to Reuse

- `macro/ppl_probe_macro_composition/run_ppl_probe_composition.py`: Reuse the
  `compute_answer_ppl`, `format_mmlu_prompt`, `evaluate_mmlu_accuracy`, and
  `compose_adapters_weighted` functions. They are tested and correct.
- `macro/individual_expert_held_out/run_individual_eval.py`: Reuse the MMLU loading
  and evaluation logic. The base MMLU accuracy (70.3%) is the reference baseline.

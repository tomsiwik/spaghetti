# Experiment Spec: composition_dropout_robustness

## Objective

Determine whether composed model quality is robust to random 20% expert dropout
using a bootstrap test. Compose 20 random 80%-subsets of the 50 pilot adapters,
measure PPL on calibration data, and compute the coefficient of variation across
subsets. Low CV (< 5%) means composition is robust; high CV means it is fragile.

Three questions:
1. Is PPL stable across random subsets? (K1: CV < 5%)
2. Does any subset significantly outperform all-50? (K2: best delta < 10%)
3. Does any subset significantly underperform all-50? (K3: worst delta < 15%)

## Model & Data

- **Base model:** Qwen2.5-7B at `/workspace/hf_cache` (4-bit NF4 quantization)
- **Adapters:** 50 pilot adapters at `/workspace/llm/adapters/`
  - Each adapter is a directory containing `adapter_model.safetensors` and
    `adapter_config.json`
  - All rank-16, all-modules (q/k/v/o/gate/up/down)
- **Calibration data:** Training data from adapter domains at
  `/workspace/llm/data/distillation/` (JSONL files with `messages` field)
  - Use 30 texts, sampled across multiple domains for balance
  - Each text truncated to 512 tokens
- **No external data downloads required**

## Procedure

### Phase 0: Setup and Discovery (~30s)

1. Set random seed = 42 for reproducibility
2. Discover all adapters in `/workspace/llm/adapters/` (directories with
   `adapter_model.safetensors`). Sort alphabetically. Use first 50.
3. Load calibration texts from training data directories:
   - Sample ~3 texts per adapter domain (from JSONL files)
   - Total: 30 calibration texts
   - Extract user + assistant content from `messages` field
   - Filter: text length > 100 characters
4. Load tokenizer from Qwen2.5-7B

### Phase 1: Reference Measurement — All 50 Experts (~45s)

1. Compose all 50 adapters via CPU weight-space merge:
   - For each adapter, load `adapter_model.safetensors` via `safetensors.torch.load_file`
   - Sum all weight deltas: `composed[key] += adapter_tensors[key]`
   - Save the composed adapter to a temp directory (bfloat16)
   - Copy `adapter_config.json` from first adapter
2. Load base model (4-bit NF4, `device_map="auto"`)
3. Load composed adapter via `PeftModel.from_pretrained`
4. Compute PPL on all 30 calibration texts:
   - For each text: tokenize, forward pass, compute cross-entropy loss
   - PPL = exp(total_loss / total_tokens)
5. Record `PPL_ref`
6. Delete model, free GPU memory, remove temp directory

### Phase 2: Base Model Measurement (~30s)

1. Load base model (4-bit NF4, `device_map="auto"`)
2. Compute PPL on same 30 calibration texts
3. Record `PPL_base`
4. Delete model, free GPU memory

### Phase 3: Bootstrap — 20 Random 80% Subsets (~8 min)

For each b = 1, 2, ..., 20:

1. Sample k = floor(50 * 0.8) = 40 adapters uniformly at random from the 50
   (using `random.sample` with seed-based RNG)
2. Compose the 40 adapters via CPU weight-space merge (same as Phase 1)
3. Load base model + composed adapter
4. Compute PPL on the 30 calibration texts
5. Compute delta from reference: `(PPL_b - PPL_ref) / PPL_ref * 100`
6. Compute delta from base: `(PPL_b - PPL_base) / PPL_base * 100`
7. Record: subset identity, PPL_b, deltas
8. Delete model, free GPU, remove temp dir

**Memory discipline:** Each iteration loads a fresh base model + composed
adapter. Never keep more than one model in GPU memory. Use `gc.collect()`
and `torch.cuda.empty_cache()` between iterations.

**Performance optimization:** The CPU merge step (~15s per subset) can be
overlapped with GPU teardown from the previous iteration. The script should
compose the next subset while the GPU clears.

### Phase 4: Analysis

Compute across the 20 bootstrap results:

1. **Mean PPL:** `mean(PPL_b)`
2. **Std PPL:** `std(PPL_b)`
3. **CV:** `std / mean * 100`
4. **Best delta:** `min(delta_from_ref)` — most improvement from dropping experts
5. **Worst delta:** `max(delta_from_ref)` — most degradation from dropping experts
6. **Histogram of deltas** (for logging, not visualization)

### Phase 5: Kill Criteria Evaluation

- K1: CV <= 5.0% --> SURVIVES (composition stable)
- K2: |best_delta| <= 10.0% --> SURVIVES (pruning not critical)
- K3: worst_delta <= 15.0% --> SURVIVES (random dropout not dangerous)

Overall: SURVIVES if all three pass.

## Kill Criteria Assessment

- **K1:** PPL CV across 20 random 80% subsets > 5% --> KILLED (fragile)
  - Expected: 1-3% based on micro results and orthogonality analysis
  - Boundary case (4-6%): interpret with caution, recommend more samples

- **K2:** Best 80% subset improves > 10% over all-50 --> KILLED (pruning critical)
  - Expected: < 5% improvement from any subset
  - If K2 kills, it means some experts are actively harmful and must be pruned

- **K3:** Worst 80% subset degrades > 15% from all-50 --> KILLED (dropout dangerous)
  - Expected: 3-8% worst-case degradation
  - If K3 kills, composition is sensitive to which experts are present

## Output

Save results to: `/workspace/llm/results/composition_dropout_robustness/results.json`

Required fields in JSON:
```json
{
  "config": {
    "n_experts": 50,
    "n_keep": 40,
    "dropout_frac": 0.8,
    "n_bootstrap": 20,
    "calib_samples": 30,
    "max_seq_len": 512,
    "base_model": "Qwen/Qwen2.5-7B",
    "seed": 42,
    "smoke_test": false
  },
  "reference_ppl": 0.0,
  "base_ppl": 0.0,
  "bootstrap_summary": {
    "mean_ppl": 0.0,
    "std_ppl": 0.0,
    "cv_pct": 0.0,
    "best_delta_pct": 0.0,
    "worst_delta_pct": 0.0,
    "median_ppl": 0.0,
    "iqr_ppl": 0.0
  },
  "kill_criteria": {
    "K1_cv_pct": 0.0,
    "K1_threshold": 5.0,
    "K1_pass": true,
    "K2_best_delta_pct": 0.0,
    "K2_threshold": 10.0,
    "K2_pass": true,
    "K3_worst_delta_pct": 0.0,
    "K3_threshold": 15.0,
    "K3_pass": true,
    "overall": "SURVIVES"
  },
  "bootstrap_details": [
    {
      "bootstrap_id": 0,
      "n_experts": 40,
      "subset": ["adapter_name_1", "adapter_name_2", "..."],
      "ppl": 0.0,
      "delta_from_ref_pct": 0.0,
      "delta_from_base_pct": 0.0
    }
  ],
  "elapsed_s": 0.0,
  "per_phase_timing": {
    "phase0_setup_s": 0.0,
    "phase1_reference_s": 0.0,
    "phase2_base_s": 0.0,
    "phase3_bootstrap_s": 0.0,
    "phase4_analysis_s": 0.0
  }
}
```

## Constraints

- **Max runtime:** 15 minutes (budget: ~$0.04 at $0.16/hr A5000)
- **Expected GPU memory:** ~10GB (4-bit Qwen2.5-7B ~5GB + adapter ~0.5GB + activations ~4GB)
- **Must support SMOKE_TEST=1:** When set:
  - N_EXPERTS = 6 (first 6 adapters only)
  - N_BOOTSTRAP = 3
  - CALIB_SAMPLES = 3
  - MAX_SEQ_LEN = 128
  - Should complete in < 60 seconds
- **No external data downloads.** All data is on disk.
- **Memory discipline:** One model in GPU at a time. Explicit cleanup between iterations.
- **Numerical stability:** Compute loss in float32 (even though model runs in bfloat16).
  Use `reduction="sum"` in CrossEntropyLoss and track total_tokens manually.

## Existing Script Note

There is an existing script at `macro/composition_dropout_robustness/run_dropout_robustness.py`.
The experiment-programmer should **review and fix** this script rather than rewriting
from scratch. Key issues to verify:

1. The compose_adapters_on_cpu function sums all deltas (correct for SOLE sum composition)
   but should be verified that adapter_config.json scaling is handled correctly
   (alpha/r scaling may be applied by PEFT on load, so raw safetensors values
   may already include the scaling factor).

2. The calibration data loading relies on training data in `/workspace/llm/data/distillation/`.
   Verify this path matches the actual data layout on RunPod.

3. Each bootstrap iteration reloads the base model from scratch. This is correct
   but slow (~30s per reload). An optimization would be to load once and use
   PEFT merge/unmerge, but the current approach is safer for memory isolation.

4. The script correctly uses bfloat16 for model loading and float32 intermediate
   for loss computation (via CrossEntropyLoss defaults).

5. SMOKE_TEST configuration looks correct (6 experts, 3 subsets, 3 calib samples).

If the existing script is structurally sound, only apply targeted fixes.
If it needs major changes, rewrite following this spec.

## Key Implementation Notes

1. **Adapter composition is additive (sum), not averaged.** Each adapter's full
   delta is added. Do NOT divide by N or k. This matches SOLE production config.

2. **Adapter config compatibility:** All 50 pilot adapters share the same config
   (rank-16, all-modules, same target_modules). Copying config from any adapter
   works for the composed output.

3. **Random seed discipline:** Use a seeded `random.Random` instance (not global
   state) for subset generation. This ensures reproducibility independent of
   other random operations.

4. **PPL computation:** Use per-token cross-entropy loss summed over all
   calibration texts, then `exp(total_loss / total_tokens)`. This gives a
   single aggregate PPL, not an average of per-text PPLs (which would be
   biased toward short texts).

5. **Delta computation direction:** `delta = (PPL_subset - PPL_ref) / PPL_ref * 100`.
   Positive delta = subset is worse than reference (lost quality from dropping experts).
   Negative delta = subset is better (dropped experts were harmful).

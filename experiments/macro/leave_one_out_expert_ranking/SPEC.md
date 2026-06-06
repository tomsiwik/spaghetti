# Experiment Spec: leave_one_out_expert_ranking

## Objective

Rank all N=50 pilot LoRA adapters by their contribution to composed model
quality, using leave-one-out perplexity deltas on generic calibration text.
Determine whether the ranking is meaningful (K1: sufficient variance),
practical (K2: completes in time), and stable (K3: consistent across
different calibration sets).

**Key context:** KL divergence was KILLED as a composition diagnostic because
unsupervised distributional distance anti-correlates with quality (rho=-0.7).
LOO-PPL measures absolute model quality, not distance from base. This is the
critical distinction.

## Model & Data

- **Base model:** Qwen/Qwen2.5-7B, loaded with 4-bit NF4 quantization
  from `/workspace/hf_cache` (or download if not cached)
- **Adapters:** All adapters at `/workspace/llm/adapters/` (expect ~50).
  Each is a PEFT LoRA with `adapter_model.safetensors` and `adapter_config.json`.
- **Calibration data:** 60 hardcoded diverse texts (see Section below).
  Split into two disjoint sets of 30 each. NO external data downloads needed.

## Calibration Texts

Hardcode exactly 60 short texts (~100-200 words each), covering:
- 10 Wikipedia/factual (general knowledge, geography, history, science)
- 10 code (Python functions, SQL queries, bash scripts, JS, Rust)
- 10 math/science (equations, proofs, physics, chemistry descriptions)
- 10 conversational/QA (questions, instructions, how-to prompts)
- 10 creative/literary (narrative fiction, poetry, descriptions)
- 10 technical/professional (medical notes, legal text, business writing)

Assign odd-indexed texts (0, 2, 4, ...) to Set A and even-indexed (1, 3, 5, ...)
to Set B. This ensures both sets have identical domain composition (5 of each
category per set).

Truncate each text to 512 tokens. This gives ~15,000 tokens per set.

## Procedure

### Phase 0: Setup (~20s)

1. Discover all adapter directories in `/workspace/llm/adapters/`
2. Sort alphabetically for reproducibility
3. Select first N_EXPERTS (50 full, or controlled by SMOKE_TEST)
4. Log adapter names and count

### Phase 1: Load Base Model + Tokenizer (~20s)

1. Load Qwen2.5-7B with 4-bit NF4 quantization, `device_map="auto"`
2. Load tokenizer with `trust_remote_code=True`
3. Set pad_token = eos_token if not set
4. Keep base model in GPU memory for entire experiment

### Phase 2: Compute Reference PPL (all N composed) (~30s)

1. Compose all N adapters via CPU weight merge:
   - For each adapter, load its `adapter_model.safetensors` on CPU
   - Accumulate: `composed[key] += tensor.float()` for all keys
   - Save composed adapter to tmpdir as `adapter_model.safetensors`
   - Copy `adapter_config.json` from first adapter
2. Load composed adapter onto base model via `PeftModel.from_pretrained()`
3. Compute PPL on Set A: `ref_ppl_a`
4. Compute PPL on Set B: `ref_ppl_b`
5. **Merge adapter into base weights** via `model.merge_and_unload()`
   This gives us W_composed in GPU memory for the subtraction approach.
6. Store the full composed state dict keys/values for reference.
7. Record wall-clock time.

### Phase 3: Leave-One-Out Loop (N iterations) (~15-25 min)

**Approach: Subtraction method (fast path)**

Since W_{-i} = W_composed - B_i @ A_i, we can subtract expert i's delta
directly from the merged model weights rather than re-composing N-1 from scratch.

For each expert i = 1..N:
1. Load expert i's adapter tensors on CPU
2. For each weight key in the adapter:
   - Subtract the tensor from the corresponding base model parameter:
     `param.data -= delta_tensor.to(param.device, param.dtype)`
3. Compute PPL on Set A: `ppl_a_i`
4. Compute PPL on Set B: `ppl_b_i`
5. Restore weights by adding the delta back:
   `param.data += delta_tensor.to(param.device, param.dtype)`
6. Compute deltas:
   - `delta_a_i = (ppl_a_i - ref_ppl_a) / ref_ppl_a * 100`
   - `delta_b_i = (ppl_b_i - ref_ppl_b) / ref_ppl_b * 100`
7. Log: expert name, delta_a, delta_b
8. Free CPU tensors

**Numerical safety check:** Every 10 iterations, verify that the model weights
match the original composed weights (compute L2 norm of difference). If drift
exceeds 1e-4, re-merge all N adapters from scratch and log a warning.

**SMOKE_TEST shortcut:** Use N=5 adapters, 6 calibration texts per set.

### Phase 4: Analysis (~1s)

1. Collect all deltas: `deltas_a = [delta_a_1, ..., delta_a_N]` (same for B)
2. **K1 check:** `sigma = std(deltas_a)`. PASS if sigma >= 0.1%
3. **K2 check:** Total elapsed time. PASS if <= 4 hours (14400s)
4. **K3 check:** Kendall tau-b between deltas_a and deltas_b using scipy.
   PASS if tau >= 0.5
5. Sort experts by delta_a (ascending = worst to best contribution)
6. Count harmful (delta < 0), neutral (-0.1% < delta < 0.1%), helpful (delta > 0.1%)
7. Identify top-5 most helpful and top-5 most harmful

### Phase 5: Bonus Analysis (if time permits, ~2 min)

If pilot50_benchmark.json exists at `/workspace/llm/results/pilot50_benchmark.json`:
1. Load per-adapter PPL improvement percentages
2. Compute Spearman rank correlation between LOO delta_a and benchmark PPL improvement
3. This tests whether LOO ranking correlates with individual adapter quality
   (complementary to K3 which tests self-consistency)

Note: The correlation may be weak because LOO measures CONTRIBUTION TO COMPOSITION
(depends on other experts present) while benchmark measures INDIVIDUAL QUALITY.
Under orthogonality, these should correlate; under interaction effects, they may diverge.

## Kill Criteria Assessment

- **K1:** `std(delta_ppl_a_pct) >= 0.1%` --> PASS; `< 0.1%` --> KILL
  - Primary metric: standard deviation of LOO deltas on Set A
  - Interpretation: ranking has enough variance to be meaningful
  - Supplementary: also report range (max - min) and IQR

- **K2:** `total_elapsed_s <= 14400` (4 hours) --> PASS; `> 14400` --> KILL
  - The subtraction approach should complete in ~20 min
  - Even CPU-merge fallback should complete in ~1 hour
  - This criterion is very conservative

- **K3:** `kendall_tau >= 0.5` --> PASS; `< 0.5` --> KILL
  - Kendall tau-b between Set A and Set B rankings
  - Also report p-value (should be < 0.001 for tau >= 0.5 at N=50)
  - If tau >= 0.7: strong stability (reliable for production use)
  - If 0.5 <= tau < 0.7: moderate stability (usable with caution)
  - If tau < 0.5: ranking is unstable, not useful

## Output

Save results to: `/workspace/llm/results/leave_one_out_expert_ranking/results.json`

Required fields in JSON:
```json
{
  "config": {
    "base_model": "Qwen/Qwen2.5-7B",
    "n_experts": 50,
    "n_calibration_texts_per_set": 30,
    "max_seq_len": 512,
    "quantization": "nf4_4bit",
    "composition_method": "subtraction",
    "smoke_test": false,
    "seed": 42
  },
  "reference_ppl": {
    "set_a": 0.0,
    "set_b": 0.0
  },
  "rankings": {
    "adapter_name": {
      "delta_ppl_a_pct": 0.0,
      "delta_ppl_b_pct": 0.0,
      "ppl_a": 0.0,
      "ppl_b": 0.0
    }
  },
  "rank_order_worst_to_best": ["adapter_name_worst", "...", "adapter_name_best"],
  "n_harmful": 0,
  "n_neutral": 0,
  "n_helpful": 0,
  "top5_harmful": [{"name": "...", "delta_a": 0.0, "delta_b": 0.0}],
  "top5_helpful": [{"name": "...", "delta_a": 0.0, "delta_b": 0.0}],
  "kill_criteria": {
    "K1_delta_std_pct": 0.0,
    "K1_threshold": 0.1,
    "K1_pass": false,
    "K1_delta_range_pct": 0.0,
    "K1_delta_iqr_pct": 0.0,
    "K2_elapsed_s": 0.0,
    "K2_threshold_s": 14400,
    "K2_pass": false,
    "K3_kendall_tau": 0.0,
    "K3_kendall_p": 0.0,
    "K3_threshold": 0.5,
    "K3_pass": false,
    "verdict": "PENDING"
  },
  "bonus_correlation": {
    "available": false,
    "spearman_rho": null,
    "spearman_p": null,
    "note": "Correlation between LOO delta and individual adapter PPL improvement"
  },
  "numerical_safety": {
    "drift_checks": 0,
    "max_drift_l2": 0.0,
    "remerge_count": 0
  },
  "timing": {
    "total_elapsed_s": 0.0,
    "phase1_load_s": 0.0,
    "phase2_reference_s": 0.0,
    "phase3_loo_total_s": 0.0,
    "phase3_per_expert_mean_s": 0.0,
    "phase4_analysis_s": 0.0,
    "phase5_bonus_s": 0.0
  }
}
```

## Constraints

- **Max runtime:** 60 minutes (budget: ~$0.16/hr A5000). Expected: ~20 min.
- **Expected GPU memory:** ~10GB with 4-bit NF4 + merged weights
- **Must support SMOKE_TEST=1:** When set, use N=5 adapters, 6 calibration texts
  per set. Should complete in < 90 seconds.
- **No external data downloads.** All calibration texts are hardcoded in the script.
- **Memory discipline:** Only one model in GPU memory at all times. Expert deltas
  loaded to CPU, moved to GPU only for the subtract/add operation.
- **Numerical stability:** PPL computation in float32 for loss accumulation even
  if model runs in bfloat16. Use log-space for cross-entropy (avoid exp overflow).

## Key Implementation Notes

1. **Subtraction approach is the preferred method.** Merge all N adapters into
   base weights once, then subtract/add individual expert deltas. This avoids
   N PEFT load/unload cycles and is ~10x faster than full recomposition.

2. **The `merge_and_unload()` call is irreversible.** After merging, the PEFT
   wrapper is gone. All weight manipulation is direct tensor arithmetic on
   the base model parameters. This is fine -- we only need the merged model.

3. **Expert delta loading:** Each adapter's safetensors contains keys like
   `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`. Map these
   to the base model parameter names by stripping LoRA-specific prefixes.
   The PEFT adapter_config.json specifies target_modules and rank.

4. **Calibration text quality matters.** The texts should be high-quality,
   grammatically correct, diverse in domain, and not adversarial. They should
   represent "typical language model input" -- not edge cases.

5. **The existing script** at `macro/leave_one_out_expert_ranking/run_leave_one_out.py`
   has the right structure but uses the slow recomposition approach (reloads
   base model for each LOO). The experiment-programmer should implement the
   subtraction approach described here, or use the existing script as a starting
   point and optimize it.

6. **LoRA weight key mapping:** PEFT stores adapter weights with names like
   `base_model.model.model.layers.{L}.{module}.lora_A.weight` and `lora_B.weight`.
   To subtract the composed delta from the model:
   ```
   delta = lora_B @ lora_A  # shape matches the target parameter
   model.param.data -= delta.to(device, dtype)
   ```
   The adapter_config.json has `target_modules` listing which modules have LoRA.
   At rank-16, delta = B (d_out, 16) @ A (16, d_in) -> (d_out, d_in).

7. **PPL computation:** Process one text at a time (not batched) to avoid
   padding artifacts. Use `torch.no_grad()` context. Accumulate total
   cross-entropy loss and total token count across all texts, then
   `PPL = exp(total_loss / total_tokens)`.

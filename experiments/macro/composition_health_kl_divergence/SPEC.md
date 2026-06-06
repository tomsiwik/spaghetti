# Experiment Spec: composition_health_kl_divergence

## Objective

Determine whether KL divergence between composed-model and base-model logit
distributions on fixed calibration tokens can detect harmful expert additions
without requiring any task labels or per-expert evaluation data.

Three questions:
1. Does KL(composed || base) correlate with composition quality loss? (K1)
2. Does a harmful expert produce a distinguishable KL spike? (K2)
3. Is the measurement fast enough for production use? (K3)

## Model & Data

- **Base model:** Qwen2.5-7B at `/workspace/models/Qwen2.5-7B` (4-bit NF4)
- **Adapters:** 50 pilot adapters at `/workspace/llm/adapters/`
- **Quality reference:** `/workspace/llm/results/pilot50_benchmark.json`
  (per-adapter PPL improvement percentages; contaminated but valid for ranking)
- **Calibration data:** 20 hardcoded diverse texts (see below), domain-agnostic.
  No external dataset download required.
- **Training data:** `/workspace/llm/training_data/{domain}.jsonl` for
  per-domain PPL evaluation (if available; otherwise use benchmark JSON)

## Calibration Texts

Use exactly 20 short texts (~50 tokens each), covering:
- 4 Wikipedia/factual (general knowledge, no domain overlap with adapters)
- 4 code (Python, SQL, bash, generic programming)
- 4 math/science (equations, physics, chemistry)
- 4 conversational/QA (questions, instructions)
- 4 creative/literary (narrative, poetry, descriptions)

These are hardcoded in the script. No external data needed.

## Procedure

### Phase 1: Base Model Calibration (~15s)

1. Load Qwen2.5-7B with 4-bit NF4 quantization
2. Tokenize all 20 calibration texts (truncate to 256 tokens)
3. Forward pass on each text, collect logits at last token position
4. Store as `base_logits`: list of 20 tensors, each shape (V,) in float32
5. Record wall-clock time

### Phase 2: KL vs N Sweep (~60s)

For each N in [5, 10, 25, 50]:
1. Select the top-N adapters (by PPL improvement from benchmark JSON)
2. Load all N adapters into PEFT model using `PeftModel.from_pretrained`
   and `load_adapter` for subsequent adapters
3. Merge via `add_weighted_adapter(adapters, weights=[1.0]*N,
   combination_type="linear", adapter_name="composed")`
4. Set active adapter to "composed"
5. Forward pass on all 20 calibration texts, collect composed logits
6. Compute KL(P_composed || P_base) for each text:
   ```python
   p_comp = F.log_softmax(composed_logits, dim=-1)
   p_base = F.log_softmax(base_logits, dim=-1)
   kl = F.kl_div(p_base, p_comp, log_target=True, reduction="sum")
   ```
7. Record mean KL, std KL, per-text KL values, and wall-clock time
8. Delete PEFT model and free GPU memory before next N

**IMPORTANT memory management:** After each N, explicitly delete the PEFT
model and call `torch.cuda.empty_cache()`. Do NOT keep multiple models in
memory. The base model must be reloaded for each N (PEFT modifies in-place).

**Alternative (faster):** If adapter merging into base weights is possible
via `merge_and_unload()`, do that and reuse the base model. But verify
that unmerging properly restores original weights.

### Phase 3: Leave-One-Out at N=10 (~120s)

Use the top-10 adapters (by PPL improvement).

1. Compute KL for the full N=10 composition (reuse from Phase 2 if available)
2. For each of the 10 adapters:
   a. Compose the remaining 9 adapters (same procedure as Phase 2)
   b. Compute KL(P_9 || P_base) on calibration texts
   c. Compute DeltaKL_i = KL_10 - KL_9_without_i
   d. Record per-adapter DeltaKL, wall-clock time
   e. Free GPU memory

3. Compute z-scores: z_i = (DeltaKL_i - mean(DeltaKL)) / std(DeltaKL)

### Phase 4: Synthetic Harmful Expert (~30s)

Create one deliberately harmful adapter to test discrimination:

1. Take the adapter with highest PPL improvement (best quality)
2. Create a "harmful" version by NEGATING its B-matrix weights:
   ```python
   for key in state_dict:
       if "lora_B" in key:
           state_dict[key] = -state_dict[key]
   ```
   Save to a temporary directory as "harmful_expert"
3. Compose top-9 adapters + harmful_expert (replacing the original)
4. Compute KL(P_10_with_harmful || P_base)
5. Compute DeltaKL_harmful and compare to DeltaKL distribution from Phase 3

### Phase 5: Correlation with Quality Impact (~30s)

1. Load per-adapter PPL improvements from pilot50_benchmark.json
2. For the 10 adapters used in Phase 3:
   - DeltaKL_i from Phase 3
   - PPL_improvement_i from benchmark (higher = better quality)
3. Compute Spearman rank correlation between DeltaKL and (1 - PPL_improvement)
   - Note: higher DeltaKL should correlate with WORSE quality (lower improvement)
   - So we expect negative correlation between DeltaKL and PPL_improvement
   - Or equivalently, positive correlation between DeltaKL and PPL_ratio (base/expert)
4. Record rho, p-value

### Phase 6: Per-Domain PPL Under Composition (~120s, optional)

If training data is available at `/workspace/llm/training_data/`:
1. For each of the 10 adapters, load 20 examples from its domain file
2. Compute PPL of the N=10 composed model on each domain
3. Compute PPL of the N=9 (leave-one-out) composed model on each domain
4. Cross-impact: does removing adapter i change PPL on adapter j's domain?
5. Correlate DeltaKL with cross-domain PPL changes

This phase is optional -- skip if training data is not available or if
runtime exceeds budget. The Phase 5 correlation is the primary K1 test.

## Kill Criteria Assessment

- **K1:** Spearman rho(DeltaKL, quality_loss) >= 0.3 --> PASS; < 0.3 --> KILL
  - Primary metric: rank correlation at N=10
  - Quality loss = (1 - PPL_improvement_pct/100) from benchmark
  - If rho > 0.5, strong pass; if 0.3-0.5, marginal pass; if < 0.3, kill

- **K2:** DeltaKL_harmful > mean(DeltaKL) + 2*std(DeltaKL) --> PASS
  - Harmful expert from Phase 4 must have z-score > 2.0
  - Additionally: z-score for worst natural adapter should be > 1.0
  - If both conditions hold, K2 passes strongly

- **K3:** Wall-clock time for one KL measurement (Phase 1 + one composition
  forward pass) < 30 seconds --> PASS
  - Measured as: time for Phase 2 at N=10 / 1 (per-composition time)
  - Include adapter loading + merge + forward + KL computation

## Output

Save results to: `/workspace/llm/results/composition_health_kl_divergence/results.json`

Required fields in JSON:
```json
{
  "config": {
    "base_model": "Qwen2.5-7B",
    "n_calibration_texts": 20,
    "n_sweep": [5, 10, 25, 50],
    "seed": 42,
    "total_adapters": 50,
    "quantization": "nf4_4bit"
  },
  "phase1_base_calibration": {
    "elapsed_s": 0.0,
    "n_texts": 20
  },
  "phase2_kl_vs_n": {
    "5": {"mean_kl": 0.0, "std_kl": 0.0, "per_text_kl": [], "elapsed_s": 0.0},
    "10": {"mean_kl": 0.0, "std_kl": 0.0, "per_text_kl": [], "elapsed_s": 0.0},
    "25": {"mean_kl": 0.0, "std_kl": 0.0, "per_text_kl": [], "elapsed_s": 0.0},
    "50": {"mean_kl": 0.0, "std_kl": 0.0, "per_text_kl": [], "elapsed_s": 0.0}
  },
  "phase3_leave_one_out": {
    "n_test": 10,
    "kl_all_10": 0.0,
    "per_adapter": {
      "adapter_name": {
        "kl_without": 0.0,
        "delta_kl": 0.0,
        "z_score": 0.0,
        "ppl_improvement_pct": 0.0,
        "elapsed_s": 0.0
      }
    }
  },
  "phase4_harmful_expert": {
    "harmful_method": "negated_lora_B",
    "source_adapter": "name",
    "kl_with_harmful": 0.0,
    "delta_kl_harmful": 0.0,
    "z_score_harmful": 0.0,
    "discrimination": true
  },
  "phase5_correlation": {
    "spearman_rho": 0.0,
    "spearman_p": 0.0,
    "n_samples": 10,
    "note": "rho between DeltaKL and (1 - ppl_improvement_pct/100)"
  },
  "kill_criteria": {
    "K1_spearman_rho": 0.0,
    "K1_threshold": 0.3,
    "K1_pass": false,
    "K2_z_score_harmful": 0.0,
    "K2_z_score_worst_natural": 0.0,
    "K2_threshold": 2.0,
    "K2_pass": false,
    "K3_time_per_composition_s": 0.0,
    "K3_threshold_s": 30,
    "K3_pass": false,
    "verdict": "PENDING"
  },
  "timing": {
    "total_elapsed_s": 0.0,
    "phase1_s": 0.0,
    "phase2_s": 0.0,
    "phase3_s": 0.0,
    "phase4_s": 0.0,
    "phase5_s": 0.0
  }
}
```

## Constraints

- **Max runtime:** 15 minutes (budget-critical: ~$0.16/hr A5000)
- **Expected GPU memory:** ~10GB with 4-bit quantization + PEFT adapters
- **Must support SMOKE_TEST=1:** When set, use N_SWEEP=[2, 3], leave-one-out
  at N=3, skip Phase 4 and Phase 6. Should complete in < 60 seconds.
- **No external data downloads.** Calibration texts are hardcoded. Quality
  data comes from pilot50_benchmark.json.
- **Memory discipline:** Delete PEFT models after each composition. Never
  keep more than one composed model in GPU memory.
- **Numerical stability:** Use float32 for all KL computations (even if
  model runs in float16). Apply log_softmax rather than softmax + log
  to avoid underflow.

## Key Implementation Notes

1. **Adapter ordering matters for reproducibility.** Sort adapters by PPL
   improvement (descending) from benchmark JSON. Use this fixed ordering
   for all phases.

2. **PEFT add_weighted_adapter quirk:** This method creates a new virtual
   adapter by linearly combining the loaded adapters. It does NOT modify
   the base model weights. The composed adapter must be set as active.

3. **KL direction:** We compute KL(P_composed || P_base), NOT the reverse.
   Use `F.kl_div(log_p_base, log_p_composed, log_target=True)` which
   computes sum(exp(log_p_composed) * (log_p_composed - log_p_base)).

4. **The harmful expert must be saved to disk** (PEFT loads from paths).
   Create a temporary directory, copy adapter_config.json, write the
   negated weights. Clean up after Phase 4.

5. **Existing script at macro/composition_health_kl_divergence/run_kl_health.py
   has structural issues.** The experiment-programmer should write a fresh
   script following this spec rather than fixing the existing one. Key issues
   in the existing script:
   - Reloads base model for every leave-one-out (extremely slow)
   - apply_composed_delta is a stub (pass)
   - No synthetic harmful expert test
   - No correlation with quality data
   - Calibration texts are reasonable but Phase structure is incomplete

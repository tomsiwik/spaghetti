# Experiment Spec: composition_weight_normalization

## Objective

Determine the optimal scaling factor $\alpha(N)$ for composing $N$ independently-trained
LoRA adapters via weight addition, and test whether it follows a power law $N^{-\beta}$.
Specifically: does $1/\sqrt{N}$ (the random-subspace prediction) outperform $1/N$
(the fully-correlated prediction) at high adapter counts (N=25, N=50)?

## Model & Data

- **Base model:** Qwen/Qwen2.5-7B (NF4 4-bit quantized, same as all prior SOLE experiments)
- **Adapters:** rank-16, all-modules LoRA adapters from the pilot-50 distillation pipeline,
  stored at `/workspace/llm/adapters/` (one subfolder per adapter with adapter_model.safetensors + adapter_config.json)
- **Eval data:** Per-adapter domain data at `/workspace/llm/data/distillation/{domain}/`,
  using eval.jsonl if available, else tail of train.jsonl (same as prior experiments;
  contaminated but relative comparisons are valid). 50 samples per domain, max 512 tokens.
- **Expected adapter count:** ~47 usable adapters (from pilot-50, minus failures)

## Procedure

### Phase 1: Single-Expert Baseline (for K2)

For each adapter domain (up to all available):
1. Load base model + single adapter via PeftModel
2. Evaluate per-domain PPL on that adapter's eval data (50 samples, 512 tokens)
3. Record single-expert PPL per domain
4. Compute average single-expert PPL across all domains

### Phase 2: Scaling Strategies (for K1, K3)

For each $N \in \{5, 10, 25, 50\}$ (or max available):
1. Select the first $N$ adapters (sorted alphabetically by folder name)
2. For each strategy in {unit ($w=1.0$), mean ($w=1/N$), sqrt ($w=1/\sqrt{N}$)}:
   a. Compose $N$ adapters on CPU: weighted sum of safetensors, save to temp dir
   b. Load base model + composed adapter via PeftModel
   c. Evaluate per-domain PPL on all domains that are in the composition
   d. Record per-domain and mean PPL
   e. Clean up temp dir and GPU memory

### Phase 3: Grid Search (for K2, power-law fitting)

At the largest $N$:
1. Test uniform weights $w \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$
2. For each, compose on CPU and evaluate same domains
3. Identify best grid weight by lowest mean PPL
4. Also run best grid weight at $N=N_{max}/2$ (for K3 transfer check)

### Phase 4: Kill Criteria Evaluation

Compute all three kill criteria from collected data.

## Kill Criteria Assessment

### K1: 1/sqrt(N) reduces composed PPL by >50% vs unit-weight at N=50

```
unit_ppl = mean PPL of unit-weight composition at largest N
sqrt_ppl = mean PPL of 1/sqrt(N) composition at largest N
reduction_pct = (unit_ppl - sqrt_ppl) / unit_ppl * 100
K1_PASS = reduction_pct > 50.0
```

**Note:** If unit-weight PPL is infinity or trillions (as expected from prior N=5 data),
K1 is trivially satisfied. The interesting case is whether $1/\sqrt{N}$ is also substantially
better than $1/N$.

### K2: Best scaling factor produces composed PPL < 2x individual expert average

```
best_ppl = min(grid_search_mean_ppls)
avg_single = mean(single_expert_ppls)
K2_PASS = best_ppl / avg_single <= 2.0
```

**Interpretation:** If K2 fails, even optimal scaling cannot recover composition quality.
This would indicate destructive interference (sign conflicts, not just magnitude).

### K3: Scaling factor transfers across N values

```
best_w = weight that minimizes PPL at N=max
transfer_ppl = PPL at N=max/2 using best_w
K3_PASS = transfer_ppl / best_ppl < 2.0 AND transfer_ppl < 3 * avg_single_ppl
```

**Interpretation:** If K3 fails, the optimal weight depends on the specific adapter set
and count, not on a stable power law. Would imply per-composition tuning is required.

## Output

Save results to: `results/composition_weight_normalization/weight_normalization_results.json`

### Required JSON fields:

```json
{
  "timestamp": "ISO 8601",
  "config": {
    "n_values": [5, 10, 25, 50],
    "eval_samples_per_domain": 50,
    "max_seq_len": 512,
    "total_adapters": 47,
    "eval_domains": 47,
    "smoke_test": false
  },
  "single_expert_avg_ppl": float,
  "single_expert_ppls": {"domain": float, ...},
  "scaling_results": {
    "strategy": {
      "N": {
        "per_domain_ppl": {"domain": float},
        "mean_ppl": float,
        "weight": float
      }
    }
  },
  "grid_search": {
    "n": int,
    "results": {"weight": {"per_domain_ppl": {...}, "mean_ppl": float}},
    "best_weight": float,
    "best_ppl": float
  },
  "transfer_check": {
    "n": int,
    "weight": float,
    "mean_ppl": float,
    "per_domain_ppl": {...}
  },
  "kill_criteria": {
    "K1_sqrt_reduces_50pct": {"unit_ppl": float, "sqrt_ppl": float, "reduction_pct": float, "pass": bool},
    "K2_best_lt_2x_single": {"best_ppl": float, "avg_single_ppl": float, "ratio": float, "pass": bool},
    "K3_transfers_across_n": {"n_source": int, "n_target": int, "ratio": float, "pass": bool}
  },
  "verdict": "PASS|KILLED",
  "killed_criteria": ["K1"|"K2"|"K3"],
  "elapsed_s": float
}
```

### Additional derived outputs (for PAPER.md analysis):

From the scaling_results, fit a power law $\alpha^*(N) = N^{-\beta}$ to the
grid-search best weights at each N. Report $\beta$ and $R^2$. Compare to
theoretical prediction $\beta \approx 0.52$ from MATH.md.

## Constraints

- **Max runtime:** 120 min (estimate: ~90 min with 47 adapters, 19 compositions, 50 samples each)
- **Expected GPU memory:** ~16GB with 4-bit base + one adapter at a time
- **Must support SMOKE_TEST=1:** N_VALUES=[3,5], EVAL_SAMPLES=5, MAX_SEQ_LEN=256
- **GPU:** A5000 (24GB) on RunPod
- **No training:** inference-only experiment
- **CPU composition:** Adapters are composed on CPU (safetensors load + weighted sum), then loaded to GPU as single PeftModel. This avoids OOM from loading N adapters simultaneously.

## Script Location

The GPU script already exists at:
`macro/composition_weight_normalization/run_weight_normalization.py`

No script changes needed unless the experiment-programmer identifies issues during review.

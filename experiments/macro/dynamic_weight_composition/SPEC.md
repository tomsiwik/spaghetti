# Experiment Spec: dynamic_weight_composition

## Objective

Compare 5 expert weighting strategies on 50 pilot LoRA adapters to determine
whether dynamic (per-query) weighting beats equal-weight pre-merge at macro scale.
This is the key dilution-fix experiment for SOLE Phase 2: Compose.

## Model & Data

- **Base model:** Qwen/Qwen2.5-7B, 4-bit NF4 quantization, bfloat16 compute
- **Adapters:** 50 pilot adapters at `/workspace/llm/adapters/` (rank-16, all-modules)
- **Eval data:** Per-domain eval texts at `/workspace/llm/data/distillation/{domain}/eval.jsonl`
  (fall back to last 200 lines of train.jsonl if eval.jsonl missing)
- **HF cache:** `/workspace/hf_cache`
- **Results:** `/workspace/llm/results/dynamic_weight_composition/`

## Procedure

### Phase 0: Discovery & Centroids (~15 min)

1. Discover all adapters in `/workspace/llm/adapters/` that have
   `adapter_model.safetensors`. Record the list.
2. For each adapter/domain, check eval data availability in
   `/workspace/llm/data/distillation/{domain}/`.
3. Load base model (4-bit NF4, device_map="auto").
4. **Compute domain centroids.** For each of the 50 domains:
   a. Load 100 training examples (first 100 from train.jsonl)
   b. Tokenize and forward through base model (no adapters), max_len=256
   c. Extract last hidden state, mean-pool across tokens and examples
   d. L2-normalize to get centroid c_j of shape (d,) where d=3584
   e. Save centroids matrix C of shape (50, 3584) to results dir

### Phase 1: Base & Single-Expert Baselines (~20 min)

5. Measure base model PPL on all domains (50 samples each, max_len=512).
6. Measure single-expert PPL: for each domain, load its adapter alone,
   measure PPL on that domain's eval data. Use PEFT `PeftModel.from_pretrained`,
   then delete and `torch.cuda.empty_cache()` between domains.
   Output: dict of {domain: ppl}.

### Phase 2: Strategy Evaluation (~40 min)

For each eval domain d in {all 50 domains} (or a representative 15 if
time-constrained), evaluate these strategies at N=50:

**Strategy A: equal_premerge**
7. Load all 50 adapters, merge with `add_weighted_adapter` at weight=1/50 each.
   Measure PPL on domain d.

**Strategy B2: embed_topk (k=5)**
8. Compute query embedding: forward eval text through base model, mean-pool
   last hidden state, L2-normalize. Compute cosine similarity to all 50 centroids.
   Select top-5 experts by cosine. Merge those 5 with equal weight 1/5.
   Measure PPL on domain d.

**Strategy C2: embed_weighted (all 50)**
9. Use same cosine scores as B2 but for all 50 experts. Apply softmax(tau=1.0)
   to get weights. Merge all 50 with these weights using `add_weighted_adapter`.
   Measure PPL on domain d.

**Strategy D: ppl_precomputed**
10. Precompute: for each expert j, measure its PPL on each domain's eval data
    (or a 10-sample subset). This produces a quality matrix Q of shape (50, 50).
    At eval time, use column d of Q as weights: w_j = softmax(-Q[j,d] / tau).
    Merge with these weights. Measure PPL.

**IMPORTANT optimization:** Strategies A, B2, C2, D all produce different weight
vectors but use the same PEFT merge mechanism. To avoid reloading 50 adapters
for each strategy x domain combination:

- Load all 50 adapters once into a PeftModel
- For each domain and strategy, compute weights, call `add_weighted_adapter`
  with those weights, measure PPL, then delete the composed adapter
- This keeps adapter loading cost constant

**Strategy C3: hybrid_k3 (optional, time permitting)**
11. Embed top-3 filter, then for each of the 3 selected experts, measure PPL
    of (base + that single expert) on the eval query. Softmax the 3 PPL scores.
    Merge top-3 with these weights. Measure PPL. Also measure wall-clock latency.

### Phase 3: Oracle Reference (optional, if time permits)

12. For a subset of 10 domains: evaluate (base + Delta_j) on domain d for
    all 50 experts j. Use the resulting 50 PPL scores as oracle weights.
    This establishes the upper bound for dynamic weighting.

### Phase 4: Latency Measurement

13. For each strategy, time 100 iterations of the weight computation step
    (excluding model forward pass for PPL eval) and the merge step.
    Report mean and P99 latency.

### Phase 5: Kill Criteria Assessment

14. Compute for each strategy vs equal-weight pre-merge:
    - Mean PPL improvement across domains
    - Max degradation on any domain
    - Count of domains worse than base
    - Latency overhead per query

## Kill Criteria Assessment

- **K1:** `improvement = (PPL_eq_mean - PPL_best_mean) / PPL_eq_mean * 100`
  Must be >= 2% for PASS. If < 2%, dynamic weighting is not worth the complexity.

- **K2:** Winning dynamic strategy latency overhead must be < 50ms.
  Embedding-based strategies should be ~0.1ms (easy PASS).
  Hybrid strategies with k=3 forward passes: measure and report.

- **K3:** Plot the Pareto frontier of (latency, mean_PPL) across all strategies.
  If equal-weight pre-merge is Pareto-optimal, KILL. If any dynamic strategy
  dominates (better quality at acceptable latency), PASS.

## Output

Save results to: `results/dynamic_weight_composition/results.json`

Required fields in JSON:
```json
{
  "timestamp": "ISO8601",
  "config": {
    "n_experts": 50,
    "n_eval_domains": "<number>",
    "eval_samples_per_domain": 50,
    "max_seq_len": 512,
    "base_model": "Qwen/Qwen2.5-7B",
    "quantization": "nf4_4bit",
    "smoke_test": false
  },
  "centroids": {
    "shape": [50, 3584],
    "saved_to": "centroids.pt"
  },
  "base_ppl": {"domain": "ppl_value"},
  "single_expert_ppl": {"domain": "ppl_value"},
  "strategies": {
    "equal_premerge": {
      "per_domain_ppl": {"domain": "ppl_value"},
      "mean_ppl": "<float>",
      "mean_degradation_vs_single_pct": "<float>",
      "domains_worse_than_base": "<int>",
      "latency_ms": 0.0
    },
    "embed_topk_k5": { "...same fields..." },
    "embed_weighted": { "...same fields..." },
    "ppl_precomputed": { "...same fields..." },
    "hybrid_k3": { "...same fields, plus latency..." }
  },
  "kill_criteria": {
    "K1_improvement_gte_2pct": {
      "best_strategy": "<name>",
      "improvement_pct": "<float>",
      "pass": "<bool>"
    },
    "K2_latency_lt_50ms": {
      "best_strategy_latency_ms": "<float>",
      "pass": "<bool>"
    },
    "K3_pareto": {
      "equal_premerge_on_frontier": "<bool>",
      "dominated_by": "<strategy or null>",
      "pass": "<bool>"
    }
  },
  "verdict": "PASS|FAIL|KILLED",
  "elapsed_s": "<float>"
}
```

Additional output files:
- `centroids.pt`: Precomputed domain centroid embeddings (50 x 3584)
- `quality_matrix.json`: Cross-domain PPL matrix Q[expert, domain] (50 x N_eval)
- `pareto_data.json`: (latency_ms, mean_ppl) tuples for Pareto plot

## Constraints

- **Max runtime:** 90 min (60 min target, 90 min hard cap)
- **Expected GPU memory:** ~16GB with 4-bit quantization + 50 loaded adapters
  (PEFT keeps adapters as separate small tensors, not merged into base)
- **Must support SMOKE_TEST=1:** In smoke mode, use 3 adapters, 2 domains,
  5 eval samples, skip oracle. Should complete in <60s.
- **Adapter loading strategy:** Load all adapters once, reuse for all strategies.
  Do NOT reload the base model between strategies.
- **PEFT version:** Must use `add_weighted_adapter` with `combination_type="linear"`.
  Weights are per-adapter scalars (not per-layer -- PEFT applies the same scalar
  to all modules/layers of a given adapter).

## Design Notes for the Experiment Programmer

1. **The key insight is weight computation, not model architecture.** All strategies
   use the same PEFT merge mechanism. The difference is how w_k is computed.

2. **Centroid computation is the one novel piece.** Forward all training data through
   base model, extract hidden states, mean-pool, normalize. Use `model.eval()`,
   `torch.no_grad()`, batch processing.

3. **add_weighted_adapter gotcha:** PEFT's `add_weighted_adapter` expects a list of
   adapter names and a list of weights. The weights are scalars per adapter
   (not per-layer). After creating a composed adapter, set it active with
   `model.set_adapter("composed_name")`. Delete it after eval with
   `model.delete_adapter("composed_name")`.

4. **Memory management is critical.** With 50 adapters loaded simultaneously,
   memory usage is: base (4-bit ~4GB) + 50 adapters (~300MB total for rank-16).
   Should fit in 24GB A5000. If not, load/unload adapters in batches.

5. **Eval domain sampling:** If 50 domains x 5 strategies exceeds time budget,
   evaluate strategies on a representative subset (e.g., 15 domains spanning
   different base PPL levels: 5 low-PPL, 5 medium, 5 high). Report which
   domains were evaluated.

6. **Quality matrix for ppl_precomputed:** This is the most expensive part.
   Measuring PPL of each expert on each domain requires N^2 evaluations.
   Optimize: use 10 eval samples (not 50) for the quality matrix, and only
   evaluate the subset of domains actually in the eval set. This reduces
   from 50*50*50 = 125K samples to 50*15*10 = 7,500 samples.

7. **Reuse pilot50_composition_quality infrastructure.** The `load_eval_texts`,
   `measure_ppl`, and `compose_and_evaluate` patterns from
   `macro/pilot50_composition_quality/run_composition_quality.py` can be
   directly adapted. Key change: make weights a parameter instead of hardcoded 1/N.

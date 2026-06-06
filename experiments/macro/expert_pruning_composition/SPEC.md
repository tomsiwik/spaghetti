# Experiment Spec: expert_pruning_composition

## Objective

Determine whether pruning low-quality experts from a SOLE composition of N=50
LoRA adapters improves aggregate model quality (PPL and MMLU). Characterize the
quality-quantity tradeoff curve and test whether ranking-based pruning (O(N log N))
is sufficient vs greedy forward selection (O(N^2)).

**Three core questions:**
1. Does removing bottom-20% experts improve composed model PPL by >1%?
2. Is the expert quality ranking stable across metrics (Kendall tau >= 0.6)?
3. Can ranking-based selection match greedy selection (making it scalable to N=500)?

## Model & Data

- **Base model:** Qwen2.5-7B at `/workspace/hf_cache` or `/workspace/models/Qwen2.5-7B`,
  loaded with 4-bit NF4 quantization via BitsAndBytesConfig
- **Adapters:** All adapters at `/workspace/llm/adapters/` (~50 directories).
  Each contains `adapter_model.safetensors` and `adapter_config.json`.
  All rank-16, all-modules (q/k/v/o/gate/up/down).
- **Pilot50 benchmark data:** `/workspace/llm/results/pilot50_benchmark.json` --
  contains per-domain PPL improvement scores for ranking.
- **Calibration data:** 60 hardcoded diverse texts (same as LOO experiment --
  see Calibration Texts section below). Split into Set A (30) and Set B (30).
  Truncated to 512 tokens each. NO external data downloads needed.
- **MMLU data:** Download via `datasets` library (hendrycks/test).
  Use 30 held-out subjects (see MMLU Subjects section).
- **LOO results (if available):** `/workspace/llm/results/leave_one_out_expert_ranking/results.json`.
  If present, incorporate LOO rankings for correlation analysis. If absent, compute
  LOO rankings as part of this experiment (Phase 2).

## Calibration Texts

Hardcode exactly 60 short texts (~100-200 words each), covering 6 categories of 10:
- 10 Wikipedia/factual (general knowledge, geography, history, science)
- 10 code (Python functions, SQL queries, bash scripts, JS, Rust)
- 10 math/science (equations, proofs, physics, chemistry descriptions)
- 10 conversational/QA (questions, instructions, how-to prompts)
- 10 creative/literary (narrative fiction, poetry, descriptions)
- 10 technical/professional (medical notes, legal text, business writing)

Assign odd-indexed texts (0, 2, 4, ...) to Set A and even-indexed (1, 3, 5, ...)
to Set B. Both sets have 5 texts from each category.

Truncate each to 512 tokens.

## MMLU Subjects

Use 30 held-out MMLU subjects (not directly mapped to any adapter domain).
Subjects to use (standard MMLU test split):

```
abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge,
college_biology, college_chemistry, college_computer_science, college_mathematics,
college_medicine, college_physics, computer_security, conceptual_physics,
econometrics, electrical_engineering, formal_logic, global_facts,
high_school_biology, high_school_chemistry, high_school_computer_science,
high_school_mathematics, high_school_physics, high_school_statistics,
high_school_us_history, high_school_world_history, human_aging,
international_law, jurisprudence, logical_fallacies, machine_learning
```

Use 5-shot prompting (standard protocol). Log-probability evaluation: for each
question, compute log P(answer_token | context) for each of A/B/C/D. Pick argmax.

## Procedure

### Phase 0: Setup (~30s)

1. Discover all adapter directories at `/workspace/llm/adapters/`.
   Sort alphabetically. Use all found (expect ~50).
2. Load `pilot50_benchmark.json` to get per-domain quality scores Q_domain_i.
3. Rank adapters by Q_domain_i descending to produce pi_domain.
4. Log: adapter count, quality score range, bottom-20% list.
5. Set random seed = 42. Load tokenizer.

### Phase 1: Load Base Model + Prepare Composed Weights (~60s)

1. Load Qwen2.5-7B with 4-bit NF4, `device_map="auto"`.
2. Compute all-N composed adapter: accumulate `sum delta_i` on CPU.
3. Save composed adapter to tmpdir, load via `PeftModel.from_pretrained`.
4. Merge into base weights via `model.merge_and_unload()`.
   Now GPU has W_composed = W + sum_{i=1}^{N} B_i A_i.
5. Compute reference PPL on calibration Set A and Set B.
6. Compute reference MMLU accuracy on 30 subjects.
7. Record: PPL_ref_a, PPL_ref_b, MMLU_ref.

### Phase 2: LOO Ranking (if LOO results not available) (~15 min)

Check if `/workspace/llm/results/leave_one_out_expert_ranking/results.json` exists.

**If available:** Load LOO rankings directly. Skip to Phase 3.

**If NOT available:** Compute LOO rankings using the subtraction method.

For each expert i = 1..N:
1. Subtract delta_i from composed weights (load adapter tensors on CPU,
   subtract from model params on GPU).
2. Compute PPL on calibration Set A.
3. Restore weights by adding delta_i back.
4. C_i = PPL_{-i} - PPL_ref (positive = expert helps).

Numerical safety: Every 10 iterations, check weight drift. Re-merge if drift
exceeds 1e-4.

Sort by C_i descending to produce pi_loo.

### Phase 3: Accumulation Curve -- Rank-Ordered (PPL) (~25 min)

Start from base model (no adapters). Reload base model fresh (merge_and_unload
is irreversible -- need clean base).

For k in {1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
1. Compose top-k experts by pi_domain ranking via CPU merge.
2. Load composed adapter onto base model.
3. Compute PPL on calibration Set A: PPL_ranked_a(k).
4. Compute PPL on calibration Set B: PPL_ranked_b(k).
5. Unload adapter.

**Optimization:** Instead of reloading base model 14 times, use the merge-then-
subtract pattern. Start with all-N composed. For decreasing k, progressively
subtract the lowest-ranked experts. This avoids repeated base model loads.

Specifically:
- Start with all 50 merged.
- Eval at k=50.
- Subtract expert pi(50) (worst). Eval at k=49.
- But we don't need every k. Jump: eval at k=50, then subtract bottom 5 -> eval
  at k=45, subtract 5 more -> eval at k=40, etc.

**Implementation approach:** Since we need non-contiguous k values, it is simpler
to compose from scratch for each k. Use the additive CPU merge approach:
- For k=50: sum all 50 deltas.
- For k=45: sum top 45 by rank.
- etc.

Each merge takes ~10s on CPU. 14 merge+eval cycles: 14 * (10s + 15s) = ~350s.
Total Phase 3: ~6 min.

### Phase 4: Accumulation Curve -- MMLU (~40 min)

For k in {5, 10, 20, 30, 40, 50}:
1. Compose top-k experts by pi_domain ranking.
2. Evaluate on 30 MMLU subjects (5-shot, log-prob).
3. Record MMLU_ranked(k).

Note: MMLU eval is slower (~5 min per k due to 30 subjects). Reduce to 6 data
points to stay within budget.

### Phase 5: Greedy Forward Selection (PPL only) (~20 min)

Build optimal subset greedily:

```
S = {}
For t = 1 to N:
    best_i = None, best_ppl = infinity
    For each candidate i not in S:
        Compose S union {i}
        Compute PPL on Set A
        If PPL < best_ppl: best_i = i, best_ppl = PPL
    S = S union {best_i}
    Record: greedy_ppl(|S|) = best_ppl
```

Full greedy is O(N^2/2) = 1225 evaluations at N=50. At ~5s each: ~100 min.
TOO SLOW.

**Optimized greedy:** Run greedy only for k=1..10 (first 10 steps):
- Step 1: 50 evals
- Step 2: 49 evals
- ...
- Step 10: 41 evals
- Total: 455 evals * 5s = ~38 min. Acceptable.

Then compare greedy_ppl(k) vs ranked_ppl(k) for k=1..10. If they agree within
0.5%, declare ranking sufficient.

**Alternative fast path:** If greedy step 1 selects the same expert as rank 1,
and greedy step 2 selects rank 2, etc., we can early-stop after confirming the
first 5 greedy steps match ranking (saves ~25 min).

### Phase 6: Bottom-K Removal Test (~5 min)

Compute the specific pruning benefit:

1. PPL(all 50, Set A) -- already computed in Phase 1.
2. PPL(top 40, Set A) -- already computed in Phase 3 at k=40.
3. Delta_prune = (PPL_top40 - PPL_all50) / PPL_all50 * 100%.

K1 assessment:
- If Delta_prune < -1%: removing bottom-20% IMPROVES PPL by >1%. K1 PASSES.
- If -1% <= Delta_prune <= 0%: removing bottom-20% marginally improves PPL. K1 FAILS.
- If Delta_prune > 0%: removing bottom-20% HURTS PPL. Null hypothesis confirmed.

### Phase 7: Ranking Stability Analysis (~1 min)

Compute pairwise Kendall tau between:
1. pi_domain (from pilot50 benchmark)
2. pi_loo (from LOO, Phase 2)
3. pi_mmlu (from individual MMLU if available at
   `/workspace/llm/results/individual_expert_held_out/`)
4. pi_ranked_a vs pi_ranked_b (calibration set stability from Phase 3)

K2 assessment: tau(pi_domain, pi_loo) >= 0.6.

### Phase 8: Scalability Assessment (~1 min)

Compare greedy vs ranking accumulation curves for k=1..10:

```
max_discrepancy = max_k |PPL_greedy(k) - PPL_ranked(k)| / PPL_ranked(k) * 100%
```

K3 assessment: If max_discrepancy < 0.5%, ranking is sufficient. If greedy
requires >O(N log N) evaluations to outperform ranking, K3 FAILS.

Operationally: if the first 5 greedy selections match the top-5 ranked experts,
declare K3 PASS (ranking = greedy under orthogonality).

## Kill Criteria Assessment

- **K1:** Removing bottom-20% experts (by Q_domain) does not improve composed
  model PPL by >1%.
  - Compute: Delta_prune = (PPL_top40 - PPL_all50) / PPL_all50 * 100%.
  - PASS if Delta_prune < -1% (pruning helps).
  - KILL if Delta_prune >= -1% (pruning does not help enough).
  - **If KILLED:** This is actually a POSITIVE result for SOLE -- it means
    "more experts is always better" and there is no quality ceiling from
    composition. Report as such.

- **K2:** Expert quality ranking is stable across evaluation datasets
  (Kendall tau >= 0.6).
  - Compute: tau(pi_domain, pi_loo) or tau(pi_domain, pi_mmlu).
  - PASS if tau >= 0.6.
  - KILL if tau < 0.6 (rankings are metric-dependent).

- **K3:** Optimal subset selection requires >O(N log N) evaluations.
  - Compute: greedy vs ranking match for first 10 steps.
  - PASS if ranking matches greedy (ranking is sufficient at O(N log N)).
  - KILL if greedy materially outperforms ranking (subset selection is expensive).
  - Note: "Materially" means >0.5% PPL difference at any k.

## Output

Save results to: `/workspace/llm/results/expert_pruning_composition/results.json`

Required fields in JSON:
```json
{
  "config": {
    "base_model": "Qwen/Qwen2.5-7B",
    "n_experts": 50,
    "quantization": "nf4_4bit",
    "composition_method": "naive_addition",
    "calib_texts_per_set": 30,
    "max_seq_len": 512,
    "mmlu_subjects": 30,
    "mmlu_shots": 5,
    "seed": 42,
    "smoke_test": false
  },
  "reference": {
    "ppl_all50_set_a": 0.0,
    "ppl_all50_set_b": 0.0,
    "ppl_base_set_a": 0.0,
    "ppl_base_set_b": 0.0,
    "mmlu_all50": 0.0,
    "mmlu_base": 0.0
  },
  "quality_rankings": {
    "pi_domain": ["best_adapter", "...", "worst_adapter"],
    "pi_loo": ["best_adapter", "...", "worst_adapter"],
    "domain_scores": {"adapter_name": 0.0},
    "loo_scores": {"adapter_name": 0.0}
  },
  "accumulation_curve_ppl": {
    "k_values": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "ppl_set_a": [0.0],
    "ppl_set_b": [0.0],
    "delta_from_base_pct": [0.0]
  },
  "accumulation_curve_mmlu": {
    "k_values": [5, 10, 20, 30, 40, 50],
    "mmlu_accuracy": [0.0],
    "delta_from_base_pp": [0.0]
  },
  "greedy_forward_selection": {
    "k_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "greedy_ppl_set_a": [0.0],
    "greedy_order": ["adapter_name_1", "adapter_name_2"],
    "rank_match_count": 0,
    "max_discrepancy_pct": 0.0
  },
  "bottom_k_removal": {
    "n_removed": 10,
    "removed_experts": ["adapter_1", "adapter_2"],
    "ppl_all50": 0.0,
    "ppl_top40": 0.0,
    "delta_prune_pct": 0.0,
    "mmlu_all50": 0.0,
    "mmlu_top40": 0.0,
    "mmlu_delta_prune_pp": 0.0
  },
  "ranking_stability": {
    "tau_domain_loo": 0.0,
    "tau_domain_loo_p": 0.0,
    "tau_domain_mmlu": null,
    "tau_domain_mmlu_p": null,
    "tau_set_a_set_b": 0.0,
    "tau_set_a_set_b_p": 0.0
  },
  "kill_criteria": {
    "K1_delta_prune_pct": 0.0,
    "K1_threshold_pct": -1.0,
    "K1_pass": false,
    "K1_interpretation": "pruning helps / pruning unnecessary (more is better)",
    "K2_tau_best": 0.0,
    "K2_threshold": 0.6,
    "K2_pass": false,
    "K3_max_discrepancy_pct": 0.0,
    "K3_threshold_pct": 0.5,
    "K3_pass": false,
    "verdict": "PENDING"
  },
  "timing": {
    "total_elapsed_s": 0.0,
    "phase0_setup_s": 0.0,
    "phase1_reference_s": 0.0,
    "phase2_loo_s": 0.0,
    "phase3_accum_ppl_s": 0.0,
    "phase4_accum_mmlu_s": 0.0,
    "phase5_greedy_s": 0.0,
    "phase6_removal_s": 0.0,
    "phase7_stability_s": 0.0,
    "phase8_scalability_s": 0.0
  }
}
```

## Constraints

- **Max runtime:** 90 minutes (budget: ~$0.24 at $0.16/hr A5000).
  Expected: ~60-70 min (Phase 4 MMLU is the bottleneck).
- **Expected GPU memory:** ~10GB (4-bit Qwen2.5-7B ~5GB + adapter ~0.5GB +
  activations ~4GB)
- **Must support SMOKE_TEST=1:** When set:
  - N_EXPERTS = 6 (first 6 alphabetically)
  - Accumulation k_values = [1, 2, 3, 4, 5, 6]
  - MMLU: 3 subjects only, 0-shot
  - Greedy: k=1..3 only
  - Calibration: 6 texts per set, 128 max tokens
  - Should complete in < 3 minutes
- **No additional model downloads.** Base model and adapters already on disk.
- **MMLU data:** May need to download via `datasets` library. Cache at
  `/workspace/hf_cache`.
- **Memory discipline:** One model in GPU at a time. For PPL accumulation curve,
  reload base model + composed adapter for each k. Use `gc.collect()` and
  `torch.cuda.empty_cache()` between iterations.
- **Numerical stability:** Float32 loss accumulation for PPL. Log-prob for MMLU.
  Process calibration texts one at a time (no batching with padding).

## Key Implementation Notes

1. **Adapter composition is ADDITIVE SUM, not averaged.** Each adapter's full
   B_i A_i is added to base weights. Do NOT divide by N or k. This is SOLE's
   production composition method.

2. **Individual quality scores come from pilot50_benchmark.json.** The
   `improvement_pct` field per domain IS the Q_domain_i score. Sort descending
   for pi_domain ranking.

3. **LOO computation reuses the leave_one_out experiment's approach.** See
   `macro/leave_one_out_expert_ranking/SPEC.md` for the subtraction method.
   If LOO results already exist, skip recomputation.

4. **MMLU evaluation must use 5-shot.** Use the standard prompt format:
   ```
   The following are multiple choice questions (with answers) about {subject}.

   {5 few-shot examples with answers}

   {test question}
   Answer:
   ```
   Compare log P(A), log P(B), log P(C), log P(D). Pick argmax.

5. **The accumulation curve requires multiple compositions at different k.**
   The most efficient approach: pre-load all 50 adapter tensors on CPU (in
   memory, ~3GB total for 50 * 60MB). For each k, sum the top-k deltas,
   save to tmpdir, load as PEFT adapter, eval, unload. This avoids re-reading
   safetensors from disk 14 times.

6. **Greedy forward selection can be accelerated** by pre-computing individual
   expert PPLs (which are needed for pi_domain ranking anyway). Then at each
   greedy step, only evaluate the top-M candidates (M=10) rather than all
   remaining experts. Under orthogonality, the greedy choice should be the
   individually-best remaining expert.

7. **LoRA weight key mapping:** PEFT stores keys like
   `base_model.model.model.layers.{L}.{module}.lora_A.weight`. When manually
   summing deltas, the composed delta for a target parameter is
   `delta = lora_B @ lora_A` (shape matches target). Sum across adapters.

8. **Handle the lora_scaling/alpha factor.** PEFT LoRA applies a scaling factor
   `alpha/r` when merging. The `adapter_config.json` specifies `lora_alpha` and
   `r`. When manually composing, either:
   - Apply scaling: `delta_i = (alpha/r) * B_i @ A_i`
   - Or load via PEFT which applies scaling automatically
   The safest approach: load each adapter via PEFT, merge_and_unload to get the
   scaled delta, then extract it. But this is slow. Alternative: read alpha and
   r from config, apply scaling manually.

9. **Phase ordering matters for GPU memory.** The script should process all
   PPL-based phases first (Phase 1-3, 5-6) in a single base-model session,
   then reload for MMLU (Phase 4). This minimizes model reloads.

   Suggested optimization:
   - Load base model once.
   - Phase 1: Merge all 50, eval reference PPL.
   - Phase 2: LOO via subtraction (if needed).
   - After Phase 2, we have W_composed in GPU. Need to go back to clean base
     for accumulation. Save W_composed CPU copy, reload base model.
   - Phase 3: For each k, compose on CPU, load adapter, eval PPL, unload.
   - Phase 5: Greedy forward selection (similar compose+eval loop).
   - Phase 4: MMLU (slower, separate model load per k).

10. **If budget is tight, skip Phase 5 (greedy).** Greedy is the most expensive
    phase and serves K3 only. If Phases 1-4 and 6-7 already give clear K1/K2
    results, greedy can be deferred. The script should support a
    `SKIP_GREEDY=1` environment variable.

## Phase Priority (if runtime constrained)

If approaching the 90-minute limit, prioritize:

1. **Must complete:** Phase 1 (reference), Phase 3 (PPL accumulation curve),
   Phase 6 (bottom-K removal), Phase 7 (ranking stability).
   These answer K1 and K2.

2. **Should complete:** Phase 2 (LOO ranking).
   Provides pi_loo for correlation analysis.

3. **Nice to have:** Phase 4 (MMLU accumulation), Phase 5 (greedy selection).
   Phase 4 answers whether PPL and MMLU agree on pruning benefit.
   Phase 5 answers K3 (scalability).

The script should checkpoint results after each phase, so partial results are
available even if it times out.

# Experiment Spec: clone_compete_evolution

## Objective

Test the core SOLE Evolve mechanism: clone an expert, fine-tune the clone with
corrections, run a shadow-scoring tournament, determine if the clone wins.
This is THE critical experiment for Phase 3 (Evolve) -- if clone-and-compete
does not work, the "living" part of the Living Composable Model is dead.

Three questions:
1. Do corrected clones reliably beat originals? (K1: win rate > 70%)
2. Does the tournament converge quickly? (K2: < 50K queries)
3. Does competition avoid domain regression? (K3: regression < 2%)

## Important Design Context

**Scoring signal**: Answer-only PPL was KILLED at macro for CROSS-domain
comparison (r=-0.63). However, clone-and-compete only compares two adapters
on the SAME domain. This experiment implicitly tests whether within-domain
PPL comparison is valid. If all K criteria pass but per-domain analysis shows
PPL inversions (lower PPL = worse actual quality), flag as a confound.

**Correction strategy**: Use the expert's own high-loss training examples as
corrections. This simulates the automated correction pipeline (teacher finds
errors, provides ground truth) without requiring actual API calls. The ground
truth IS the training data itself -- the expert just needs to learn it better.

**Existing script**: There is a script at `macro/clone_compete_evolution/run_clone_compete.py`.
Review it against this spec. The script is structurally sound but has several
issues to fix (see Implementation Notes below).

## Model & Data

- **Base model:** Qwen2.5-7B at `/workspace/hf_cache` (4-bit NF4 quantization)
- **Adapters:** 50 pilot adapters at `/workspace/llm/adapters/`
  - Each is a directory with `adapter_model.safetensors` + `adapter_config.json`
  - All rank-16, all-modules (q/k/v/o/gate/up/down)
- **Training data:** `/workspace/llm/data/distillation/{domain}/train.jsonl`
  - JSONL format with `messages` field (chat format)
- **No external data downloads required**
- **Test domains:** 5 diverse domains from pilot 50:
  - `python` (code), `bash` (code), `math` (reasoning),
    `medical` (knowledge), `sql` (structured)

## Procedure

### Phase 0: Setup (~30s)

1. Set random seed = 42
2. Load base model (4-bit NF4, `device_map="auto"`, bfloat16 compute)
3. Load tokenizer (Qwen2.5-7B, set pad_token = eos_token)
4. Fix lm_head dtype if needed (cast to bfloat16)
5. Verify all 5 adapter directories exist; log and skip missing ones
6. Create results directory at `/workspace/llm/results/clone_compete_evolution/`

### Phase 1: Generate Corrections (~10 min total, ~2 min/domain)

For each domain:

1. Load training data from `/workspace/llm/data/distillation/{domain}/train.jsonl`
2. Reserve last 200 lines as evaluation set (general queries)
3. From remaining lines, sample up to 200 candidate correction examples
4. Load expert adapter via `PeftModel.from_pretrained(base_model, adapter_dir)`
5. Score each candidate: compute per-example loss with `model(**inputs, labels=input_ids)`
6. Select top-50 highest-loss examples as corrections (expert's weakest points)
7. Save to `/workspace/llm/results/clone_compete_evolution/corrections/{domain}_corrections.jsonl`
8. **Unload adapter** (delete PeftModel, cache empty, gc collect)

**Data split strategy**: Use the training data itself. Last 200 lines = general
eval set (to measure regression). First N-200 lines = correction candidate pool.
This avoids needing separate eval files that may not exist.

**Memory discipline**: Load and unload each adapter individually. Never keep
more than one PeftModel in memory at a time.

### Phase 2: Clone-and-Compete Tournaments (~8 min/domain, ~40 min total)

For each domain with corrections:

#### Step 2a: Clone (~5s)
1. Copy adapter directory: `{adapter_dir}` -> `{adapter_dir}_clone_exp`

#### Step 2b: Fine-tune Clone (~60-90s)
1. Load clone as trainable: `PeftModel.from_pretrained(..., is_trainable=True)`
2. Load correction examples as HuggingFace dataset
3. Fine-tune with SFTTrainer:
   - max_steps = 50
   - batch_size = 2, gradient_accumulation = 2 (effective batch = 4)
   - learning_rate = 1e-4
   - warmup_steps = 5
   - bf16 = True (or fp16 if bf16 not supported)
   - optim = "adamw_8bit"
   - max_length = 512
4. Save fine-tuned clone back to clone directory
5. Record: fine-tuning time, final training loss
6. **Unload**: delete model, trainer, gc collect, empty cache

#### Step 2c: Shadow Scoring (~6 min)

Score both original and clone on TWO query sets:

**General queries** (for K3 regression): 200 examples from the domain's
eval split. These test whether the clone maintains general domain competence.

**Correction queries** (for K1 win determination): The 50 correction examples.
These test whether the clone learned from the corrections.

Scoring function (answer-conditioned PPL approximation):
1. Load adapter via `PeftModel.from_pretrained(base_model, adapter_path)`
2. For each text: tokenize, forward pass, `outputs.loss` gives per-token avg loss
3. Record loss per example
4. **Unload** adapter after scoring

Score in order: original-general, clone-general, original-corrections, clone-corrections.
(Load/unload adapter 4 times total per domain.)

#### Step 2d: Determine Winner

1. Clone wins on corrections if: mean(clone_correction_losses) < mean(orig_correction_losses)
2. Regression % = (exp(mean_clone_general) - exp(mean_orig_general)) / exp(mean_orig_general) * 100
3. Convergence check: evaluate at subset sizes [50, 100, 200] correction queries.
   The smallest subset where the winner matches the full-set winner is the convergence point.

#### Step 2e: Cleanup
1. Remove clone directory (`shutil.rmtree`)
2. Save per-domain checkpoint to `checkpoint.json` after each domain

### Phase 3: Statistical Analysis

Compute across all valid (non-errored) domain results:

1. **Clone win rate**: n_clone_wins / n_domains
2. **Mean regression %**: average of per-domain regression_pct
3. **Max regression %**: worst per-domain regression_pct
4. **Max convergence queries**: largest convergence point across domains
5. **Per-domain PPL improvement on corrections**: (orig_corr_ppl - clone_corr_ppl) / orig_corr_ppl * 100
6. **Effect size**: For each domain, Cohen's d between original and clone loss distributions

### Phase 4: Kill Criteria Evaluation

- **K1: clone win rate > 70%** -> need >= 4/5 domains where clone wins on corrections
  - Value: win_rate * 100
  - Threshold: 70%
  - PASS if value > threshold

- **K2: convergence < 50K queries** -> measure largest convergence point
  - Value: max_convergence_queries across domains
  - Threshold: 50,000
  - PASS if value < threshold
  - Note: Our evaluation budget is 200 queries/domain. If winner is clear at 200,
    K2 passes trivially (200 << 50K).

- **K3: max domain regression < 2%** -> no domain's general PPL worsens by >2%
  - Value: max_regression_pct across domains
  - Threshold: 2.0%
  - PASS if value <= threshold

Overall: PASS if all three pass. FAIL if any fails.

## Output

Save results to: `/workspace/llm/results/clone_compete_evolution/clone_compete_results.json`

Required fields in JSON:
```json
{
  "timestamp": "ISO-8601",
  "config": {
    "domains": ["python", "bash", "math", "medical", "sql"],
    "ft_steps": 50,
    "ft_lr": 1e-4,
    "n_corrections": 50,
    "eval_queries": 200,
    "convergence_checkpoints": [50, 100, 200],
    "base_model": "Qwen/Qwen2.5-7B",
    "seed": 42,
    "smoke_test": false
  },
  "per_domain": {
    "python": {
      "domain": "python",
      "elapsed_s": 0.0,
      "ft_time_s": 0.0,
      "ft_loss": 0.0,
      "n_corrections": 50,
      "n_general_queries": 200,
      "general_queries": {
        "original_ppl": 0.0,
        "clone_ppl": 0.0,
        "regression_pct": 0.0
      },
      "correction_queries": {
        "original_ppl": 0.0,
        "clone_ppl": 0.0,
        "improvement_pct": 0.0,
        "clone_wins": true
      },
      "convergence_queries": 50,
      "effect_size_cohens_d": 0.0,
      "clone_wins": true,
      "regression_exceeds_2pct": false
    }
  },
  "aggregate": {
    "n_domains": 5,
    "n_valid": 5,
    "clone_win_rate": 0.0,
    "mean_regression_pct": 0.0,
    "max_regression_pct": 0.0,
    "mean_improvement_pct": 0.0,
    "max_convergence_queries": 0,
    "mean_effect_size": 0.0
  },
  "kill_criteria": {
    "K1_clone_win_rate_gt_70pct": {
      "value": 0.0,
      "threshold": 70.0,
      "pass": true
    },
    "K2_convergence_lt_50k_queries": {
      "value": 0,
      "threshold": 50000,
      "pass": true
    },
    "K3_domain_regression_lt_2pct": {
      "value": 0.0,
      "threshold": 2.0,
      "pass": true
    }
  },
  "verdict": "PASS",
  "elapsed_s": 0.0
}
```

## Constraints

- **Max runtime:** 90 minutes (budget: ~$0.24 at $0.16/hr A5000)
  - Expected: ~75 minutes for 5 domains
  - Per-domain timeout: 30 minutes (safety net)
  - Total timeout: 2 hours (hard limit via MAX_RUNTIME env var)
- **Expected GPU memory:** ~10GB (4-bit Qwen2.5-7B ~5GB + LoRA + activations)
- **Must support SMOKE_TEST=1:** When set:
  - FT_STEPS = 5 (instead of 50)
  - N_CORRECTIONS = 5 (instead of 50)
  - EVAL_QUERIES = 10 (instead of 200)
  - CONVERGENCE_CHECKPOINTS = [5, 10]
  - PER_DOMAIN_TIMEOUT = 120s
  - MAX_RUNTIME = 300s
  - Should complete in < 5 minutes
- **No external data downloads.** All adapters and data on disk.
- **Memory discipline:** One PeftModel in GPU at a time. Explicit cleanup between
  every adapter load/unload cycle. Use gc.disable()/gc.enable() around heavy
  compute phases (nanochat pattern).
- **Checkpointing:** Save results after each domain completes. Resume from
  checkpoint if script is restarted.

## Implementation Notes (Fixes for Existing Script)

The existing `run_clone_compete.py` is a good starting point. Key issues to
verify and fix:

1. **Data split**: The script falls back to `train.jsonl` tail for eval, which
   is fine. But it uses the SAME data for correction generation AND general eval.
   Fix: partition the data explicitly. Last 200 lines = general eval. Lines before
   that = correction candidate pool. Ensure no overlap between correction set
   and general eval set.

2. **Answer-conditioned PPL**: The script uses full-sequence loss (`outputs.loss`
   on full input), not answer-conditioned PPL (loss on answer tokens only).
   For within-domain comparison, full-sequence loss should work (both original
   and clone see the same prompts, so prompt-portion loss is a constant offset).
   However, for maximum fidelity to the proven micro result, implement
   answer-conditioned scoring: mask prompt tokens in the loss computation.
   This requires finding the assistant response start position in the tokenized
   chat template.

3. **Effect size**: Add Cohen's d computation per domain. This quantifies how
   large the correction improvement is, independent of sample size. Important
   for assessing practical significance.

4. **Improvement percentage**: Add correction improvement metric
   `(orig_corr_ppl - clone_corr_ppl) / orig_corr_ppl * 100`. The existing
   script only reports raw PPL values; the improvement percentage is more
   interpretable.

5. **Statistical test**: Add per-domain paired t-test or Wilcoxon signed-rank
   test on per-example losses (original vs clone on correction set). Report
   p-value alongside the win/lose determination. This helps distinguish a
   genuine improvement from noise.

6. **SIGALRM + CUDA**: The existing script uses `signal.SIGALRM` for per-domain
   timeout. This CANNOT interrupt CUDA operations (known issue from HumanEval
   eval -- see HYPOTHESES.yml pilot50_held_out_eval evidence). Use a threading-
   based timeout instead, or simply rely on the MAX_RUNTIME check between
   domains (simpler, sufficient).

7. **Clone directory naming**: Use `{domain}_clone_exp` (current behavior is fine).
   Ensure cleanup happens even if the tournament errors out (use try/finally).

## Relationship to Other Experiments

- **Depends on**: exp_distillation_pilot_50 (adapters exist), exp_hash_ring_remove_expert (removal is safe)
- **Implicitly tests**: exp_relative_ppl_within_domain (within-domain PPL ranking)
- **Blocks**: exp_evolution_convergence (10-cycle quality monotonicity)
- **Informs**: exp_task_accuracy_evolve_signal (if PPL works within-domain, task accuracy signal may be unnecessary)
- **Informs**: exp_automated_correction_pipeline (validates that corrections improve experts)

## Expected Outcomes

**Optimistic (PASS)**: Clone wins 4-5/5 domains (80-100%). Regression < 1%.
Convergence at 50-100 queries. Effect sizes > 0.5 (medium to large). This
would validate the entire Evolve phase mechanism.

**Neutral (PARTIAL)**: Clone wins 3-4/5 domains (60-80%). Some regression
on specific domains (1-3%). Convergence varies by domain. Corrections help
on some domains but not others. Would need to investigate per-domain patterns.

**Kill (FAIL)**: Clone wins < 3/5 domains. Or regression > 2% on any domain.
Or PPL shows inversions (lower PPL on corrections but higher on general).
Would indicate the correction + fine-tuning mechanism does not reliably
improve experts.

## Adversarial Concerns

1. **Training data as ground truth is circular**: We generate corrections from
   the expert's high-loss training examples, then fine-tune the clone on those
   same examples. The clone should trivially learn them. Counterargument: the
   test is whether the clone *also* maintains general performance (K3). The
   corrections are a subset of training data the expert already saw but failed
   to learn well -- the clone just gets more gradient on those examples.

2. **5 domains is low power**: A 4/5 win rate has p=0.19 under null (not
   statistically significant). Mitigation: report per-domain effect sizes and
   p-values. If 4/5 win with large effect sizes (d > 1.0), practical significance
   is established even without statistical significance at the 5-domain level.

3. **PPL as quality proxy at macro**: Within-domain PPL comparison is untested
   at macro scale. If the experiment passes but PPL is later shown to be
   unreliable within-domain, the results would need re-evaluation with task
   accuracy scoring.

4. **No actual traffic**: This experiment simulates the tournament with static
   data, not real serving traffic. The hash ring mechanics (adding clone,
   routing traffic, shadow scoring in real-time) are not tested. This is
   a mechanism test, not an integration test.

# Experiment Spec: task_accuracy_evolve_signal

## Objective

Measure whether a 10-question held-out MMLU subset produces stable adapter
rankings (Kendall tau >= 0.7 vs 100-question gold standard). This validates
task accuracy as the quality signal for the Evolve phase clone-and-compete
tournament, replacing the killed PPL-based signals.

Secondary objectives:
- Measure wall-clock evaluation time per adapter per domain
- Compare accuracy ranking vs PPL ranking vs gold-standard ranking
- Identify the minimum number of questions needed for tau >= 0.7

## Model & Data

- **Base model**: Qwen2.5-7B at `/workspace/models/Qwen2.5-7B`
- **Adapters**: All available at `/workspace/llm/adapters/` (up to 20).
  Use adapter discovery pattern from CODING_GUIDELINES.md.
- **Eval data**: MMLU test split via `datasets` library (`cais/mmlu`)
- **Subjects**: 10 diverse subjects spanning knowledge domains:
  ```
  abstract_algebra, anatomy, college_computer_science, college_physics,
  econometrics, high_school_biology, high_school_us_history,
  machine_learning, professional_medicine, world_religions
  ```
  Selection criteria: (1) >= 100 test questions, (2) span science/humanities/
  professional domains, (3) include subjects likely matched by adapters
  (medicine, CS) and unmatched subjects (history, religion).

## Procedure

### Phase 1: Data Preparation (no GPU)

1. Load 10 MMLU subjects via `datasets` library
2. For each subject with >= 100 test questions:
   a. Use first 100 questions as gold standard set $G_s$
   b. Generate 5 random 10-question subsets $T_s^{(1)}, \ldots, T_s^{(5)}$
      drawn WITHOUT replacement from $G_s$ (seed=42)
3. Also generate subsets of size 25 and 50 for the "minimum questions"
   sweep: $T_s^{(j,k)}$ for $k \in \{10, 25, 50\}$, $j \in \{1,\ldots,5\}$
4. Pre-format all prompts in MMLU format:
   ```
   {question}
   A. {choice_a}
   B. {choice_b}
   C. {choice_c}
   D. {choice_d}
   Answer:
   ```

### Phase 2: Evaluation via vLLM (GPU)

Use vLLM offline batch inference with LoRA support. This is mandatory --
do NOT use sequential HF generate().

1. Initialize vLLM engine:
   ```python
   from vllm import LLM, SamplingParams
   llm = LLM(
       model="/workspace/models/Qwen2.5-7B",
       enable_lora=True,
       max_lora_rank=16,
       max_num_seqs=64,
       gpu_memory_utilization=0.85,
       trust_remote_code=True,
   )
   ```

2. **Base model evaluation**: Batch all 1000 prompts (10 subjects x 100
   questions) through vLLM with `max_tokens=1` and `temperature=0`.
   Score by extracting logprobs for tokens A/B/C/D from the output.

   IMPORTANT: Use `SamplingParams(max_tokens=1, temperature=0, logprobs=5)`
   to get logprobs for the top tokens. Then check which of A/B/C/D has
   highest logprob. This is a single forward pass per prompt, no generation
   loop needed.

3. **Per-adapter evaluation**: For each adapter, repeat the batch evaluation
   with a LoRA request:
   ```python
   from vllm.lora.request import LoRARequest
   lora_req = LoRARequest(adapter_name, adapter_id, adapter_path)
   outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
   ```
   Record wall-clock time for each adapter's full evaluation.

4. **Answer extraction**: For each output, extract the predicted answer:
   ```python
   # From logprobs, find which of A/B/C/D has highest probability
   # Handle both "A" and " A" token variants
   ```

### Phase 3: Ranking Analysis (no GPU)

1. For each subject $s$ and each adapter $i$:
   - Compute gold accuracy: $a_i(G_s)$ (accuracy on 100 questions)
   - Compute subset accuracy for each draw $j$ and size $k$:
     $a_i(T_s^{(j,k)})$

2. For each subject $s$:
   - Compute gold ranking of adapters by accuracy
   - For each draw $j$ and size $k$: compute subset ranking
   - Compute Kendall tau: $\tau_{s,j,k}$ between gold and subset rankings

3. Aggregate:
   - Per-subject mean tau: $\bar{\tau}_{s,k} = \text{mean}_j(\tau_{s,j,k})$
   - Overall mean tau: $\bar{\tau}_k = \text{mean}_s(\bar{\tau}_{s,k})$
   - Report for each $k \in \{10, 25, 50\}$

4. Compare accuracy ranking vs answer-only PPL ranking:
   - For each adapter on each subject, compute answer-only PPL
     (NLL of correct answer token given the prompt)
   - Compute Kendall tau between PPL ranking and gold accuracy ranking
   - This directly tests K3

5. Ranking stability: for each subject, count how many of the 5 draws
   change which adapter is ranked #1. Report fraction of subjects where
   the top adapter is stable across all 5 draws.

## Kill Criteria Assessment

- **K1**: $\bar{\tau}_{10} \geq 0.7$ (10-question tau vs 100-question gold)
  - PASS if mean tau >= 0.7
  - NUANCED if mean tau < 0.7 but $\bar{\tau}_{25} \geq 0.7$
    (10 questions insufficient, but 25 suffice)
  - KILL if $\bar{\tau}_{50} < 0.7$ (even 50 questions don't work)

- **K2**: Per-domain evaluation time < 60s/adapter
  - Measure: total wall-clock for one adapter across all 10 subjects,
    divided by 10
  - PASS if < 60s, KILL if > 60s

- **K3**: Accuracy ranking agrees with gold-standard AND/OR disagrees
  with PPL ranking
  - Compute tau(accuracy_10q, gold_100q) and tau(ppl, gold_100q)
  - PASS if accuracy tau > ppl tau (accuracy is better signal than PPL)
  - KILL if both accuracy AND ppl tau < 0.3 (no signal works)

## Output

Save results to: `results/task_accuracy_evolve_signal/results.json`

Required fields in JSON:
```json
{
  "experiment": "task_accuracy_evolve_signal",
  "timestamp": "ISO-8601",
  "base_model": "/workspace/models/Qwen2.5-7B",
  "config": {
    "subjects": ["list of 10 subjects"],
    "n_adapters": 20,
    "n_draws": 5,
    "subset_sizes": [10, 25, 50],
    "gold_size": 100,
    "seed": 42
  },
  "base_accuracy": {
    "per_subject": {"subject_name": {"correct": N, "total": M, "accuracy": 0.XX}},
    "overall": 0.XX
  },
  "adapters": {
    "adapter_name": {
      "per_subject": {
        "subject_name": {
          "gold_accuracy": 0.XX,
          "gold_ppl": X.XX,
          "subset_accuracies": {
            "10": [0.XX, 0.XX, 0.XX, 0.XX, 0.XX],
            "25": [0.XX, 0.XX, 0.XX, 0.XX, 0.XX],
            "50": [0.XX, 0.XX, 0.XX, 0.XX, 0.XX]
          }
        }
      },
      "eval_time_s": X.X
    }
  },
  "rankings": {
    "per_subject": {
      "subject_name": {
        "gold_ranking": ["adapter1", "adapter2", ...],
        "tau_by_size": {"10": {"mean": 0.XX, "std": 0.XX, "per_draw": [...]},
                        "25": {...}, "50": {...}},
        "tau_ppl_vs_gold": 0.XX,
        "top1_stable_across_draws": {"10": true/false, "25": true/false, "50": true/false}
      }
    },
    "overall": {
      "mean_tau_by_size": {"10": 0.XX, "25": 0.XX, "50": 0.XX},
      "mean_tau_ppl_vs_gold": 0.XX,
      "top1_stability_by_size": {"10": X/10, "25": X/10, "50": X/10}
    }
  },
  "timing": {
    "per_adapter_mean_s": X.X,
    "per_domain_mean_s": X.X,
    "total_s": X.X
  },
  "kill_criteria": {
    "K1_mean_tau_10": 0.XX,
    "K1_mean_tau_25": 0.XX,
    "K1_mean_tau_50": 0.XX,
    "K1_threshold": 0.7,
    "K1_verdict": "PASS|NUANCED|KILL",
    "K2_per_domain_time_s": X.X,
    "K2_threshold_s": 60,
    "K2_verdict": "PASS|KILL",
    "K3_acc_tau_vs_gold": 0.XX,
    "K3_ppl_tau_vs_gold": 0.XX,
    "K3_verdict": "PASS|KILL"
  }
}
```

## Constraints

- **Max runtime**: 30 min (should complete in < 10 min with vLLM)
- **Expected GPU memory**: ~16GB (Qwen2.5-7B in auto dtype + LoRA)
- **Must support SMOKE_TEST=1**: When set, use 2 subjects, 3 adapters,
  2 draws, gold_size=20, subset_sizes=[5, 10]
- **Must NOT use sequential HF generate()** -- vLLM batch inference only
- **Adapter discovery**: Use filesystem scan pattern from CODING_GUIDELINES.md
  (fall through to directory listing if benchmark JSON is empty)

## Implementation Notes

### vLLM Log-Probability Scoring

The standard approach for MMLU with vLLM:

```python
# Generate with max_tokens=1 to get logprobs for the next token
params = SamplingParams(max_tokens=1, temperature=0, logprobs=10)
outputs = llm.generate(prompts, params, lora_request=lora_req)

for output in outputs:
    logprobs = output.outputs[0].logprobs[0]  # dict: token_id -> LogProb
    # Find logprob for each answer token
    best = max("ABCD", key=lambda c: get_logprob(logprobs, tokenizer, c))
```

### Answer-Only PPL Computation

For K3 comparison, compute answer-only PPL. For each question with
correct answer $c$:

```python
# Prompt: "Question... Answer:"
# Full: "Question... Answer: C"
# PPL = exp(-log P(C | prompt))
```

This can be done via vLLM's logprobs output: the logprob of the generated
token IS the answer-conditioned log probability. No separate forward
pass needed -- just extract the logprob for the correct answer token
from the same generation output.

### Handling Ties

With 10-question subsets, many adapters will tie (e.g., all get 7/10).
Use midrank method for tie-breaking in Kendall tau computation
(scipy.stats.kendalltau handles this automatically).

### Subject Selection Fallback

If any of the 10 target subjects has < 100 questions in the test split,
skip it and log a warning. The experiment is valid with >= 5 subjects.

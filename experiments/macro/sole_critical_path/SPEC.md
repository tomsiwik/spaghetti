# Experiment Spec: SOLE Critical Path (Batch of 3)

## Overview

Three critical-path macro experiments run as a single sequential GPU session via
`macro/sole_critical_path/run_all_critical.py`. The script already exists and
must NOT be modified.

**Script:** `macro/sole_critical_path/run_all_critical.py`
**Results:** `results/sole_critical_path/`

## Model and Data

- **Base model:** Qwen2.5-7B (`/workspace/models/Qwen2.5-7B`)
- **Adapters:** 5 rank-16 all-modules LoRA adapters in `/workspace/llm/adapters/`:
  bash, math, medical, python, sql
- **Training data:** `/workspace/llm/data/distillation/{name}/train.jsonl`
  (last CALIB_SAMPLES entries used as calibration text)
- **Calibration:** 30 texts per set, truncated to 512 tokens
- **Quantization:** fp16 for eval, NF4 4-bit for union LoRA training (Exp 3)
- **Smoke test:** `SMOKE_TEST=1` reduces calibration to 5 texts, 256 tokens,
  10 training steps

## Experiments

### Experiment 1: Poisoned Adapter Detection (Leave-One-Out PPL)

**HYPOTHESES.yml node:** `exp_poisoned_adapter_detection`
**Also serves:** `exp_leave_one_out_expert_ranking` (same mechanism)
**Script function:** `run_poisoned_detection()`
**Estimated runtime:** ~15 min

**Procedure:**
1. Load base model (fp16, device_map="auto")
2. Build calibration set: 30 texts from the tail of all adapters' training data
3. Compute base PPL on calibration set
4. Compose all N=5 adapters via sequential PeftModel merge, compute all-N PPL
5. For each adapter i:
   a. Load fresh base model
   b. Compose N-1 adapters (all except i)
   c. Compute PPL_{-i}
   d. Compute Delta_i = (PPL_{-i} - PPL_all) / PPL_all * 100%
6. Rank adapters by PPL_{-i} (lowest PPL = most harmful when present)
7. Save results

**Output file:** `results/sole_critical_path/poisoned_detection.json`
**Output schema:**
```json
{
  "experiment": "poisoned_adapter_detection",
  "base_ppl": float,
  "all_composed_ppl": float,
  "leave_one_out": {
    "<adapter_name>": {"ppl": float, "delta_pct": float}
  },
  "ranking": ["<adapter_name>", ...],
  "most_harmful": "<adapter_name>",
  "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

**Kill Criteria Assessment:**
- **K1 (detection):** `most_harmful == "sql"` -- PASS if sql identified as most harmful
- **K2 (pruning):** `(all_composed_ppl - leave_one_out["sql"]["ppl"]) / all_composed_ppl > 0.50`
  -- PASS if removing sql reduces PPL by >50%
- **K3 (runtime):** total experiment time < 30 min

### Experiment 2: PPL-Probe Weighted Composition

**HYPOTHESES.yml node:** `exp_ppl_probe_macro_composition` (related: `exp_ppl_probe_macro_composition_v2`)
**Script function:** `run_ppl_probe()`
**Estimated runtime:** ~20 min

**Procedure:**
1. Load base model (fp16)
2. Build probe set: 10 texts from all adapters' training data
3. For each adapter i:
   a. Load adapter i alone via PeftModel
   b. Compute per-adapter PPL on probe set
   c. Unload
4. Compute PPL-probe weights: softmax(1/ppl_i, tau=1.0)
5. Compose all adapters with EQUAL weights (1/N), compute PPL
6. Compose all adapters with PPL-PROBE weights, compute PPL
7. Compare base PPL, top-1 PPL, equal-weight PPL, PPL-probe PPL
8. Save results

**Output file:** `results/sole_critical_path/ppl_probe_composition.json`
**Output schema:**
```json
{
  "experiment": "ppl_probe_macro_composition",
  "base_ppl": float,
  "equal_weight_ppl": float,
  "ppl_probe_ppl": float,
  "top1_ppl": float,
  "top1_adapter": "<name>",
  "weights": {"<name>": float},
  "adapter_ppls": {"<name>": float},
  "improvement_pct": float,
  "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

**Note:** This experiment is supplementary validation. The primary PPL-probe
result is already in from `exp_ppl_probe_macro_composition_v2` (SUPPORTED:
+0.36pp over base at t=0.5). This run confirms the direction using PPL metrics
instead of MMLU accuracy.

### Experiment 3: SOLE vs Monolithic LoRA

**HYPOTHESES.yml node:** `exp_sole_vs_monolithic_v2`
**Script function:** `run_sole_vs_monolithic()`
**Estimated runtime:** ~30 min

**Procedure:**
1. Load tokenizer
2. Build union dataset:
   a. For each adapter, load its train.jsonl
   b. Apply chat template to all messages
   c. Concatenate and shuffle (seed=42)
3. Train union LoRA:
   a. Load base model with NF4 quantization
   b. Configure LoRA (r=16, alpha=16, same target modules as pilot)
   c. Train via SFTTrainer: 300 steps, batch 1, grad_accum 4, lr=1e-4, cosine
   d. Save union adapter
4. Evaluate union adapter:
   a. Load fresh base model (fp16)
   b. Load union adapter via PeftModel
   c. Compute union PPL on calibration texts (5 per domain = 25 total)
5. Evaluate SOLE composition:
   a. Load fresh base model (fp16)
   b. Sequentially merge all 5 domain adapters (PeftModel + merge_and_unload)
   c. Compute SOLE PPL on same calibration texts
6. Compare and save results

**Output file:** `results/sole_critical_path/sole_vs_monolithic.json`
**Output schema:**
```json
{
  "experiment": "sole_vs_monolithic",
  "union_ppl": float,
  "sole_ppl": float,
  "union_train_steps": int,
  "union_train_time_s": float,
  "n_domains": int,
  "n_union_examples": int,
  "winner": "union" | "sole",
  "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

**Kill Criteria Assessment:**
- **K1 (per-domain win rate):** Not directly measured by the script (aggregate
  PPL only, no per-domain breakdown). The script reports the winner by aggregate
  PPL. To fully evaluate K1, a post-hoc per-domain PPL analysis would be needed.
  Partial assessment: if union wins aggregate AND by large margin (>10%), K1
  likely triggered.
- **K2 (aggregate PPL):** `(sole_ppl - union_ppl) / union_ppl > 0.10` -- KILL
  if union LoRA aggregate PPL is >10% better than SOLE.

**Known limitation:** The SOLE composition in this script uses UNSCALED addition
(sequential PeftModel merge at full alpha=1.0), NOT the 1/N scaling that
exp_ppl_probe_macro_composition_v2 showed is near-lossless. This means SOLE PPL
in this experiment may be WORSE than it should be. If SOLE loses, the result
should be re-evaluated with 1/N scaling before declaring killed.

## Execution

### Run command
```bash
# Full run
python run_all_critical.py

# Smoke test
SMOKE_TEST=1 python run_all_critical.py

# Individual experiments
python run_all_critical.py --only poisoned
python run_all_critical.py --only ppl-probe
python run_all_critical.py --only monolithic
```

### GPU queue submission
```bash
uv run python3 tools/gpu_queue.py submit macro/sole_critical_path/run_all_critical.py
```

### Expected runtime
| Experiment | Smoke | Full |
|-----------|-------|------|
| Exp 1: Poisoned detection | ~2 min | ~15 min |
| Exp 2: PPL-probe | ~3 min | ~20 min |
| Exp 3: SOLE vs monolithic | ~5 min | ~30 min |
| **Total** | **~10 min** | **~65-90 min** |

### Expected GPU memory
- Exp 1-2: ~14 GB (fp16 Qwen2.5-7B)
- Exp 3 training: ~16 GB (NF4 4-bit + LoRA + optimizer states)
- Peak: ~16 GB (fits A5000 24GB with margin)

### Expected cost
~$0.16-0.24 (1.0-1.5 hrs at $0.16/hr A5000)

## Results Harvesting

After GPU task completes, results are in:

```
results/sole_critical_path/
  poisoned_detection.json     -- Exp 1
  ppl_probe_composition.json  -- Exp 2
  sole_vs_monolithic.json     -- Exp 3
  union_adapter/              -- Trained union LoRA (Exp 3)
  summary.json                -- Overall status + timing
```

### Kill Criteria Evaluation Rules

**Exp 1 (exp_poisoned_adapter_detection):**

| Criterion | Field | Condition | Verdict |
|-----------|-------|-----------|---------|
| K1 | `most_harmful` | == "sql" | PASS/FAIL |
| K2 | `leave_one_out.sql.delta_pct` | abs value > 50 | PASS/FAIL |
| K3 | summary.json elapsed | < 1800s (30 min) | PASS/FAIL |

If all K pass: **PROVEN**. Detection mechanism validated.
If K1 fails: **KILLED**. LOO does not correctly identify harmful adapters.
If K2 fails with K1 pass: **SUPPORTED**. Detection works but pruning insufficient.

**Exp 2 (ppl_probe_macro_composition):**

This is directional validation. No hard kill criteria in this run -- the primary
result is already in from v2 (SUPPORTED). The PPL comparison here provides
additional evidence.

| Metric | Condition | Interpretation |
|--------|-----------|----------------|
| `improvement_pct` | > 0% | PPL-probe helps (consistent with v2) |
| `ppl_probe_ppl` | < `base_ppl` | Composition beats base (consistent with v2) |
| `ppl_probe_ppl` | < `equal_weight_ppl` | Routing beats equal-weight |

**Exp 3 (exp_sole_vs_monolithic_v2):**

| Criterion | Field | Condition | Verdict |
|-----------|-------|-----------|---------|
| K2 | `sole_ppl`, `union_ppl` | (sole-union)/union > 0.10 | KILL |
| Winner | `winner` | "sole" or "union" | Informational |

If union wins by >10%: **KILLED** (with caveat about unscaled composition).
If SOLE wins or within 10%: **SUPPORTED**. Modularity is free (or nearly free).
If SOLE wins by >10%: **PROVEN**. Composition outperforms monolithic.

**Important caveat for Exp 3:** If SOLE loses, re-run with 1/N scaled composition
before declaring killed. The script uses unscaled addition which is known to be
suboptimal (PPL in trillions at N=5 without scaling). The fair comparison is
SOLE with 1/N scaling vs monolithic.

## Constraints

- Max runtime: 2 hours (A5000)
- Expected GPU memory: 16 GB peak (NF4 training phase)
- Must support `SMOKE_TEST=1` (reduces to ~10 min)
- All experiments are function-scoped (GPU memory freed between experiments)
- Sequential execution (no parallelism needed -- single GPU)
- Results must be self-contained JSON files for automated harvesting

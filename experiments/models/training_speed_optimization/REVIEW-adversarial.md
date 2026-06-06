# Peer Review: Training Speed Optimization

## NotebookLM Findings

Skipped -- this is an optimization profiling experiment, not a hypothesis-driven mechanism test. The claims are empirical measurements, not theoretical derivations requiring deep mathematical review.

## Mathematical Soundness

### FLOP Estimate (MATH.md Section 4)

MATH.md claims:
- Forward FLOPs: ~2 * 2.4B * 256 = ~1.2T
- Backward FLOPs: ~2.4T
- Total: ~3.6T per step
- Theoretical step time: 3.6T / 27 TFLOPS = 133ms

**Issue 1: The 27 TFLOPS figure is unverified.** MATH.md states "M5 Pro theoretical compute: ~27 TFLOPS (bfloat16)" without citation. Apple does not publish bf16 TFLOPS for the M5 Pro GPU. The actual observed baseline (107ms mean, 92ms min) is *faster* than the 133ms "theoretical minimum," which means either (a) the FLOP estimate is too high, (b) the 27 TFLOPS figure is too low, or (c) the model is not actually doing 3.6T FLOPs per step. This inconsistency is never addressed. The min of 92ms implies the hardware delivers at least 3.6T/0.092 = 39 TFLOPS if the FLOP count is correct, which seems high for an M5 Pro.

**Verdict:** The theoretical lower bound is wrong or the FLOP estimate is wrong. One or the other. This does not invalidate the experiment (it is profiling, not proving a bound), but the MATH.md Section 5 "Worked Example" that uses this bound as a reference point is misleading.

### Throughput Calculation (Code vs PAPER.md)

The code computes samples/sec as:
- Batch=1: `TRAIN_STEPS / total_s` (line 741-742), so 100 / 10.727 = 9.3. Correct.
- Batch=4: `4000 / total_s` (line 670), so 4000 / 71.581 = 55.9. This is `batch_size * n_steps / total_s`.
- Batch=8: `8000 / total_s` (line 680), so 8000 / 138.017 = 58.0.

These calculations are internally consistent. The 7.52x figure (69.9 / 9.3) is correctly computed.

### Speedup Field in results.json Is Confusing

The `speedups` field in results.json computes `baseline_mean / variant_mean` for per-step time:
- O5a_batch4: 107.27 / 715.81 = 0.15 (i.e., each step is 6.7x SLOWER)
- O6b_all_optimized_b8: 107.27 / 1144.41 = 0.094 (10.7x slower per step)

This is technically correct (it measures ms/step speedup, not throughput speedup) but having a field called "speedups" where the best result is 0.094 is confusing. The `best_speedup_ms_per_step` is 1.012 (O3, a noise-level improvement). The actual useful metric is `best_throughput_speedup` at 7.516. The JSON structure buries the real finding.

## Experimental Design

### Proper Isolation -- Partially Achieved

Each variant calls `zero_lora_params(model)` and creates a fresh optimizer. This is good -- it means LoRA weights and optimizer state are reset between variants. However:

1. **Variants run sequentially in the same process.** MLX metal state, memory fragmentation, and cache effects from earlier runs could influence later runs. A truly isolated benchmark would run each variant in a separate process. The observed std values are small enough (1-12ms) that this is likely not a major concern, but it is a methodological weakness.

2. **No repeated trials.** Each configuration runs once (100 steps after 5 warmup). Statistical confidence comes from the 100 step-level measurements, not from independent runs. For the batch=1 variants where differences are <2%, this means the experiment cannot distinguish signal from noise -- which it correctly acknowledges.

3. **Warmup is only 5 steps.** For the compiled variant (O4), the first several steps include compilation overhead. With only 5 warmup steps, some compilation cost may leak into the timed steps, explaining the high variance (std=41.25ms). The p50 of 86.8ms vs mean of 116.2ms strongly suggests outlier recompilation events are counted. More warmup (e.g., 20 steps) or outlier trimming would give a fairer picture of steady-state compiled performance.

### Convergence Definition Is Weak

The `check_convergence` function (line 593-599) checks whether the average of the last 20 losses is less than 98% of the average of the first 20 losses. This is an extremely low bar:
- Batch=1 final loss: 2.7969 (all four batch=1 variants converge to the same value)
- Batch=4 final loss: 1.7578
- Batch=8 final loss: 0.8438

**The different batch sizes converge to wildly different final losses.** Batch=8 reaches 0.8438 vs batch=1 at 2.7969, a 3.3x difference. This is expected: batch=8 sees 8x more data per gradient step and 800 total samples vs 100 for batch=1 (both run 100 steps). But the paper's K2 claim "convergence preserved" glosses over the fact that these are not comparable training regimes. With the same learning rate, larger batches get an implicit effective-LR-per-sample reduction, yet their loss is *lower* -- this is because they see more diverse data per step, not because they converge "better."

PAPER.md Limitations #5 correctly notes "Learning rate not adjusted for batch size." This is the right caveat, but K2 PASS should be qualified: convergence is preserved in the sense that loss decreases monotonically, not in the sense that the same optimization problem is being solved.

### The Central Paradox Is Adequately Flagged

PAPER.md Section 5 ("Training Time Implications") clearly states:
- Batch=1, 200 steps: 21.4s wall-clock
- Batch=4, 200 steps: 143.2s wall-clock (SLOWER)

The paper correctly explains that throughput (samples/sec) and training speed (wall-clock for N gradient updates) are different metrics, and that batch>1 only helps when processing more data. This is the most important finding in the experiment and it is not buried. The Practical Recommendations section correctly advises keeping batch=1 for the current 200-step regime.

**However**, the headline claim "7.52x throughput improvement" is misleading without immediate qualification. Anyone reading the experiment title "Training Speed Optimization" and seeing "7.52x" will assume training is 7.5x faster. It is not. Training is 5-7x slower in wall-clock for the same number of gradient updates. The 7.52x is a GPU utilization metric. The paper does explain this, but the results.json `verdict: SUPPORTED` with `best_throughput_speedup: 7.516` creates a misleading summary for automated consumption.

### Confound: O2 (Pre-tokenized) vs O0 (Baseline) Use Different Data Paths

The baseline (O0) uses `tokenizer.encode` on raw text, while O2 and O3+ use pre-tokenized data. But the timing starts *after* data preparation (line 237-241 for baseline: tokenization happens before `t0`). Wait -- looking more carefully:

```python
# Baseline (lines 231-241)
tokens = tokenizer.encode(texts[idx])  # BEFORE t0
tokens = tokens[:MAX_SEQ_LENGTH + 1]
x = mx.array(tokens[:-1])[None, :]
y = mx.array(tokens[1:])[None, :]

t0 = time.perf_counter()  # Timing starts HERE
```

Actually, tokenization is outside the timed region. The `mx.array` creation is also outside. So the baseline timing only captures the GPU compute + eval. This is correct -- the timing is fair. The pre-tokenization "optimization" is measured correctly as negligible because tokenization was never in the timed loop to begin with. The -1.1% for O2 is noise, which makes sense.

Wait -- but MATH.md Section 2b claims pre-tokenization saves "~0.1-1ms per step" by avoiding `tokenizer.encode`. If tokenization is outside the timed region, this saving would not be captured. Let me re-read... Yes, in the baseline, `tokenizer.encode` is called before `t0`. So the experiment correctly measures that pre-tokenization has no effect on *step time*, but MATH.md incorrectly predicted it would help by counting tokenization as part of the step. The MATH.md prediction is wrong, but the code is correct. Minor inconsistency.

### Padding Confound for Batched Variants

The batched variants (O5a, O5b, O6, O6b) pad all sequences to MAX_SEQ_LENGTH=256. The batch=1 variants use variable-length sequences (up to 256). If many training sequences are shorter than 256, the batched variants do unnecessary computation on padding tokens. This means the "throughput" comparison slightly understates the true throughput advantage of batching (since batch=1 does less work per sample on short sequences) or overstates it (since batch>1 does wasted work on pads).

The loss function correctly masks padded positions (`y != -100`), so convergence is not affected. But the timing comparison has a small confound.

## Novelty Assessment

This is not a novel research contribution. It is standard engineering profiling. The finding that "batch>1 improves GPU utilization" is well-known. The specific finding that "for 200-step LoRA fine-tuning on a 2.4B model, batch=1 is actually faster in wall-clock" is a useful practical observation for this project but is not publishable.

The experiment correctly positions itself as profiling rather than research (it is in `micro/models/` not `macro/`). No novelty issues.

## Macro-Scale Risks (advisory)

1. **The practical recommendation (keep batch=1 for 200 steps) may change** if adapter training is extended beyond 200 steps or if data volume per domain increases. Any change to the training regime should re-evaluate this.

2. **Memory was not tracked.** PAPER.md Limitation #4 acknowledges this. At batch=8 with 30 layers and seq_len=256, peak memory could be significant. Before using batch=8 in production alongside other memory consumers (multiple loaded adapters, KV cache for serving), memory profiling is needed.

3. **The mx.compile benefit (+17% at batch>=4) depends on fixed shapes.** Variable-length batches would destroy this benefit. Any production batching strategy must pad to fixed lengths.

## Verdict

**PROCEED**

The experiment is well-executed profiling with one key insight (batch>1 improves throughput but not wall-clock training speed for fixed step counts) that is clearly explained and correctly caveated. The results are internally consistent and the code implements what it claims.

Issues that prevent full confidence but do not warrant REVISE:

1. The FLOP estimate in MATH.md is inconsistent with observed timings (theoretical minimum 133ms, observed minimum 92ms). This is a documentation error, not a results error.

2. The headline "7.52x speedup" is technically samples/sec throughput, not wall-clock training speed. The paper explains this but the results.json summary is misleading for automated consumption.

3. Single-process sequential execution with no repeated trials is a minor methodological weakness, but the effect sizes (6-7x for batching, <2% for Python tricks) are large enough that this does not threaten the conclusions.

4. The convergence check (2% threshold) is too weak to claim "convergence preserved" across different batch sizes that solve different optimization problems, but the paper's Limitation #5 (LR not adjusted) adequately flags this.

None of these issues change the actionable conclusions: use batch=1 for current 200-step training, use batch>=4 + mx.compile for larger datasets.

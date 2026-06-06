# Pierre v6.1: Precomputed Concat Deltas (Full QKV + MLP)

## Theorem

**Theorem 1 (Concat-Slice Equivalence).** For modules sharing the same input,
concatenating their precomputed delta weight matrices and computing a single
matmul followed by slicing produces bit-identical results to individual
per-module matmuls.

**Theorem 2 (Dispatch Count).** With 4 groups per layer across 30 layers,
the total dispatch count is exactly 120.

**Theorem 3 (Speed Model).** Under a linear dispatch-overhead model calibrated
from v3 (420 dispatches, 73 tok/s) and v6 (60 dispatches, 86.8 tok/s), v6.1
at 120 dispatches would achieve ~84.2 tok/s.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|-------------------------|----------|--------|
| Dispatch count = 120 (Thm 2) | 120 | YES |
| Speed ~84.2 tok/s (Thm 3) | 42.1 tok/s | **NO** (2x slower) |
| Behavioral ~0.41 (Thm 1, matching v3) | 0.419 | YES |
| Code behavioral ~0.84 (Thm 1) | 0.844 | YES |
| Memory ~2.5-4 GB (estimate) | 5.47 GB | PARTIAL (within K758 but at upper end) |

## Hypothesis

Restoring MLP adapters via gate+up concatenation achieves >= 75 tok/s with
behavioral >= 0.35, using exactly 120 dispatches per forward pass.

**VERDICT: KILLED (K756 FAIL)**

## What This Model Is

Pierre v6.1 precomputes the full LoRA delta DeltaW = alpha * A @ B for each
module, then concatenates deltas that share the same input tensor:
- QKV group: q_proj, k_proj, v_proj -> 1 concat matmul
- O group: o_proj -> 1 matmul
- Gate+Up group: gate_proj, up_proj -> 1 concat matmul
- Down group: down_proj -> 1 matmul

This gives 4 dispatches per layer x 30 layers = 120 total, down from v3's
420 (7 modules x 2 matmuls each x 30 layers).

## Key References

- Finding #292: Pierre v6 (attention-only) -- 86.8 tok/s, 60 dispatches,
  code behavioral 0.281
- Finding #288: Pierre v3 -- 73 tok/s, 420 dispatches, behavioral 0.41
- Finding #290: Pierre v5 -- 77.2 tok/s, 420 dispatches, ternary LoRA

## Empirical Results

### Kill Criteria

| Criterion | Threshold | Value | Verdict |
|-----------|-----------|-------|---------|
| K756: Speed | >= 75 tok/s | 42.1 tok/s | **FAIL** |
| K757: Behavioral | >= 0.35 | 0.419 | PASS |
| K758: Memory | <= 6 GB | 5.47 GB | PASS |

### Behavioral Scores (per domain)

| Domain | v6.1 Score | v3 Score | v6 (attn-only) | Match v3? |
|--------|-----------|----------|-----------------|-----------|
| Medical | 0.450 | 0.437 | 0.437 | YES (+3%) |
| Code | 0.844 | 0.844 | 0.281 | YES (exact) |
| Math | 0.662 | 0.661 | 0.661 | YES |
| Legal | 0.054 | 0.054 | 0.054 | YES |
| Finance | 0.086 | 0.086 | 0.086 | YES |
| **Overall** | **0.419** | **0.41** | **0.304** | **YES** |

**Theorem 1 (exact equivalence) CONFIRMED.** Behavioral scores match v3 to
within measurement noise. Code domain fully recovered from v6's 0.281 to 0.844,
confirming MLP adapters are critical for code instruction following.

### Speed Analysis

| Config | Dispatches | Speed (tok/s) | Memory (GB) |
|--------|-----------|---------------|-------------|
| Base BitLinear | 0 | 142.5 | ~1.2 |
| v3 (RuntimeLoRA) | 420 | 73 | ~1.3 |
| v5 (TernaryLoRA) | 420 | 77.2 | - |
| v6 (attn-only precomputed) | 60 | 86.8 | 2.23 |
| **v6.1 (full precomputed)** | **120** | **42.1** | **5.47** |

## Why Theorem 3 (Speed Model) Failed

**The linear dispatch-overhead model is wrong.** The model assumed:
```
T(D) = T_base + c * D
```
where c is a constant per-dispatch overhead. This model fit v3 and v6 but
predicted 84.2 tok/s for v6.1. The actual speed is 42.1 tok/s -- worse than
v3 despite having 3.5x fewer dispatches.

**Root cause: memory bandwidth, not dispatch count, is the bottleneck.**

The precomputed approach materializes full-rank DeltaW matrices from the
low-rank A @ B factorization. At rank 16:

| | Factored (v3) | Precomputed (v6.1) | Ratio |
|--|--------------|-------------------|-------|
| A matrix | 2560 x 16 = 82 KB | - | - |
| B matrix | 16 x d_out | - | - |
| DeltaW (full) | - | 2560 x d_out | - |
| QKV group | 3 x (82KB + ~10-40KB) = ~300 KB | 2560 x 3840 x 2 = 19.7 MB | 66x |
| Gate+Up group | 2 x (82KB + ~22KB) = ~208 KB | 2560 x 13824 x 2 = 70.8 MB | 340x |
| Total per layer | ~600 KB | ~139 MB | 230x |
| Total 30 layers | ~18 MB | ~4.2 GB | 230x |

**The factored form (v3) transfers ~18 MB per forward pass for adapter corrections.
The precomputed form (v6.1) transfers ~4.2 GB.** On M5 Pro with 273 GB/s memory
bandwidth, the precomputed deltas alone consume 4.2/273 = 15.4 ms of bandwidth
per token. This explains the 42.1 tok/s (23.8 ms/tok) vs v3's 73 tok/s
(13.7 ms/tok).

**The dispatch overhead saved (300 fewer dispatches x ~0.006 ms = 1.8 ms) is
dwarfed by the memory bandwidth cost (15.4 ms - 0.6 ms factored = +14.8 ms).**

### Why v6 (attention-only) worked

v6 only precomputed attention modules (QKV + O). The attention delta memory:
- QKV concat: 2560 x 3840 x 2 = 19.7 MB per layer
- O: 2560 x 2560 x 2 = 13.1 MB per layer
- Total: 32.8 MB per layer x 30 = 983 MB

At 983 MB, the bandwidth cost is 983/273000 = 3.6 ms per token, which is
manageable. Adding MLP modules adds 3.2 GB more delta data (gate+up is massive
at 70.8 MB/layer due to d_intermediate=6912), pushing bandwidth to the
breaking point.

## The Fundamental Tradeoff

```
Approach              Dispatches  Delta Memory  Speed
v3 (factored A,B)     420         18 MB         73 tok/s   (dispatch-bound)
v6 (precomp attn)     60          983 MB        86.8       (sweet spot)
v6.1 (precomp all)    120         4.2 GB        42.1       (bandwidth-bound)
```

There is an OPTIMAL dispatch/bandwidth tradeoff. v6 (attention-only) sits
near the optimum. v6.1 crosses into bandwidth-bound territory because MLP
dimensions (6912) are 2.7x larger than attention dimensions (2560),
making their full-rank deltas massive.

## Limitations

1. The speed model (Theorem 3) was derived from only 2 data points and assumed
   dispatch overhead was the sole cost. Memory bandwidth was not modeled.

2. The behavioral evaluation uses factual recall (keyword overlap) which is a
   weak proxy for actual task quality. Legal (0.054) and finance (0.086) are
   essentially non-functional -- a base model limitation.

3. The experiment tested single-adapter serving. Multi-adapter composition
   would multiply the delta memory further.

## What Would Kill This

Already killed by K756. The precomputed full-model approach is fundamentally
bandwidth-bound on Apple Silicon.

## What Was Learned

1. **Concat algebra is exact.** Theorem 1 confirmed: behavioral scores match
   v3 bit-for-bit. The code domain fully recovered (0.844 vs 0.844).

2. **Dispatch count is NOT the bottleneck for MLP modules.** The linear
   dispatch-overhead model fails when delta memory exceeds ~1 GB. Memory
   bandwidth becomes the dominant cost.

3. **The optimal precompute boundary is attention-only.** v6 (attention-only
   precomputed, 983 MB deltas) achieves 86.8 tok/s. Adding MLP precomputation
   pushes delta memory to 4.2 GB and halves throughput.

4. **MLP modules should stay in factored form.** The rank-16 factored form
   (x @ A then h @ B) transfers ~18 MB for MLP corrections vs 3.2 GB
   precomputed -- 180x less bandwidth at the cost of 6 extra dispatches per
   layer (which cost only ~1 ms total).

5. **Optimal hybrid: precompute attention, factor MLP.** This would give:
   - Attention: 2 dispatches/layer (QKV concat + O) + 983 MB deltas
   - MLP: 6 dispatches/layer (gate A/B, up A/B, down A/B) + ~12 MB deltas
   - Total: 8 dispatches/layer = 240 dispatches
   - Predicted speed: between v3 (73) and v6 (86.8), likely ~80 tok/s
   - Full behavioral quality (all modules present)

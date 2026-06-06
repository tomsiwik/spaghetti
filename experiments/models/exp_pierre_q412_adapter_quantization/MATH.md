# MATH.md — Q4.12 PoLAR Adapter Quantization

## Hypothesis

PoLAR adapter weights (rank=6 LoRA matrices A, B) can be quantized to Q4.12
fixed-point (int16, 4 integer bits + 12 fractional bits) without losing more
than 2pp task accuracy, while halving on-disk size and enabling NEON int16
widening MAC kernels for ~3× faster composition computation.

## Theoretical grounding

### Q4.12 in talos-vs-macbook
talos-vs-macbook reported Q4.12 quantization error ~0.0001 per weight on
microGPT (16-dim model). Several generated names match fp32 byte-for-byte.
On Apple Silicon, `vmlal_s16` (int16 widening MAC into int32 accumulator)
runs at full pipeline throughput.

### Why this fits PoLAR
PoLAR enforces ||A^T A − I||_F ≈ 0 and ||B B^T − I||_F ≈ 0 (joint Stiefel,
F#442). Stiefel-constrained matrices have entries in [-1, 1]: orthonormal rows
have unit row norm, so per-entry magnitude ≤1 by Cauchy-Schwarz.

For Q4.12: range is [-8, 8) with resolution 1/4096 ≈ 0.000244. PoLAR adapters'
natural value range is well within this, so quantization is loss-bound by
resolution, not range. Per-weight quant error: ~0.0001 (matches talos).

### Why it's mostly a size win, not a speed win
Pierre's bottleneck is base model forward (~6.5B FLOPs/token, dominates).
Adapter delta computation is ~0.0002× base forward — already negligible.
Quantization speedup ON ADAPTERS doesn't move the needle on inference latency.

The real wins:
1. **On-disk size**: 5MB → 2.5MB per adapter. At N=25 skills × 7 layers in scope,
   that's 90MB → 45MB. Matters for deployment / cold-start.
2. **Memory bandwidth**: int16 half the bytes loaded → DRAM-bound paths benefit.
   Adapter delta during composition becomes more L2-cache-friendly.
3. **Setup for full-int pipeline**: with base 4-bit and adapters int16, the
   entire forward path is integer arithmetic (with fp32 RMSNorm/softmax).
   Future Metal kernel fusion could exploit this; today it's just hygiene.

## Predictions

1. **K1**: Q4.12 task accuracy within 2pp of fp32 single-adapter on each of
   GSM8K/HumanEval/MedQA. Per-weight quant error is 1000× smaller than typical
   adapter weight magnitudes; should be invisible in task metrics.

2. **K2**: Q4.12 composition delta computation ≥3× faster (single-thread NEON).
   Math: int16 widening MAC produces int32 accumulator — same number of MACs
   per output but half the load bandwidth, full pipeline throughput on M5 Pro
   NEON. Conservative 3× because RMSNorm/softmax stay fp32 (need conversion
   back).

3. **K3**: 5MB → 2.5MB per adapter on disk. Deterministic 50% from int16 vs
   fp32. Aggregate: N adapters × 2.5MB.

4. **K4**: Per-weight error ≤0.0001. Talos showed this for microGPT; should
   transfer to PoLAR's Stiefel-constrained matrices because both stay in
   [-1, 1] range.

## Implementation plan

### Quantization function
```python
def quantize_q412(w_fp32: np.ndarray) -> np.ndarray:
    # Q4.12: 4 integer bits, 12 fractional bits → range [-8, 8), resolution 2^-12
    SCALE = 4096  # 2^12
    w_int16 = np.round(np.clip(w_fp32 * SCALE, -32768, 32767)).astype(np.int16)
    return w_int16

def dequantize_q412(w_int16: np.ndarray) -> np.ndarray:
    return w_int16.astype(np.float32) / 4096
```

### NEON kernel for composition delta
```c
// per layer: composed_delta = Σ w_i × (A_i @ B_i)
//   A_i shape (d_in=2560, r=6); B_i shape (r=6, d_out)
// Output: composed (d_in, d_out)

// Q4.12 version: int16 weights, int32 accumulators
void compose_q412(
    const int16_t* A_concat,  // shape (N_adapters, d_in, r) packed
    const int16_t* B_concat,  // shape (N_adapters, r, d_out) packed
    const float* gate_weights, // shape (N_adapters,) — gate softmax
    int32_t* delta_int32,     // shape (d_in, d_out) — composed
    float* delta_fp32,        // shape (d_in, d_out) — final, after dequant
    int N, int d_in, int r, int d_out
) {
    // For each (i, j) output: 
    //   delta[i,j] = Σ_n gate[n] × Σ_k A[n,i,k] × B[n,k,j]
    // NEON: int16x8 widens to int32x4 via vmlal_s16
    // ... [full kernel implementation with unrolling]
}
```

### Files (planned)
- `quantize.py` — load fp32 PoLAR adapter, write int16 Q4.12 + scale metadata
- `compose_q412.c` — NEON C kernel for Q4.12 composition delta
- `compose_fp32.c` — equivalent fp32 reference for benchmarking
- `bench.c` — back-to-back compare: load adapters, run N composition deltas,
  measure throughput + per-weight error
- `validate_accuracy.py` — load Q4.12 adapter, run GSM8K/HumanEval/MedQA via
  mlx_lm, compare to fp32 reference
- `run_experiment.py` — orchestration

## Risks

1. **Stiefel drift after quantization**: Q4.12 quantization noise may push
   ||A^T A - I|| above 0.01 threshold. Mitigation: re-retract after quantization
   (single SVD per layer, ~100ms total).

2. **RMSNorm/softmax fp32 boundaries**: each "exit" from int math costs
   conversion. If we hit too many boundaries per layer, the speedup vanishes.
   Adapters apply BEFORE the layer's RMSNorm, so this is safe.

3. **Int16 overflow on large d_out**: Σ over 2048 elements of int16×int16 =
   max 2048 × 2^30 = 2^41 → fits int64 but exceeds int32. Use int64 accumulator
   for d_out > 1024, or chunk the accumulation.

## Pre-registered KCs

K2134: Q4.12 task accuracy within 2pp of fp32 on GSM8K/HumanEval/MedQA
K2135: Q4.12 composition delta ≥3× faster than fp32 (single-thread NEON)
K2136: Adapter on-disk size halves (5MB → 2.5MB)
K2137: Per-weight |ΔW_q412 − ΔW_fp32| ≤ 0.0001

## References
- ../talos-vs-macbook/bench_c_q412.c — Q4.12 NEON kernel reference
- ../talos-vs-macbook/README.md "how the C version works"
- F#442 — PoLAR Stiefel constraint guarantees [-1,1] entry range
- F#444 — PoLAR scale invariance (3× stability vs LoRA across scale 3-24)

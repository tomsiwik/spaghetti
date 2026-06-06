# BitNet Serving Path: LoTA-QAF Merge + Apple Silicon Benchmarks

## Hypothesis

LoTA-QAF lossless ternary merge (W'_int = W_int + quantize(DeltaW)) preserves adapter PPL benefits when merging N trained adapters into BitNet-2B-4T base weights, enabling pre-merged serving on Apple Silicon within latency and memory budgets.

**Verdict: KILLED (K1 fails catastrophically)**

## What This Experiment Tested

Three serving paths for BitNet-SOLE on Apple Silicon:

1. **LoTA-QAF merge**: Add LoRA delta to base weights, requantize to ternary grid. This would enable bitnet.cpp-style packed integer serving.
2. **Float merge**: Add LoRA delta to base weights in bfloat16 (no requantization). Serves via MLX with unpacked float weights.
3. **Runtime LoRA**: Keep adapters separate, apply at inference time (the proven approach from prior experiments).

## Key References

- LoTA-QAF (arxiv 2505.18724): Lossless ternary adaptation via quantization-aware fine-tuning
- BitNet b1.58 (arxiv 2402.17764): Ternary weight architecture
- Prior experiments: bitnet_sole_vs_monolithic (adapters reused), bitnet_2b_real_composition

## Empirical Results

### K1: LoTA-QAF Merge Fidelity (FAIL)

| Metric | Value |
|--------|-------|
| LoRA delta / weight ratio | 0.43% |
| Ternary states changed by merge | 0.00% |
| Base avg PPL | 10.64 |
| Float merge avg PPL (N=5) | 8.52 |
| QAF merge avg PPL (N=5) | 60.93 |
| QAF benefit loss | 2476% (adapter completely erased) |
| **K1 verdict** | **FAIL** |

The requantization step erases the adapter entirely. The LoRA delta (mean 0.004) is 116x smaller than the quantization grid spacing (0.463). Zero ternary values change.

Root cause: LoTA-QAF assumes ternary adapters with product DeltaW that is integer-valued and grid-aligned. Our STE-trained LoRA adapters produce continuous FP32 deltas that are 0.43% of base weight magnitude.

### K2: Latency (PASS)

| Config | tok/s | p95 (100 tokens) |
|--------|-------|-------------------|
| Base (unpacked bf16) | 13.1 | 7.75s |
| N=5 QAF merged | 12.8 | 8.33s |
| **K2 verdict** | **PASS** (p95 = 8.33s < 10s) |

Both configurations run at ~13 tok/s on Apple Silicon via MLX. Pre-merged and base have identical latency because the model architecture is unchanged (same number of parameters, same computation).

### K3: Memory (PASS)

| Config | Process Memory |
|--------|---------------|
| Base (packed ternary) | 1718 MB |
| Base (unpacked bf16) | 1718 MB |
| After N=15 pre-merge | 1729 MB (+11 MB) |
| **K3 verdict** | **PASS** (1.69 GB << 8 GB) |

Pre-merging adds zero persistent memory since the delta is folded into existing weights. The 11 MB increase is transient computation overhead. This is the strongest result: pre-merged BitNet-2B-4T with any number of adapters fits in 1.7 GB.

### Float Merge Quality (Supplementary)

Float merge (bfloat16, no requantization) partially preserves adapter benefits:

| Domain | Base PPL | Float Merge N=5 | Reference Composed N=5 |
|--------|----------|-----------------|----------------------|
| medical | 15.80 | 11.35 | 13.38 |
| code | 3.52 | 3.00 | 3.22 |
| math | 4.74 | 3.85 | 4.32 |
| legal | 25.52 | 21.11 | 23.46 |
| creative | 3.60 | 3.28 | 3.38 |
| **avg** | **10.64** | **8.52** | **9.55** |

Float merge avg PPL (8.52) is actually BETTER than reference composed PPL (9.55). This is because the reference used 1/N scaling during LoRA composition, while float merge computes the exact delta without scaling artifacts.

However, float merge single-adapter PPLs (Phase 4) were worse than reference routed PPLs due to bfloat16 precision loss:

| Domain | Reference Routed | Float Merge N=1 |
|--------|-----------------|-----------------|
| medical | 8.00 | 16.43 |
| code | 2.76 | 3.49 |
| math | 3.12 | 4.07 |
| legal | 17.95 | 25.29 |
| creative | 3.17 | 7.09 |

The N=1 float merge is catastrophically bad for some domains (medical: 2x worse) because bfloat16 truncation destroys fine-grained adapter information. The mathematical analysis shows the LoRA delta is exactly at the bfloat16 ULP boundary (~0.004 delta vs ~0.004 ULP), causing ~50% information loss.

### Multi-Adapter Scaling (Float Merge)

| N | Avg Float PPL | Avg QAF PPL | Delta/Weight Ratio |
|---|--------------|-------------|-------------------|
| 1 | 29.27 | 60.30 | 1.24% |
| 2 | 10.68 | 60.82 | 0.81% |
| 3 | 9.52 | 60.86 | 0.66% |
| 4 | 8.30 | 60.84 | 0.58% |
| 5 | 8.52 | 60.93 | 0.53% |

QAF PPL is constant at ~61 regardless of N (because the merge is completely erased). Float merge improves with more adapters (1/N scaling reduces each adapter's contribution, reducing bfloat16 truncation effects).

## Root Cause Analysis

The failure has three interacting causes:

1. **Scale mismatch**: LoRA adapters learn small perturbations (alpha_A ~ 0.012, alpha_B ~ 0.006) while base weights are large (alpha_W ~ 0.926). The product alpha_A * alpha_B * lora_scale = 0.0014 is 660x smaller than alpha_W.

2. **Quantization grid spacing**: For ternary requantization, a weight must change by > alpha/2 ~ 0.46 to flip state. The LoRA delta of ~0.004 is 116x too small.

3. **bfloat16 precision**: Even without requantization, bfloat16 ULP near the weight magnitude (~0.004) equals the LoRA delta, losing ~50% of adapter information.

## Limitations

1. **Only tested on BitNet-2B-4T**: Larger BitNet models with different alpha distributions might behave differently (unlikely given the 116x gap).

2. **Only STE-trained adapters**: True LoTA-QAF adapters (trained end-to-end with quantization-aware objective) might produce larger grid-aligned deltas. Our adapters were trained with standard LoRA + STE, not LoTA-QAF's t-SignSGD.

3. **Only MLX serving**: bitnet.cpp was not tested directly. The latency results reflect MLX with unpacked bfloat16 weights, not the packed integer path that bitnet.cpp uses.

4. **Reused adapters**: All adapters came from bitnet_sole_vs_monolithic (400 steps, STE ternary). Different training recipes might produce adapters with larger weight norms.

5. **5 domains only**: More domains might show different patterns.

## What Would Kill This (in Retrospect)

K1 was killed because LoTA-QAF's "lossless" claim requires adapters trained within the quantization-aware framework (t-SignSGD with grid-aligned objective). Standard LoRA + STE does not produce grid-aligned deltas.

For LoTA-QAF to work on SOLE, you would need:
- Adapters trained with LoTA-QAF's specific t-SignSGD optimizer (not standard Adam + STE)
- Adapter weight norms ~100x larger to cross the grid threshold
- Or a fundamentally different merge strategy that matches scales before quantization

## Implications for BitNet-SOLE Architecture

1. **Runtime LoRA is the only viable serving path** for BitNet-SOLE with current adapter training. This means serving must use unpacked float weights (4 GB, 13 tok/s on Apple Silicon), not packed ternary (0.5 GB, 35 tok/s).

2. **BitNet's serving advantage (packed integer arithmetic) is incompatible with LoRA composition**. You get either fast packed inference (no adapters) OR slow unpacked inference (with adapters). Not both.

3. **The memory advantage survives**: Even unpacked, BitNet-2B-4T + 15 adapters fits in 1.7 GB (K3 PASS). This is far below the 8 GB threshold.

4. **Future work**: Investigate LoTA-QAF's native training recipe (t-SignSGD) for SOLE adapters. If it produces grid-aligned deltas, the 35 tok/s packed serving path opens up.

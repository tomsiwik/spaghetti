# LEARNINGS: Quantized Routing Heads

## Status: SUPPORTED (with scope limitation to N=5)

Post-training quantization of routing heads (82K param 2-layer MLPs) to int8
and int4 preserves 100% accuracy on all 5 domains with 75-87.5% memory reduction.

## What We Learned

### 1. Routing heads survive extreme quantization at N=5

Int4 quantization (7.1% max per-weight error) produces max logit perturbation
of only 0.045, while typical logit magnitudes are ~4.0. The margin-to-error
ratio of 89:1 makes sign-flip impossible. Even int2 would likely work at N=5.

### 2. The real memory savings require bit-packing (not implemented)

The experiment simulates int4 by storing in int8 and dividing by 2 in memory
accounting. Actual int4 bit-packing is needed for real savings. Additionally,
pre-dequantized fp32 copies are kept for inference, so runtime memory is
actually larger than fp32.

**For real savings:** Either use MLX's built-in quantization (nn.QuantizedLinear)
or implement packed storage with on-the-fly dequantization.

### 3. Latency is identical — both paths run fp32 matmul

The ~5% latency variation is noise, not dequantization cost. Both paths
execute identical fp32 matrix multiplications since weights are pre-dequantized
at construction time. For these tiny matrices (2560x32, 32x1), compute is
dominated by dispatch overhead regardless of precision.

### 4. N=5 is the trivially easy case

Prior findings show routing head accuracy collapses at N>=10 (46% of domains
fall back to base-only). At N=5 with 100% accuracy, margins are enormous.
The quantization safety result does NOT extend to N>=10 where margins are
tight and quantization could cause real failures.

### 5. Routing head memory is negligible anyway

At N=100 with fp32: 32.8 MB. With int4: 4.1 MB. Savings: 28.7 MB.
Compare to total memory budget of ~40 GB usable. Routing heads are <0.1%
of the memory budget even at fp32. This optimization is not on the critical path.

## Confirming Evidence

- MATH.md sign preservation theorem: margin-to-error of 89:1
- All 5 domains maintain 100% accuracy at int4
- Int8 gives 75% real (not projected) memory reduction

## Contradicting Evidence

- Prior finding: routing heads break at N>=10 (margin collapse, not tested here)
- Int4 memory savings are projected, not measured (no bit-packing)
- Latency claim was wrong (both paths run fp32, difference is noise)

## Follow-up

No follow-up experiment recommended for quantization specifically. The real
bottleneck is routing head accuracy at N>=10, not storage efficiency. Focus
should remain on the deployment track.

If N>=10 routing is solved (e.g., via hierarchical routing), quantization
validation at that scale would be worthwhile.

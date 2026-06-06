# Ternary B-Matrix: Fully Ternary LoRA Adapters

## Hypothesis

Making the LoRA B-matrix ternary (matching the already-ternary Grassmannian A-skeleton)
yields fully ternary adapters that compose with ratio < 1.5 and enable 15.8x storage
compression, with no degradation in per-domain quality.

**Verdict: SUPPORTED.** Both STE and PTQ ternary B achieve composition ratio < 1.07,
well below the 1.5 success threshold and 2.0 kill threshold.

## What This Experiment Tests

Current LoRA adapters in the BitNet-SOLE architecture use:
- **A-matrix**: Frozen, ternary-compatible, Grassmannian-initialized (orthogonal across adapters)
- **B-matrix**: Trainable, FP32

This experiment tests whether B can also be ternary, making the entire adapter {-1,0,+1}
valued. Fully ternary adapters enable:
1. **Pure addition composition** -- merging N adapters requires only integer addition
2. **15.8x storage compression** -- 2 bits per parameter vs 32 bits
3. **Simpler hardware** -- no floating-point multiply needed for adapter application

Three conditions tested:
1. **FP32 B** (baseline): Standard LoRA with continuous B
2. **STE ternary B**: B quantized to {-alpha, 0, +alpha} via Straight-Through Estimator during training
3. **PTQ ternary B**: B trained as FP32, then post-training quantized to ternary

## Key References

- BitNet b1.58 (Ma et al., 2024) -- absmean ternary quantization with STE
- LoRA (Hu et al., 2021) -- low-rank adaptation
- Grassmannian packing for adapter orthogonality (project prior work)
- Prior experiment: `ternary_base_from_scratch_mlx` -- STE training pipeline

## Empirical Results

Architecture: d_model=128, n_layers=4, n_heads=4, rank=8, 5 domains (names quintary split).

### Per-Domain PPL (Individual Adapters)

| Domain | FP32 B | STE Ternary B | PTQ Ternary B |
|--------|--------|---------------|---------------|
| a_e    | 1.511  | 1.502 (0.994x) | 1.522 (1.007x) |
| f_j    | 1.539  | 1.515 (0.985x) | 1.513 (0.983x) |
| k_o    | 1.509  | 1.502 (0.995x) | 1.519 (1.006x) |
| p_t    | 1.552  | 1.533 (0.988x) | 1.535 (0.989x) |
| u_z    | 1.567  | 1.543 (0.985x) | 1.508 (0.962x) |
| **Mean** | **1.536** | **1.519** (0.989x) | **1.519** (0.989x) |

**K2 PASS**: Both ternary B conditions are actually *better* than FP32 B on average
(0.989x, not >1.5x). STE ternary B shows a slight regularization benefit.

### Composition (5-Adapter Merge)

| Condition | Mean Single PPL | Mean Composed PPL | Composition Ratio | Mean |cos| |
|-----------|----------------|-------------------|-------------------|------------|
| FP32 B    | 1.536          | 1.608             | **1.047**         | ~0         |
| STE Ternary B | 1.519      | 1.623             | **1.068**         | ~0         |
| PTQ Ternary B | 1.519      | 1.595             | **1.050**         | ~0         |

**K1 PASS**: All conditions compose with ratio well below 2.0 (all below 1.07).
**S1 PASS**: All conditions below 1.5 success threshold.

The +0.021 increase from FP32 B (1.047) to STE ternary B (1.068) is marginal.
PTQ ternary B (1.050) is nearly identical to FP32 B.

### Storage Compression

| Condition | Per-Adapter (FP32 storage) | Per-Adapter (Ternary packed) | Compression |
|-----------|---------------------------|------------------------------|-------------|
| FP32 B    | 148.3 KB                  | N/A                          | 1x          |
| Ternary B | 148.3 KB (latent)         | 9.4 KB                       | **15.8x**   |

### Adapter Orthogonality

Mean |cos| between adapter deltas: ~0.000001 for all conditions.
The Grassmannian A-skeleton ensures near-perfect orthogonality regardless of B precision.
This confirms the MATH.md prediction: interference bound depends on A_i^T A_j, not B precision.

### Quantization Error

PTQ ternary B has mean relative quantization error of ~0.52 per B-matrix. Despite this
large per-element error, composition quality is preserved because:
1. Each adapter's quantization error is confined to its own Grassmannian subspace
2. The 1/N averaging during composition further attenuates per-adapter noise
3. The base model (ternary) already operates in a noise-tolerant regime

## Kill Criteria Assessment

| Criterion | Threshold | STE Result | PTQ Result | Verdict |
|-----------|-----------|------------|------------|---------|
| K1 (id=256): Composition ratio | < 2.0 | 1.068 | 1.050 | **PASS** |
| K2 (id=257): Max per-domain PPL ratio | < 1.5 | 0.995 | 1.007 | **PASS** |

| Success | Threshold | STE Result | PTQ Result | Verdict |
|---------|-----------|------------|------------|---------|
| S1 (id=28): Composition ratio | < 1.5 | 1.068 | 1.050 | **PASS** |

## Limitations

1. **Toy scale**: d=128, rank=8, character-level names data. Real models (d=2560+, rank=16+)
   may show different quantization sensitivity.
2. **Uniform task difficulty**: All 5 domains are character-level name generation subsets.
   Diverse tasks (code, math, reasoning) may stress B-matrix capacity more.
3. **No multi-seed validation**: Single seed per condition. The composition ratios are
   close enough (1.047 vs 1.068) that seed variance could change rankings.
4. **Base model is small**: 798K params. At larger scale, adapter B-matrices may need
   more precision to capture complex task-specific patterns.
5. **No latency measurement**: We claim "pure addition composition" but did not measure
   actual inference speedup from ternary operations.
6. **Surprisingly good**: STE ternary B slightly *improves* individual PPL vs FP32 B (0.989x).
   This could be a regularization effect that disappears at scale, or a genuine benefit of
   constraining B to a low-entropy manifold.

## What Would Kill This

- At macro scale (d=2560, rank=16, real tasks): composition ratio > 2.0 with ternary B
  while FP32 B stays < 1.5. This would indicate that B precision matters when tasks are
  complex enough to require fine-grained weight adjustments.
- LoRI 90% sparse B (not tested here) outperforming ternary B significantly, suggesting
  structured sparsity is a better compression strategy than quantization.
- Ternary B failing on high-entropy tasks (code generation, mathematical reasoning)
  where the rank-8 ternary bottleneck cannot represent the required update patterns.

## Recommendations

1. **Use PTQ ternary B as the default**: Simpler pipeline (train FP32, quantize after),
   near-identical composition quality to STE, and trivially reversible.
2. **STE ternary B for maximum quality**: Slight PPL advantage suggests training
   awareness of quantization helps the optimizer find ternary-friendly solutions.
3. **Proceed to macro validation**: Test on BitNet-2B-4T with real domain adapters.
4. **Investigate the "ternary regularization" effect**: Why does STE B produce better
   individual PPL? If reproducible at scale, this is a publishable finding.

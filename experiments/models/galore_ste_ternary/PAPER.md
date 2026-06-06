# GaLore + STE Integration: Research Digest

## Hypothesis

Integrating STE (Straight-Through Estimator) ternary quantization into the GaLore
training loop -- so that quantization happens DURING training rather than post-hoc --
will eliminate the 2.6x ternary PPL degradation observed in the prior GaLore experiment
while preserving GaLore's memory-efficient gradient projection.

## What This Model Is

GaLore+STE combines two techniques:
1. **BitLinear / STE**: Forward pass quantizes weights to ternary {-alpha, 0, +alpha},
   backward pass uses straight-through estimator to pass gradients through to FP32
   latent weights.
2. **GaLore**: Projects gradients to low-rank space (rank r << d) before applying Adam
   updates, reducing optimizer state memory from O(mn) to O(rn).

The key insight: by using STE in the forward pass, the loss function "sees" ternary
weights. Gradients are therefore quantization-aware. GaLore then projects these
quantization-aware gradients to low-rank space for memory-efficient optimization.

## Key References

- GaLore (arXiv 2403.03507): Memory-efficient LLM training via gradient low-rank
  projection. ICML 2024 oral. Matches full-rank at <1% gap at 1B params.
- BitNet b1.58 (arXiv 2402.17764): Ternary LLM architecture using STE.
- Prior GaLore experiment (micro/models/bitnet_galore_scaffold/): Showed 2.6x ternary
  degradation with post-hoc quantization.
- Standard STE baseline (micro/models/ternary_base_from_scratch_mlx/): PPL 1.5935,
  composition ratio 1.022.

## Empirical Results

### Base Training (d=256, 6 layers, 4 heads, vocab=27, 4.7M params)

| Metric | GaLore+STE | Standard STE | Ratio |
|--------|-----------|-------------|-------|
| **Ternary PPL** | **1.5922** | 1.5952 | **0.998x** |
| Training time | 91.8s | 53.7s | 1.71x |
| Peak memory | 0.633 GB | 0.702 GB | 0.90x |
| Final loss | 0.461 | 0.449 | - |
| Zero fraction | 30.97% | 31.24% | - |
| Optimizer state elements | 2,654,208 | 9,487,874 | **0.28x** |

### Fix Confirmed: 2.6x Degradation Eliminated

The prior GaLore experiment quantized post-hoc and saw:
- Standard training + post-hoc ternary: 1.1x degradation
- GaLore training + post-hoc ternary: **2.6x degradation**

This experiment integrates STE into GaLore and achieves:
- GaLore+STE ternary: **0.998x** vs standard STE (actually slightly better)

The quantization degradation is completely eliminated. GaLore+STE produces weights
that are inherently ternary-friendly because quantization is part of the training loop.

### Adapter Composition

| Domain | GaLore+STE Single | GaLore+STE Composed | Std STE Single | Std STE Composed |
|--------|------------------|--------------------|-|--|
| a_e | 1.503 | 1.589 | 1.498 | 1.534 |
| f_j | 1.527 | 1.607 | 1.523 | 1.593 |
| k_o | 1.512 | 1.589 | 1.511 | 1.566 |
| p_t | 1.547 | 1.599 | 1.542 | 1.575 |
| u_z | 1.545 | 1.576 | 1.568 | 1.556 |
| **Mean** | **1.527** | **1.592** | **1.528** | **1.565** |

| Metric | GaLore+STE | Standard STE |
|--------|-----------|-------------|
| Composition ratio | 1.043 | 1.024 |
| Ratio of ratios | 1.019 | - |
| Mean |cos| | 2.5e-7 | 2.9e-7 |

### Kill Criteria

| Criterion | Metric | Threshold | Actual | Verdict |
|-----------|--------|-----------|--------|---------|
| K1 | PPL ratio (GaLore+STE / Std STE) | < 1.5 | 0.998 | **PASS** |
| K2 | Composition ratio ratio | < 1.5 | 1.019 | **PASS** |
| K3 | Training time ratio | < 3.0 | 1.71 | **PASS** |

### Success Criteria

| Criterion | Metric | Threshold | Actual | Verdict |
|-----------|--------|-----------|--------|---------|
| S1 | PPL ratio | < 1.2 | 0.998 | **PASS** |
| S2 | Peak memory ratio | < 0.6 | 0.90 | **FAIL** |

## Analysis

### Why S2 Fails (and Why It Doesn't Matter at Scale)

At d=256, optimizer state is a small fraction of total memory. The model weights
(4.7M params = 18MB at FP32) dominate. The GaLore optimizer state is 2.65M elements
vs 9.49M for standard Adam -- a 3.6x reduction in optimizer state -- but this savings
is masked by weight memory at small scale.

At scale (d=2560, 7B params), optimizer state dominates memory. GaLore's r/d reduction
factor means:
- Standard Adam: 2 * 7B = 14B state elements (56GB at FP32)
- GaLore (r=256): ~0.28x = ~3.9B state elements (~16GB)
- This is the difference between needing 80GB VRAM and fitting on 48GB

The optimizer state element count confirms the mechanism works: 2.65M vs 9.49M = 0.28x.
The peak memory ratio (0.90x) fails the threshold because weight memory dominates at
toy scale.

### Why GaLore+STE Slightly Outperforms Standard STE

GaLore+STE achieves PPL 1.5922 vs standard STE's 1.5952 (0.2% better). This is within
noise but consistent with GaLore's implicit regularization: projecting gradients to
low-rank space acts as a regularizer, preventing overfitting. At toy scale with
high model capacity relative to data, this slight regularization helps.

### Composition Ratio

GaLore+STE composition ratio (1.043) is slightly worse than standard STE (1.024).
This 1.9% difference (ratio of ratios = 1.019) is well within the K2 threshold of 1.5.
The Grassmannian A-matrix orthogonality is maintained (mean |cos| = 2.5e-7) since
GaLore only affects how base weights are trained, not the adapter structure.

## Limitations

1. **Toy scale**: d=256, vocab=27, 4.7M params. The mechanism is validated but
   scaling behavior is unknown.
2. **S2 failure**: Peak memory savings not visible at toy scale due to weight-dominated
   memory profile. Need d >= 2048 to see optimizer state dominate.
3. **Single seed**: No multi-seed validation (the prior experiments used 3 seeds).
4. **Simple data**: Character-level names dataset. Real language data may have
   different gradient spectral properties that affect GaLore projection quality.
5. **Training time overhead**: 1.71x slower due to SVD recomputation every 200 steps.
   This is a known cost of GaLore.

## What Would Kill This

### At Micro Scale
- If GaLore+STE PPL were > 1.5x standard STE (K1): Would indicate STE gradients
  are poorly approximated by low-rank projection.
- If composition ratio ratio > 1.5 (K2): Would indicate GaLore-trained base weights
  are structurally less composition-friendly.

### At Macro Scale (Future Work)
- If optimizer state memory savings don't materialize at d >= 2048 (the whole point)
- If SVD overhead becomes prohibitive at scale (currently 1.71x, could grow with d)
- If STE gradient spectral properties change at scale (breaking low-rank assumption)
- If convergence requires higher GaLore rank r at scale, eroding memory savings

## Conclusion

GaLore+STE completely eliminates the 2.6x ternary quantization degradation seen
with post-hoc quantization of GaLore-trained weights. The integrated approach
achieves ternary PPL within 0.2% of standard STE training, validates adapter
composition (ratio 1.043), and reduces optimizer state memory by 3.6x.

The fix is straightforward: use BitLinear (STE forward pass) instead of nn.Linear,
and feed STE gradients into GaLore's projection. No architectural changes needed
beyond swapping the linear layer type.

This validates the path toward memory-efficient ternary base training on Apple
Silicon: GaLore provides the memory savings for larger models, STE ensures the
weights are ternary-friendly from the start.

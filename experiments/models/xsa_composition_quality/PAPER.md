# XSA for Adapter Composition: Research Digest

## Hypothesis

Exclusive Self-Attention (XSA) — removing self-value bias from attention outputs
via orthogonal projection — reduces inter-expert interference during LoRA adapter
composition, improving composed model quality.

**Falsifiable:** If XSA degrades single-adapter quality by more than 3% PPL on
any domain (K1), shows no composition ratio improvement (K2), or loses on 3+ of 5
domains when composed (K3), the hypothesis is killed.

## What This Experiment Tests

XSA is a zero-parameter attention modification from the parameter-golf competition
(arXiv 2603.09072). For each attention position i, it projects out the self-value
component from the attention output:

    z_i = y_i - (y_i^T v_i / ||v_i||^2) * v_i

This forces cos(z_i, v_i) = 0, eliminating the "attention similarity bias" where
attention outputs redundantly encode self-information already carried by the
residual connection. Per parameter-golf findings, XSA is applied only to the
last 2 layers (of 4 total), as full-depth application hurts.

The hypothesis was that this self-value removal would reduce the interference
channel when multiple LoRA adapters are pre-merged, because each adapter's V
perturbation at position i would no longer contribute to the output via the
self-attention weight.

## Key References

- arXiv 2603.09072 (Exclusive Self-Attention): Zero-param attention fix via
  orthogonal projection of self-value bias
- Parameter-golf (OpenAI): XSA discovered as effective in last 3-4 layers of
  11-layer models at d=512, 8 heads
- arXiv 2510.03262 (Rethinking Inter-LoRA Orthogonality): Weight-space
  orthogonality != semantic disentanglement due to nonlinearities

## Empirical Results

### Setup
- Micro transformer: d=128, H=4, L=4, FFN=512 (~1M params)
- Ternary-quantized base, 5 toy domains (arithmetic, reverse, repeat, sort, parity)
- LoRA rank=8, Grassmannian A (frozen), trainable B with ternary STE
- XSA applied to layers 2-3 (last 2 of 4)
- 3-seed validation (42, 123, 456)

### K1: Single-Adapter Quality Degradation (threshold: 3%)

| Domain | Standard PPL | XSA PPL | Degradation |
|--------|-------------|---------|-------------|
| arithmetic | 3.607 | 3.841 | +6.49% |
| reverse | 4.947 | 5.114 | +3.39% |
| repeat | 2.723 | 3.045 | **+11.85%** |
| sort | 4.321 | 4.359 | +0.87% |
| parity | 1.690 | 1.708 | +1.03% |

**K1 RESULT: FAIL (11.85% > 3% threshold).** XSA substantially degrades
single-adapter quality, especially on the repeat domain.

### K2: Composition Ratio (lower = better)

| Condition | Mean Composition Ratio |
|-----------|----------------------|
| Standard | 1.9005 |
| XSA | 1.9206 |

**K2 RESULT: FAIL.** XSA ratio (1.9206) is HIGHER (worse) than standard (1.9005).
XSA provides no composition benefit.

### K3: Per-Domain Composed PPL

| Domain | Standard Composed | XSA Composed | Winner |
|--------|------------------|--------------|--------|
| arithmetic | 6.533 | 8.957 | Standard |
| reverse | 7.663 | 7.212 | XSA |
| repeat | 6.745 | 7.311 | Standard |
| sort | 6.970 | 6.279 | XSA |
| parity | 3.422 | 3.468 | Standard |

**K3 RESULT: FAIL.** XSA wins only 2/5 domains (reverse, sort). Standard wins 3/5.

### Diagnostics

| Metric | Standard | XSA |
|--------|----------|-----|
| Mean adapter |cos| | 0.010793 | 0.011008 |
| Composition ratio improvement | -- | -1.06% |

Adapter orthogonality is essentially identical between conditions (both ~0.011),
confirming that XSA does not affect weight-space decorrelation.

## Kill Criteria Assessment

- **K1 (id=201):** FAIL -- 11.85% max degradation (threshold 3%)
- **K2 (id=202):** FAIL -- XSA ratio 1.9206 >= standard 1.9005
- **K3 (id=203):** FAIL -- XSA loses on 3/5 domains

**VERDICT: KILLED** (all three kill criteria failed)

## Analysis: Why XSA Fails at Micro Scale

1. **Capacity reduction:** At d_h=32, removing the self-value direction eliminates
   1/32 = 3.1% of the representational capacity per head. At d_h=64 or 128 (as in
   parameter-golf), this is 1.6% or 0.8% — much less impactful. The capacity cost
   dominates any interference reduction benefit at this scale.

2. **Short sequences:** With T ~ 8-24 tokens, there are few other positions to
   gather context from. The self-value component is a larger fraction of useful
   information when context is sparse.

3. **Ternary base amplification:** The ternary-quantized base already has reduced
   expressiveness. Removing an additional direction from the attention output
   compounds the capacity limitation.

4. **No interference mechanism to fix:** Adapter |cos| is already ~0.011 with
   Grassmannian A matrices — there is minimal inter-adapter interference to reduce.
   XSA was solving a problem that does not exist in this architecture.

## Limitations

- Micro scale only (d=128, 4 heads). Parameter-golf used d=512, 8 heads where XSA
  improved results. The capacity cost of removing 1 direction is proportionally
  larger at d_h=32.
- Toy character-level domains with short sequences. Real text with longer sequences
  and richer contextual dependencies might benefit differently.
- XSA was applied during both base training AND adapter training. An alternative
  is to train without XSA and apply it only at inference during composition.

## What Would Kill This (at larger scale)

Even at larger scale, XSA for composition would require:
- Single-adapter quality to not degrade (< 3% PPL)
- Measurable composition ratio improvement (> 2%)
- Benefit on majority of domains

The fundamental issue — that XSA removes capacity to solve an interference problem
that Grassmannian orthogonality already handles — suggests this is unlikely to work
regardless of scale.

## Key Learning

**Zero-parameter modifications are not free.** XSA adds zero learnable parameters
but removes representational capacity (one direction per head). In the composition
setting with Grassmannian A-matrices, inter-adapter interference is already
minimal (~0.011 cosine), so there is no interference to reduce. The capacity cost
outweighs any potential benefit.

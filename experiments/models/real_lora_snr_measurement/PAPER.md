# Real LoRA SNR Measurement: Research Digest

## Hypothesis

Real LoRA deltas from pilot-50 training may exhibit low SNR conditions where
the adaptive rank selection fallback (snap to r_95 when SNR < 10) would
provide practical benefit. If all experts have SNR >> 10 uniformly, the
fallback is correct but vacuous.

## What This Experiment Is

A purely observational (no training) analysis of the spectral profile of 5
real LoRA adapters trained on Qwen2.5-7B at rank-16 (all-modules: q, k, v, o,
gate, up, down projections across 28 layers). For each of 980 LoRA deltas
(5 experts x 28 layers x 7 modules), we compute the SVD of B@A and measure
SNR (sigma_1/sigma_r), r_95, r_99, and the rank diversity ratio rho = r_99/r_95.

## Key References

- `micro/models/adaptive_rank_snr_fallback/`: Designed the SNR < 10 fallback rule
- `micro/models/delta_rank_scaling/`: Showed r_95 ratio scales as d^(-0.15)
- Pilot-50 distillation: adapters trained with Unsloth on SlimOrca subsets

## Empirical Results

### Global Statistics (N=980 entries)

| Metric | Mean | Median | Min | Max | Std |
|--------|------|--------|-----|-----|-----|
| SNR | 25.3 | 16.7 | 2.6 | 393.1 | 30.7 |
| r_95 | 8.4 | 9 | 1 | 16 | - |
| r_99 | 12.3 | 13 | 1 | 16 | - |
| rho (r_99/r_95) | 1.61 | 1.44 | 1.07 | 6.00 | 0.55 |

### SNR Distribution

- **26.5% of entries have SNR < 10** (the fallback threshold)
- 4.6% of entries have SNR < 5 (severely low)
- The fallback is NOT vacuous -- it triggers on roughly 1 in 4 LoRA deltas

### Per-Module Breakdown

| Module | SNR Mean | SNR Median | r_95 | r_99 | rho |
|--------|----------|------------|------|------|-----|
| mlp.gate_proj | 46.9 | 33.3 | 5.3 | 9.2 | 1.92 |
| mlp.down_proj | 40.7 | 30.6 | 5.1 | 9.6 | 2.13 |
| mlp.up_proj | 27.3 | 21.5 | 7.7 | 11.6 | 1.61 |
| self_attn.v_proj | 18.3 | 14.4 | 9.3 | 13.4 | 1.46 |
| self_attn.o_proj | 16.1 | 12.1 | 9.7 | 13.7 | 1.50 |
| self_attn.q_proj | 16.0 | 11.0 | 10.8 | 14.0 | 1.33 |
| self_attn.k_proj | 12.0 | 9.2 | 11.2 | 14.4 | 1.31 |

**Key finding:** There is a clear structural split between MLP and attention modules:
- **MLP modules** (gate, down, up) have high SNR (27-47), concentrate energy in
  few directions (r_95 ~ 5-8), and have high rho (1.6-2.1). These are spectrally
  concentrated -- most of the update is in a low-dimensional subspace.
- **Attention modules** (k, q, v, o) have moderate SNR (12-18), use nearly full
  rank (r_95 ~ 9-11), and have lower rho (1.3-1.5). These spread their updates
  more evenly across all 16 rank dimensions.

The low-SNR entries (< 10) are overwhelmingly from attention modules, especially
k_proj (median SNR = 9.2, right at the threshold).

### Per-Expert Summary

| Expert | SNR Mean | SNR Median | rho Mean | % SNR < 10 |
|--------|----------|------------|----------|------------|
| bash | 29.8 | 19.4 | 1.67 | - |
| math | 21.8 | 14.2 | 1.55 | - |
| medical | 28.2 | 19.9 | 1.68 | - |
| python | 17.9 | 9.3 | 1.44 | - |
| sql | 28.9 | 21.4 | 1.71 | - |

Python has distinctly lower SNR (median 9.3) and lower rho (1.44), suggesting
code-domain LoRA deltas are more spectrally diffuse than natural language domains.

### Kill Criteria Assessment

**K1: All experts have SNR >= 10 (fallback is vacuous)**
- **SURVIVES.** 26.5% of entries have SNR < 10, and 4.6% have SNR < 5.
- The fallback is NOT vacuous. It would activate on a meaningful fraction of
  real LoRA deltas, particularly in attention modules.

**K2: r_99/r_95 ratio varies less than 1.5x across experts**
- **KILLED.** The expert-level rho range ratio is 1.19x (sql 1.71 / python 1.44).
- Across experts, the mean rho values are remarkably similar (1.44 to 1.71),
  differing by only 19%.
- The diversity is within-expert (across modules), not between-experts.

### Interpretation

The adaptive rank fallback is **mechanistically valid** (K1 survives -- low-SNR
conditions do exist in practice) but the **per-expert** diversity needed to
justify different rank allocations per expert does not exist (K2 killed -- all
experts have similar spectral profiles).

The real diversity is **per-module-type**, not per-expert:
- MLP modules could use rank 6-10 and capture 95% of variance
- Attention modules need rank 11-14 for the same coverage

This suggests that the correct granularity for adaptive rank is **per-module-type**
(a fixed policy: lower rank for MLP, higher for attention), not **per-expert**
adaptive selection.

## Implications for SOLE

1. **The SNR < 10 fallback triggers in practice** on ~26.5% of entries,
   validating the mechanism design from `adaptive_rank_snr_fallback`.

2. **Per-expert adaptive rank is not useful** -- all experts have similar
   spectral profiles within each module type. A fixed per-module-type rank
   policy would capture the same benefit with zero runtime cost.

3. **MLP modules are low-rank** (r_95 ~ 5-8 of 16) -- there may be significant
   parameter savings available by using lower rank for MLP projections.

4. **Attention modules use nearly full rank** (r_95 ~ 9-11 of 16) -- attention
   LoRA deltas are more spectrally diffuse, consistent with the finding that
   attention layers are essential for quality (FFN-only was killed at macro).

5. **Python/code experts are more diffuse** (lower SNR, lower rho) than natural
   language experts, potentially reflecting the higher structural complexity of
   code syntax.

## Limitations

1. Only 5 experts measured (out of 50 trained). The full pilot set may show
   more diversity, though the per-module pattern is likely structural.

2. These adapters were trained with the same hyperparameters (rank-16, same LR,
   same steps). Different training configurations could produce different
   spectral profiles.

3. SNR is measured on the raw delta B@A without considering the base model
   weight magnitudes. The effective SNR in the context of the full weight
   matrix could differ.

4. The experiment does not measure whether adaptive rank actually improves
   downstream quality -- only whether the spectral conditions that motivate
   it exist.

## What Would Kill This

- If a larger sample (all 50 experts) shows significant per-expert rho diversity
  (range ratio > 1.5x), K2 would be reversed.
- If the per-module-type pattern (MLP concentrated, attention diffuse) breaks
  for some model architectures, the fixed-policy recommendation would not
  generalize.

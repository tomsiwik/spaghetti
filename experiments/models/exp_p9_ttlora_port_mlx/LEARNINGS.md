# LEARNINGS: exp_p9_ttlora_port_mlx

**Status**: SUPPORTED | **Finding**: #515

## Core Finding

TT-LoRA (arXiv:2504.21190) ports cleanly to MLX with 8.3x parameter compression over LoRA r=6 (5,584 vs 46,080 params per layer for q+v) and only 1.36x cached latency overhead.

## Why

Tensor train decomposition factorizes ΔW into a chain of rank-r cores. The boundary cores are small (r×s_k or s_k×r), so parameter cost scales as O(r·Σs_k) rather than O(m·n) for standard weight or O((m+n)·r) for LoRA. For Gemma 4's shapes (2560/2048/512), this yields ~2.5-3K params per projection vs 18-27K for LoRA r=6.

## Implications for Next Experiment

→ **exp_p9_ttlora_quality**: Test whether 8.3x parameter compression translates to quality retention on GSM8K. Key question: can we train a TT-LoRA adapter that matches LoRA r=6 behavioral quality with 8x fewer parameters? The cached reconstruction path is ready; training requires uncached mode (gradient flows through cores).

## Key Numbers

| Metric | Value |
|--------|-------|
| Params per layer (q+v) | 5,584 |
| Compression vs LoRA r=6 | 8.3x |
| Forward consistency (K1) | 0.0 (exact) |
| Cached latency overhead (K3) | 1.36x max |
| Full adapter (42 layers, q+v) | ~235K params |
| Equivalent LoRA r=6 | ~1.94M params |

## Known Issues

MATH.md prediction table has a minor numeric error: used s_bar=8 for q_proj instead of s_bar=7.33 (due to one interior core with factor 4 in [5,8,8,8,**4**,8,8,8]). Formula is correct; only the worked example is off. PAPER.md correctly reports measured values.

## Composition

TT-LoRA adapters compose identically to standard LoRA after reconstruction: W + Σ α_i ΔW_i. The Grassmannian orthogonality machinery applies to the reconstructed ΔW matrices without modification.

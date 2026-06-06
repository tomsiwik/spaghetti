# LEARNINGS: M2P on Real Models — Why It Failed and How SHINE Actually Works

**Status:** KILLED — implementation diverged from SHINE in every critical dimension
**Finding:** #373 (killed), #374 (killed — 4 bugs), v2 also failed (flat loss)

## The Failure

M2P loss flat at 11.93 ≈ ln(vocab_size) = pure random output. 0% accuracy on GSM8K
while SFT achieves 26% (base: 20%). The M2P generates degenerate text ("strapstrap...").

## Root Cause: Our M2P ≠ SHINE

Investigation of SHINE source code (GitHub: Yewei-Liu/SHINE, configs/Qwen3-0.6B.yaml):

| | **SHINE (actual)** | **Our M2P** | **Gap** |
|---|---|---|---|
| d_M2P | = d_model (1024) | 128 | 8x bottleneck |
| Output scaling | sqrt(0.001) = 0.032 | None | Loss = ln(vocab) = noise |
| Encoder | Frozen LLM itself | Separate 2-layer MLP | Completely different |
| Per-layer info | Layer pos embeddings | Mean-pool (destroyed) | Zero layer signal |
| Compression | ~1:2 (expanding) | 5376:1 | Off by 10,000x |
| Training | 6B tokens + warmup | 2000 examples | Different regime |
| LR | 5e-5, warmup 2000 | 1e-4, no warmup | 2x too high |

## Why Toy Models Worked

At d_model=64-256, d_M2P=64 was EQUAL TO or LARGER than d_model.
The "compression" was actually an expansion. Failure only at d >> d_M2P.

## Fixes for v3

1. Add output scaling (×0.001) — unblocks learning from random noise
2. Set d_M2P = d_model — remove the bottleneck entirely
3. Keep per-layer structure — layer positional embeddings, not mean-pool
4. Add warmup — 100-200 step linear ramp
5. LR = 5e-5, gradient clipping = 1.0

## Implications for Pierre

Grassmannian A orthogonality is UNAFFECTED (QR property, not M2P).
But "1.15ms generation" needs revision — SHINE reuses base LLM as encoder,
so generation = one full forward pass (~10ms at 0.6B, ~50ms at 4B).

Key insight: keep frozen Grassmannian A (our unique contribution) +
adopt SHINE's encoder for B generation (their proven recipe).

# LEARNINGS.md — exp_intrinsic_dim_real_tasks

## Core Finding

The M2P bottleneck (d_M2P=64) captures only 77% of q_proj adapter energy at 90% threshold;
true intrinsic dimension is d_int=86 (q) / 69 (v) for a single-domain GSM8K SFT adapter.

## Why

The adapter's singular value spectrum is near-flat (σ_1²=2.4%) — 28 layers adapt with
diverse, near-orthogonal strategies rather than sharing a common low-rank subspace.
This is characteristic of near-isotropic behavior (each layer independently adapts),
not the coherent low-dimensional structure assumed by MATH.md. Cites Aghajanyan 2021
(arXiv:2012.13255): LoRA's intrinsic dimension is task-dependent, not architecture-dependent.

## Implications for Next Experiment

Expand d_M2P from 64 → 100 (minimum to achieve 90% energy capture for both projections).
This directly motivates exp_m2p_vera_bottleneck: VeRA-style shared basis reduces
parameter count (357M → 4.7M, 76x) while accommodating the expanded bottleneck.
Status is provisional — the flat spectrum mechanism (why layers adapt independently)
lacks a formal derivation; the actionable conclusion (64→100) is data-supported.

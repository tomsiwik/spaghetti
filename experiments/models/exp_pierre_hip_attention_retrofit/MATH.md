# MATH.md — HiP Attention retrofit (training-free hierarchical pruning)

## Hypothesis

HiP Attention (arxiv 2406.09827) is **training-free** hierarchical
attention pruning — it produces a token-mask via cheap hierarchical
scoring and applies dense attention only on the selected blocks. Critical
property for Pierre: **the mask can be computed once per layer and
reused across all K parallel adapter forwards**, amortizing the routing
decision cost.

This unlocks practical MoLE/X-LoRA per-token routing in Pierre by
collapsing 7× LoRA forward cost amid sparse attention.

> **Does HiP retrofit onto Gemma 4 E4B's full-attention layers (the 1/6
> non-windowed) yield ≥1.4× decode speedup at 32K context with PoLAR
> adapters bit-for-bit unchanged and behavioral score within −0.5pp
> of baseline?**

## HiP mechanism

```
Per layer:
  1. Hierarchical block scoring: cheap top-down pruning of blocks
  2. Top-k block selection per query
  3. Dense attention over selected blocks only
  Cost: O(n·log(n)·k) instead of O(n²)
```

## Pre-registered Kill Criteria

- **K1 (BEHAVIORAL)** Behavioral score on Pierre's 3-bench drops ≤ 1.0pp vs softmax baseline.
- **K2 (SPEEDUP)** Decode tok/s @ 32K context ≥ 1.2× softmax baseline.
- **K3 (ADAPTER MATH)** DARE recomposition test (load + recompose K=7 adapters): full-delta DARE result on HiP backend within ±0.5pp of softmax-attention DARE result.
- **K4 (FULL-ATTENTION LAYERS)** PPL on Gemma 4's 1/6 full-attention layers rises ≤ 5% (HiP applies only to those — sliding-window 5/6 unchanged).

## Implementation status

**SPEC ONLY — implementation pending.**

Required engineering:
1. MLX HiP kernel (~150 LOC). Hierarchical scorer + block-sparse attention.
2. Drop into Gemma 4's full-attention layer(s) only — preserve sliding-window in 5/6 layers.
3. Self-test: at full-density (k = n_blocks), HiP must reproduce dense attention.

run_experiment.py exits with INCONCLUSIVE + implementation_pending.

## References

- HiP paper (arxiv 2406.09827)
- Synergy: practical MoLE/X-LoRA — HiP mask reused across K LoRA forwards
- Research agent SSA survey: `a2c10d5138d8eea52`

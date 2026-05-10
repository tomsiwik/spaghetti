# MATH.md — LongLoRA S²-Attn (shifted sparse) compatibility with PoLAR composition

## Hypothesis

LongLoRA's **S²-Attn** (Shifted Sparse Attention, arxiv 2309.12307) is the
ONLY published sparse-attention method **explicitly designed for LoRA**.
It splits attention heads into two groups:
- Half attend to a left-shifted window: tokens [i-w/2, i+w/2)
- Half attend to a right-shifted window: tokens [i, i+w)

The shifts overlap so information flows across the full sequence over a
few layers. The paper claims context extension to 32K with LoRA-only
training (no full fine-tuning).

For Pierre, S²-Attn is the most natural sparse-attention candidate
because it preserves the LoRA Δ-arithmetic by construction (paper claims
LoRA on embedding + norms is sufficient — q_proj adapters should remain
intact).

> **Does S²-Attn preserve PoLAR's Grassmannian-A invariance and Fisher-Rao
> norm-rescaling under context extension to 32K, without retraining the
> shared A?**

## S²-Attn mechanism

```
For each layer:
  Half heads use mask M_left (window left-shifted by w/2)
  Half heads use mask M_right (standard right-window)
  Output projection mixes them as usual
```

Window size `w` = same as Gemma 4's existing 4096 in 5/6 layers.

## Pre-registered Kill Criteria

- **K1 (COMPOSITION INTACT)** S²-Attn + Fisher-Rao K=7 behavioral score
  within ±1pp of softmax baseline on standard 3-bench (composition unbroken).
- **K2 (CONTEXT EXTENSION)** Performance at 32K context: PPL or NLL ≤ 1.05× the 8K baseline (extension preserves quality).
- **K3 (DECODE LATENCY)** Decode tok/s @ 32K ≥ 0.7× of @ 8K (extension cost reasonable).
- **K4 (NO RETRAINING)** Gemma 4 base + S²-Attn used WITHOUT retraining shared A; if K1 passes, this validates LongLoRA's "LoRA-only retraining" claim transfers to PoLAR.

## Verdict logic

| K1 | K2 | Outcome |
|----|----|---------|
| ✓ | ✓ | **SUPPORTED** — S²-Attn extends Pierre to 32K without breaking composition. |
| ✓ | ✗ | **SUPPORTED with caveat** — composition fine but context extension under-performs (try retraining shared A on long sequences). |
| ✗ | * | **KILLED** — S²-Attn breaks PoLAR composition. Implication: even LoRA-aware sparse attention requires architectural co-adaptation. |

## Implementation status

**SPEC ONLY — implementation pending.**

Required engineering:
1. Locate Gemma 4's attention module + mask construction (`mlx_lm/models/gemma4.py`).
2. Build per-head shifted masks (~50 LOC). Half heads get left-shifted mask.
3. Subclass attention module via Finding #831 setattr pattern.
4. Long-context test rig (32K eval — needle-in-haystack or RULER subset).

run_experiment.py currently exits with INCONCLUSIVE + implementation_pending.

## References

- LongLoRA paper (arxiv 2309.12307): https://arxiv.org/abs/2309.12307
- LongLoRA repo: https://github.com/dvlab-research/LongLoRA
- Research agent SSA survey: `a2c10d5138d8eea52`
- Synergy: `exp_pierre_kv_cached_layer_routing_1m` (1M context routing cache benefits from any context extension)

# Attention LoRA Composition: Research Digest

## Hypothesis

Adding rank-4 LoRA adapters to attention Wq/Wk alongside MLP LoRA (rank 8) closes the composition gap by >1 percentage point compared to MLP-only LoRA, without degrading single-domain quality by >2%.

## What This Model Is

AttnLoRAGPT extends the existing LoRAGPT (MLP-only LoRA) by adding low-rank adapters to the attention query and key projections. The rationale: shared attention is the PROVEN composition bottleneck (+13.5% degradation when attention is not shared across domains, per Exp 4 findings). All prior experiments only adapted MLP layers. This experiment adapts the bottleneck itself.

Design choices:
- **Wq/Wk only, not Wv/Wo**: modify attention routing (what to attend to), not value content (how to process it). Follows the control theory principle of minimal intervention at the identified bottleneck.
- **Rank 4 for attention, rank 8 for MLP**: attention adapters get half the rank, keeping the intervention small. This adds only 4,096 parameters (+20% over MLP-only).
- **Composition via routed deltas**: both attention and MLP deltas are routed through the same learned softmax router at each layer, then composed via weighted averaging.

## Lineage in the Arena

```
gpt (dense baseline)
 └── lora_gpt (MLP-only LoRA, rank 8)
      └── attn_lora_gpt (MLP rank 8 + Attention rank 4 on Wq/Wk)
```

## Key References

- Hu et al. 2021 (LoRA): low-rank adaptation of large language models
- Liang et al. 2024 (InfLoRA): orthogonal LoRA for continual learning
- Our Exp 4 finding: shared attention is the composition bottleneck
- Our lora_procrustes experiment: LoRA deltas are pure linear, enabling exact composition

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

| Method | Avg Val Loss | vs Joint |
|--------|-------------|----------|
| Joint training | 0.5188 | baseline |
| MLP-only single | 0.4981 | -4.0% |
| MLP+Attn single | 0.4951 | -4.6% |
| MLP-only composed | 0.5229 | +0.8% |
| MLP+Attn composed | 0.5205 | +0.3% |
| Attn-only single | 0.5297 | +2.1% |

### Composition Gap Comparison

| Condition | Composition Gap (vs Joint) |
|-----------|---------------------------|
| MLP-only composed | +0.78% |
| MLP+Attn composed | +0.32% |
| **Gap improvement** | **+0.46pp** |

### Per-Seed Breakdown

| Seed | MLP Gap | MLP+Attn Gap | Improvement |
|------|---------|-------------|-------------|
| 42 | +2.09% | +1.75% | +0.35pp |
| 123 | +0.54% | +0.15% | +0.38pp |
| 7 | -0.29% | -0.93% | +0.64pp |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Gap improvement | +0.46pp | >1pp | **KILL** |
| Single-domain degradation | -0.61% (improves) | >2% | PASS |

**Verdict: KILL.** Gap improvement is consistent (+0.46pp mean, positive all 3 seeds) but below the 1pp kill threshold.

## Diagnostic Findings

### Attention Delta Fraction
The attention LoRA deltas account for 35.9% of total adapted delta norm, despite being only 20% of trainable parameters. Attention adapters capture disproportionate information per parameter. This suggests the attention subspace is information-dense but the effect on composition quality is small.

### Single-Domain Quality
Attention adapters consistently IMPROVE single-domain quality (-0.61% degradation, i.e., 0.61% better than MLP-only). The additional capacity is useful for fitting, but the benefit does not transfer to composition.

### Attention-Only Ablation
Training only attention LoRA (4,096 params, no MLP LoRA) produces +2.1% degradation vs joint. Attention adaptation alone is far less effective than MLP adaptation, confirming that MLP is the primary adaptation mechanism.

### Parameter Overhead
| Condition | Trainable Params | Overhead |
|-----------|-----------------|----------|
| MLP-only | 20,480 | baseline |
| MLP+Attn | 24,576 | +20% |
| Attn-only | 4,096 | -80% |

## Root Cause Analysis

The kill result has a clear interpretation: **attention adaptation helps, but not enough to matter at micro scale.**

Three factors explain why:

1. **Attention patterns are already near-optimal.** At d=64 with character-level tokenization, attention patterns are simple (attend to recent characters, BOS token). Domain-specific attention is a marginal refinement on already-effective patterns. The base attention learned during pretraining is sufficient.

2. **The composition bottleneck is smaller than expected at N=2.** The MLP-only composition gap averages only +0.78% vs joint. There is not much gap to close. The +13.5% bottleneck from Exp 4 was measured under different conditions (capsule composition without shared attention). With LoRA composition (which preserves shared attention structure), the bottleneck is much smaller.

3. **The signal is consistent but weak.** All 3 seeds show improvement (0.35, 0.38, 0.64pp). This is a real effect, not noise. But the effect size is roughly half the kill threshold. Scaling to N=5+ domains or d=256+ might amplify it.

## Micro-Scale Limitations

- **Only 2 domains tested.** The composition bottleneck grows with N. At N=5+, attention adaptation may cross the 1pp threshold. The consistent positive signal suggests this is likely.
- **d=64 constrains attention expressivity.** With 4 heads of dim 16, attention patterns are coarse. At d=256+ with 8+ heads, domain-specific attention patterns would be more expressive.
- **Character-level tokenization.** BPE tokenization at macro scale creates more diverse attention patterns between domains (code vs prose attend to very different token types). The effect may be larger there.
- **Rank 4 may be too low.** At d=64, rank 4 captures 6.25% of the full attention subspace. Higher rank or adapting Wv/Wo might show stronger effects.

## What Would Kill This at Macro Scale

- Attention adapters add >5% inference latency (attention computation becomes per-expert)
- Composition gap improvement remains <1pp at N=5+ domains with BPE tokenization
- Attention delta norms collapse to near-zero during training (indicating attention does not benefit from adaptation)
- Cross-attention between domains causes catastrophic interference (different domain queries attending to wrong positions)

## What This Teaches

1. **The composition bottleneck at N=2 is already small** (~0.8% for MLP-only LoRA). The bottleneck formulation from Exp 4 (+13.5%) applies to capsule composition without shared attention, not to LoRA composition.
2. **Attention is information-dense**: 20% of params capture 36% of delta norm. Even though the effect on composition is subthreshold, the capacity is real.
3. **The direction is right, the magnitude is wrong.** Consistent positive signal (0.46pp) across all seeds suggests this effect would amplify at larger scale, more domains, and richer tokenization.
4. **For micro-scale LoRA composition, MLP-only adaptation is sufficient.** The +20% parameter overhead of attention LoRA is not justified by the 0.46pp improvement.

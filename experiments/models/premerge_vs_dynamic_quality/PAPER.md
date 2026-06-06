# Pre-Merge vs Dynamic Routing Quality: Research Digest

## Hypothesis

Pre-merging all N expert LoRA deltas into a single model (averaging at 1/N strength)
degrades quality by >5% compared to dynamic top-k routing (selecting best experts at
full strength per query), especially as N grows from 5 to 20.

## What This Model Is

A direct comparison of the two simplest serving strategies for composable LoRA experts:

1. **Pre-merge**: Compute W_merged = W_base + (1/N) * sum(delta_i) once, serve as a
   single model. Zero overhead, no router needed, no runtime LoRA swapping.

2. **Dynamic top-k**: For each query, use cosine similarity to find the k best experts,
   merge their LoRAs at (1/k) weight, serve. Requires routing + runtime composition.

The question determines the architecture: if pre-merge works at N<50, the system is
dramatically simpler (no S-LoRA, no vLLM adapter swapping, just a single checkpoint).

## Lineage in the Arena

```
orthogonality_by_domain_type
  |-- content_aware_routing (KILLED: experts didn't specialize)
  |-- premerge_vs_dynamic_quality (this experiment)
```

Reuses MicroMLP + Markov chain data generation infrastructure from
orthogonality_by_domain_type. Adds base model pretraining (which the parent
experiments lacked) and forward_with_delta for pre-merged weight evaluation.

## Key References

- LoRA Soups (Ostapenko et al., 2024): Weight averaging of LoRA adapters for
  multi-task transfer. Our pre-merge strategy is equivalent to uniform LoRA Soups.
- TIES-Merging (Yadav et al., 2023): Trim-elect-merge resolves sign conflicts in
  delta merging. Not tested here (would improve pre-merge quality).
- Model Soups (Wortsman et al., 2022): Averaging fine-tuned models improves OOD
  performance without inference overhead.

## Empirical Results

### Aggregate (3 seeds)

| N  | Pre-merge | Top-1   | Top-2   | Oracle  | Base    | PM vs T1  | PM vs T2  |
|----|-----------|---------|---------|---------|---------|-----------|-----------|
| 5  | 3.2489    | 3.2480  | 3.2511  | 3.2485  | 3.2540  | +0.03%    | -0.07%    |
| 8  | 3.3276    | 3.3255  | 3.3275  | 3.3258  | 3.3293  | +0.06%    | +0.00%    |
| 12 | 3.3740    | 3.3720  | 3.3734  | 3.3722  | 3.3746  | +0.06%    | +0.02%    |
| 16 | 3.3941    | 3.3925  | 3.3935  | 3.3926  | 3.3945  | +0.05%    | +0.02%    |
| 20 | 3.3969    | 3.3956  | 3.3964  | 3.3957  | 3.3972  | +0.04%    | +0.02%    |

### Kill Criteria Assessment

| Criterion | Threshold | Observed | Verdict |
|-----------|-----------|----------|---------|
| K1: Pre-merge >5% worse than top-k | >5% | +0.06% max | NOT triggered |
| K2: Dynamic zero advantage at N<20 | <0.5% gap at all N<20 | 0.03-0.06% | TRIGGERED |

**Result: K2 TRIGGERED -- dynamic routing provides no advantage over pre-merge at N<20.**

### Expert Specialization

| Seed | Base Loss | Oracle Loss | Specialization Gap |
|------|-----------|-------------|-------------------|
| 42   | 3.4451    | 3.4450      | 0.0%              |
| 142  | 3.3512    | 3.3476      | 0.1%              |
| 242  | 3.3954    | 3.3944      | 0.0%              |

**Mean specialization: 0.0% improvement over base.**

## Micro-Scale Limitations

This is the critical caveat: **experts did not meaningfully specialize**. The LoRA
adapters learned per-domain corrections that were too small to produce measurably
different outputs on test data compared to the base model alone. This means:

1. The quality comparison between pre-merge and dynamic routing is **vacuous** --
   all strategies produce nearly identical output because the expert signal is
   negligible relative to the base model.

2. The K2 "kill" is a consequence of insufficient specialization, not a proof that
   routing is unnecessary. At macro scale with real expert specialization (e.g.,
   a Python expert that actually improves code completion by 10%), the 1/N dilution
   in pre-merge would produce measurably worse results.

3. Previous experiments (content_aware_routing) hit the same wall: micro-scale
   MicroMLP with 4 layers and d=64 cannot learn enough from Markov chain data for
   LoRA to specialize meaningfully on top of the base.

**Why specialization failed:**
- The base MLP with d=64 has limited capacity (~200K parameters).
- Pretraining on mixed data only reduces loss from 3.466 to 3.35-3.44 (modest learning).
- LoRA adapters (rank 8) can only modify a small subspace of the already-limited model.
- The Markov chain data, while having different transition matrices per domain,
  does not produce enough statistical separation for a 4-layer MLP to exploit.
- LoRA training loss drops (especially seed 142: code domains reach ~3.0) but this
  represents overfitting to training data rather than domain specialization.

**What this experiment DOES establish:**
- The comparison methodology is correct and efficient (~44s for full experiment).
- Pre-merge and dynamic routing are functionally equivalent when expert specialization
  is negligible (which is the theoretical prediction for orthogonal zero-delta experts).
- The infrastructure for testing at N=5..20 with multiple strategies works.

## Directional Finding

The mathematical analysis predicts that pre-merge vs dynamic routing quality gap
is proportional to the specialization gap (oracle vs base). Specifically:

- If oracle beats base by X%, the maximum possible gap between pre-merge and
  dynamic top-1 is bounded by X% * (1 - 1/N).
- At 0.0% specialization, the gap is zero regardless of strategy.
- The interesting regime is at macro scale where specialization is 5-20%.

**Projected at macro scale (hypothetical):**
If experts provide 10% improvement (oracle loss 10% better than base), and
N=20, then pre-merge dilutes each expert to 1/20 = 5% of its improvement.
Dynamic top-1 preserves the full 10%. Gap would be approximately 5% of
oracle loss, which is exactly at the K1 threshold.

This suggests a crossover around N=20 at macro scale, which aligns with the
inference_latency_vs_N finding that pre-merge has O(1) cost and dynamic has
O(k) cost.

## What Would Kill This

**At micro scale (already killed by K2):**
- Experts must specialize to at least 1% improvement over base for the
  comparison to be meaningful. This requires either a more capable base model
  (larger d, more layers) or richer training data (real text, not Markov chains).

**At macro scale (the real test):**
- K1 validated: pre-merge is >5% worse than dynamic top-k for N>20 with
  real expert specialization. This would confirm dynamic routing is needed.
- K2 validated: pre-merge matches dynamic at N<20. This would confirm
  pre-merge as viable for small expert counts.
- TIES/DARE merging closes the gap: if smarter merging (sign resolution,
  random pruning) eliminates the dilution penalty, pre-merge wins at all N.
- Cross-domain interference at scale: if experts are not orthogonal at
  macro scale (cos >> 0.0002), pre-merge accumulates interference that
  dynamic routing avoids.

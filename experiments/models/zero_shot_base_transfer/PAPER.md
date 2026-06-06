# Zero-Shot Base Transfer: Research Digest

## Hypothesis

LoRA experts trained on a full pretrained base can be deployed zero-shot
on SVD-reconstructed bases (W_skeleton + SVD_k(Delta)) without any expert
retraining, with expert quality degrading gracefully with base
approximation rank.

**Falsifiable**: If expert loss on any SVD base exceeds 2x loss on full
base, zero-shot transfer fails. If >50% of experts fail, the mechanism
is unreliable.

## What This Experiment Is

The parent experiment (base_free_composition) proved that experts compose
well on SVD-reconstructed bases, but it RETRAINED experts for each
condition. That conflates two claims: (1) SVD bases support composition,
and (2) experts transfer across base variants. This experiment isolates
claim (2).

Protocol:
1. Train a micro GPT conventionally (the "pretrained base")
2. Train N=4 LoRA experts on the pretrained base (standard LoRA training)
3. Save expert LoRA deltas (A @ B * scale for each MLP layer)
4. Build SVD-reconstructed bases at ranks {32, 16, 8, 4}
5. Apply the SAME expert deltas (trained on full base) to each SVD base
6. Measure expert quality on each condition WITHOUT retraining

The critical innovation: experts are trained ONCE and never retrained.
This tests whether LoRA deltas are portable across base model variants.

## Lineage in the Arena

```
adapter_taxonomy_wild (survey, proven)
  \-- relora_composition_test (proven)
       \-- base_free_composition (proven, retrained per condition)
            \-- zero_shot_base_transfer (this experiment)
```

## Key References

- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
- Eckart & Young, 1936, "The approximation of one matrix by another of lower rank"
- Parent experiment: base_free_composition (retrained baseline)

## Empirical Results

### Zero-Shot Transfer Quality (3-seed average, d=64, L=4, r=8)

| Condition | Base Loss Ratio | ZS Expert Loss Ratio | Kill (<2x) |
|-----------|----------------|---------------------|-----------|
| pretrained | 1.000 | 1.000 | SURVIVES |
| delta_full | 1.000 | 1.000 | SURVIVES |
| delta_r32 | 1.002 | 1.003 | SURVIVES |
| delta_r16 | 1.019 | 1.042 | SURVIVES |
| delta_r8 | 1.100 | 1.167 | SURVIVES |
| delta_r4 | 1.229 | 1.321 | SURVIVES |
| skeleton | 6.936 | 8.992 | N/A (negative control) |

### Comparison: Zero-Shot vs Retrained (parent experiment)

| Condition | Zero-Shot Loss Ratio | Retrained Loss Ratio | Transfer Gap |
|-----------|---------------------|---------------------|--------------|
| delta_r32 | 1.003 | 1.001 | 0.002 |
| delta_r16 | 1.042 | 1.014 | 0.028 |
| delta_r8 | 1.167 | 1.050 | 0.117 |
| delta_r4 | 1.321 | 1.095 | 0.226 |

### Kill Criteria Evaluation

| Criterion | Threshold | Worst Result | Verdict |
|-----------|-----------|-------------|---------|
| Expert loss > 2x on SVD base | > 2.0 | 1.321 (delta_r4) | **SURVIVES** |
| Expert cos > 5x on SVD base | > 5.0 | 1.00x (identical deltas) | **SURVIVES** |
| >50% experts fail | > 50% | 0% (0/48 pairs) | **SURVIVES** |

**All three kill criteria are disproven. The hypothesis SURVIVES.**

Overall verdict across all 3 seeds: **SURVIVES** (unanimously).

## Key Findings

### 1. Zero-Shot Transfer Works at High Base Quality

At rank-32 and rank-16, zero-shot transfer is nearly free:
- Rank-32: only 0.3% quality loss (vs 0.1% retrained)
- Rank-16: only 4.2% quality loss (vs 1.4% retrained)

The transfer gap (cost of not retraining) is 0.2% and 2.8% respectively.
For practical base swapping, this means experts can be reused instantly.

### 2. Zero-Shot Experts AMPLIFY Base Error (Unlike Retrained Experts)

A fundamental asymmetry emerged:
- Retrained experts compensate for base error (expert loss < base loss)
- Zero-shot experts amplify base error (expert loss > base loss)

At rank-8: base is 10% worse, retrained experts are 5% worse, but
zero-shot experts are 16.7% worse. The expert delta was optimized for
a specific weight landscape, and perturbations to that landscape are
amplified through the expert's learned corrections.

### 3. The Transfer Gap Grows with Base Perturbation

The retraining dividend (benefit of retraining over zero-shot) scales
with base perturbation magnitude. At rank-32, retraining provides
negligible benefit. At rank-4, retraining recovers 22.6% of quality.

### 4. Skeleton-Only is Catastrophic for Zero-Shot

On the random skeleton (no delta at all), zero-shot experts perform
9x worse. Unlike retrained experts (which partially compensate, achieving
only 1.27x degradation), zero-shot experts have no way to adapt.
This confirms that the base delta provides essential context that
experts depend on.

### 5. No Individual Expert Failures

Across all 3 seeds, 4 experts, and 4 SVD conditions (48 evaluations),
zero individual experts exceeded the 2x failure threshold. The transfer
is uniformly graceful, not bimodal.

## What This Enables

### Base Swapping Without Expert Retraining

If the difference between base_v1 and base_v2 can be expressed as
a low-rank delta (which is likely for fine-tuning and continued
pretraining), then:

1. Upgrade Qwen2.5 to Qwen3 -> swap the base adapter
2. All existing experts continue working zero-shot
3. Quality cost: ~4% for rank-16 base delta (likely acceptable)
4. Optional: retrain highest-value experts later for ~3% recovery

### Practical Decision Framework

| Base Change Magnitude | Zero-Shot Transfer | Action |
|----------------------|-------------------|--------|
| Small (rank-32 delta) | < 1% loss | Deploy zero-shot, no retraining |
| Moderate (rank-16) | ~4% loss | Deploy zero-shot, retrain critical experts |
| Large (rank-8) | ~17% loss | Retrain all experts recommended |
| Major (rank-4) | ~32% loss | Full retraining required |

## Micro-Scale Limitations

1. **d=64 means high sensitivity**: At macro scale (d=3584), LoRA deltas
   are proportionally smaller relative to base weights. The amplification
   effect may be weaker at scale, making zero-shot transfer work BETTER.

2. **Same skeleton assumption**: Both training and transfer bases share
   the same random initialization. Real base swapping involves different
   initializations, adding another source of error.

3. **SVD perturbation only**: Real base model updates involve continued
   pretraining (non-SVD perturbation). The transfer gap for arbitrary
   weight changes is unknown.

4. **Toy data**: Character-level name generation with overlapping domains.

5. **No fine-tuning recovery tested**: A few adaptation steps on the new
   base might close most of the transfer gap cheaply. This middle ground
   (between zero-shot and full retraining) was not tested.

6. **Expert loss only**: No text generation quality evaluation. NTP loss
   is a proxy for coherent output.

## What Would Kill This

### At Micro Scale
- Showing that non-SVD base perturbations (e.g., random noise, continued
  training) cause disproportionately larger transfer gaps
- Evidence that certain expert-domain combinations are brittle to transfer
  while others are robust (bimodal distribution)
- A d=128 or d=256 experiment showing the amplification factor c increases
  with model dimension

### At Macro Scale
- Zero-shot transfer on real base model upgrades (e.g., Qwen2.5 to Qwen3)
  producing >50% quality loss on most experts
- The base delta between model versions being full-rank (no low-rank
  structure), eliminating the rank-dependent graceful degradation
- Expert fine-tuning on the new base requiring >100 steps to recover
  (making "cheap adaptation" impractical)

## Artifacts

- `micro/models/zero_shot_base_transfer/zero_shot_base_transfer.py` -- full experiment
- `micro/models/zero_shot_base_transfer/test_zero_shot_base_transfer.py` -- 8 tests
- `micro/models/zero_shot_base_transfer/results_seed_42.json`
- `micro/models/zero_shot_base_transfer/results_seed_123.json`
- `micro/models/zero_shot_base_transfer/results_seed_7.json`
- `micro/models/zero_shot_base_transfer/results_aggregate.json`
- `micro/models/zero_shot_base_transfer/MATH.md` -- mathematical foundations
- Total experiment time: ~30 seconds per seed on Apple Silicon (M-series)

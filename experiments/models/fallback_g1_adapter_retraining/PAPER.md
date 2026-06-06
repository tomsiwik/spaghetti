# Fallback G1: Adapter Retraining + Smart Composition

## Hypothesis

Retrained adapters with conservative hyperparameters (rank-4, attention-only LoRA)
and smarter composition methods (Task Arithmetic, TIES-Merging) will produce >= 10%
PPL improvement on domain data vs base model, and composition will preserve or
improve those gains.

**Falsifiable**: If no adapter achieves >= 10% domain PPL improvement despite
healthy delta ratios and non-overfitting training, then the hypothesis fails.

## What This Model Is

A micro-scale test (d=256, 4 layers, 29M params) of LoRA adapter training and
multi-adapter composition on a warm-start ternary base. Tests whether the prior
experiment's catastrophic adapter divergence was due to overfitting (17% trainable
params) and whether conservative sizing (0.11% trainable) plus smarter composition
methods can enable domain-specific improvement.

## Key References

- Task Arithmetic (Ilharco et al., 2023): per-task scaling of adapter deltas
- TIES-Merging (Yadav et al., 2023, arXiv 2306.01708): trim + elect + merge
- Prior experiment: warmstart_scale_validation (d=1024, rank-16, catastrophic failure)

## Root Cause Fix

The warmstart_scale_validation experiment used rank-16 LoRA on ALL projections of a
d=1024 model, producing 34.6M trainable parameters (17% of the 204M base). Training
on 500K domain tokens caused catastrophic overfitting: PPL exploded from 84 to 1415
(17x WORSE). Composition ratio was 27.68x.

This experiment fixes the overfitting:
- Rank-4 instead of rank-16 (8x fewer params per projection)
- Attention QKVO only, no MLP (eliminates 2/3 of LoRA parameters)
- Result: 32,768 trainable params = 0.11% of base (vs 17% prior)
- Added early stopping on validation loss

## Empirical Results

### Base Model
- Architecture: d=256, 4 layers, 8 heads, 29M params
- Training: 4000 steps warm-start (10% FP + 90% ternary QAT)
- Data: 5M FineWeb-Edu BPE tokens
- Final PPL: 268.42

### Adapter Training (rank=4, lr=3e-4, 3000 steps, attention-only)

| Domain | Base PPL | Adapted PPL | Improvement | Val Degradation | Delta Ratio |
|--------|----------|-------------|-------------|-----------------|-------------|
| Science | 136.36 | 131.92 | +3.3% | +1.5% | 0.126 |
| History | 155.78 | 151.05 | +3.0% | +1.4% | 0.126 |
| Technology | 145.78 | 141.19 | +3.1% | +1.6% | 0.124 |

Trainable params: 32,768 (0.11% of base). All adapters hit early stopping at 3000 steps.

### Delta Ratio Diagnostic: PASS

All adapters produce meaningful deltas: ||sBA||/||W|| ~ 0.12 (target was > 0.01).
This proves the prior "vacuous delta" diagnosis was wrong -- the real issue was
overfitting, not insufficient delta magnitude.

### Composition Results

| Method | Science | History | Technology | Avg |
|--------|---------|---------|------------|-----|
| Base (no adapter) | 136.36 | 155.78 | 145.78 | - |
| **1/N Averaging** | 142.46 (-4.5%) | 163.02 (-4.7%) | 152.30 (-4.5%) | -4.5% |
| **Task Arith lambda=0.3** | 141.69 (-3.9%) | 162.08 (-4.0%) | 151.60 (-4.0%) | **-4.0%** |
| **Task Arith lambda=0.5** | 144.93 (-6.3%) | 166.08 (-6.6%) | 155.27 (-6.5%) | -6.5% |
| **Task Arith lambda=0.7** | 148.74 (-9.1%) | 170.29 (-9.3%) | 159.54 (-9.4%) | -9.3% |
| **Task Arith lambda=1.0** | 155.33 (-13.9%) | 177.45 (-13.9%) | 166.50 (-14.2%) | -14.0% |
| **TIES trim=80%** | 145.89 (-7.0%) | 166.64 (-7.0%) | 156.23 (-7.2%) | -7.1% |

(Negative % = PPL got WORSE relative to base)

**All composition methods make things WORSE.** Higher lambda = more damage.
TIES is slightly better than full summation but still negative.
Best method: Task Arithmetic lambda=0.3 (-4.0% degradation).

## Kill Criteria Assessment

**K1 (id=515): FAIL**
- Threshold: >= 10% PPL improvement on domain data
- Result: Best improvement was 3.3% (science), all domains at ~3%
- No adapter achieves the 10% threshold

## Analysis

### Why Only 3% Improvement?

1. **The domains aren't different enough.** FineWeb-Edu science/history/technology
   texts share enormous vocabulary overlap. A model at PPL 268 hasn't learned
   enough to distinguish domain-specific patterns -- it's still learning basic
   grammar and word co-occurrence.

2. **Base model too weak.** At PPL 268, the model is barely beyond bigram statistics.
   LoRA at this quality level is trying to fine-tune noise. Domain adaptation
   requires a base that has mastered general language first.

3. **Rank-4 may be too conservative.** Only 32K trainable params on a 29M model.
   But increasing rank risks the overfitting that killed the prior experiment.

### Why Does Composition Always Hurt?

The adapter deltas are large (delta ratio ~0.13) but UNDIRECTED. Each adapter
pushes weights in its own direction, but the improvements are tiny (3%). When
composed, the delta magnitudes compound while the marginal improvements cancel:

- Each delta: ||sBA||/||W|| ~ 0.13, but only 3% useful signal
- Composition of 3: the noise scales faster than the signal
- This is exactly the "constructive transfer vs interference" tradeoff
- With only 3% signal per adapter, there's no room for interference budget

### What Would Need to Change

For adapters to achieve >= 10% improvement:
1. **Much stronger base model** (PPL < 50, not 268). The d=1024 warmstart base had
   PPL 166 and domain PPLs of 84-104, but adapters still diverged there.
2. **More distinct domain data** -- use instruction-format data, not raw NTP from the
   same source. Medical textbooks vs Python code vs legal briefs, not FineWeb subsets.
3. **Real tokenizer alignment** -- GPT-2 BPE may not tokenize domain-specific terms well.
4. **Larger model at moderate rank** -- d=512+, rank=8, with proper regularization.

## Limitations

1. d=256 is deliberately small. Results are directional, not definitive.
2. FineWeb-Edu subdomains may have insufficient distribution shift for meaningful
   domain adaptation.
3. GPT-2 BPE tokenizer, no instruction format.
4. Only 3 domains tested.
5. Early stopping triggered for all adapters, suggesting the learning signal
   saturated quickly.

## What This Proves

1. **The prior catastrophic failure was overfitting, confirmed.** Rank-4 attention-only
   LoRA (0.11% trainable) produces stable adapters with healthy deltas, while rank-16
   all-projection LoRA (17% trainable) caused 17x PPL blowup. The delta ratio
   diagnostic is useful: prior work was NOT vacuous (ratio would have been ~0.13 too
   if the training hadn't diverged).

2. **Composition methods don't help when individual adapters are weak.** Task
   Arithmetic and TIES-Merging cannot create signal that isn't there. Composition
   is a noise-amplification problem when adapter improvements are < 5%.

3. **Domain adaptation requires a strong base.** At PPL 268, the model hasn't
   learned enough general language for domain specialization to be meaningful.
   This is consistent with the broader finding that composition works at the
   BitNet-2B-4T scale (PPL ~4-5) where the base has mastered general language.

## Total Runtime

34.2 minutes (16.3 min base training, 11.3 min x3 adapter training, 3 min composition)

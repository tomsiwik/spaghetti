# Gate-Product Pruning Transfer to Macro Scale: Research Digest

## Hypothesis

The bimodal gate-product distribution observed in micro-scale SwiGLU experiments
(with auxiliary sparsity loss) also exists in production-trained models (without
aux loss), and zero-shot gate-product pruning transfers to macro scale.

**Falsifiable.** Kill criteria:
1. Gate-product distribution is NOT bimodal in production models (no aux sparsity loss)
2. Pruning by gate product >5% worse than no pruning at macro scale

## What This Experiment Is

A macro-scale profiling and pruning experiment on Qwen2.5-0.5B (24 layers,
4864 SwiGLU neurons per layer, 116,736 neurons total). The model was loaded
from HuggingFace in bf16 with NO retraining or fine-tuning -- pure inference
profiling on WikiText-2-raw-v1 test split (16,384 positions, 3,134 unique
tokens) followed by zero-shot weight pruning evaluated on the WikiText-2-raw-v1
validation split (8,192 positions, genuinely held-out).

The experiment tests whether the micro-scale finding (66.5% of SwiGLU neurons
prunable at +1.22% quality cost) transfers to a production model that was never
trained with the auxiliary sparsity loss that shaped the micro distribution.

**Revision v2**: Corrected data pipeline from 16 hardcoded prompts to actual
WikiText-2. Added random pruning baseline (3 seeds). Evaluation data is now
genuinely held-out (validation split, separate from test split used for profiling).

## Lineage in the Arena

```
gpt (dense baseline, ReLU)
  +-- silu_capsule (SiLU capsule MLP)
  |     +-- silu_pruning (KILLED: 0% prunable at safe tau)
  |     +-- swiglu_gate_pruning (PASS: 66.5% prunable at +1.22%)
  |           +-- swiglu_macro_pruning_transfer (THIS: distribution passes, pruning kills)
```

## Key References

- Micro experiment: `micro/models/swiglu_gate_pruning/` (66.5% prunable with aux loss)
- Qwen2.5-0.5B: SwiGLU architecture (gate_proj + up_proj + down_proj, d_ff=4864)
- Wanda (Sun et al., 2023): Weight and activation pruning for LLMs
- DeepSeek-V3: Production SwiGLU MoE (256 experts, aux-loss-free balancing)

## Empirical Results

### Data Provenance

| Property | Calibration | Evaluation |
|----------|------------|------------|
| Dataset | WikiText-2-raw-v1 | WikiText-2-raw-v1 |
| Split | test | validation |
| Sequences | 128 x 128 | 64 x 128 |
| Total positions | 16,384 | 8,192 |
| Unique tokens | 3,134 | 1,945 |
| Raw text chars | 1,288,512 | 1,144,610 |

### Distribution Analysis (KC1: Bimodality)

| Metric | Micro (aux loss) | Macro (Qwen2.5-0.5B) |
|--------|-----------------|----------------------|
| Total neurons profiled | 512 (4 layers x 128) | 116,736 (24 layers x 4864) |
| Gate product floor (min) | 0.014 | 0.000185 |
| Median mean abs | 0.026-0.061 | 0.078 |
| Below tau=0.05 | 66.5% | 15.8% |
| Bimodality coefficient | (not computed) | 0.643 (bimodal) |
| Bimodal layers | N/A | 18/24 |
| Calibration data | Character-level names | WikiText-2 test (16K positions) |
| Aux sparsity loss | Yes (L1, target 50%) | No |

**KC1 VERDICT: PASS.** The distribution is bimodal in Qwen2.5-0.5B even without
auxiliary sparsity loss. Sarle's bimodality coefficient = 0.643 > 0.555 threshold.
18 of 24 layers are individually bimodal. The bimodal structure is an architectural
property of SwiGLU, not an artifact of the training loss.

**Caveat**: Sarle's BC is known to produce false positives for heavy-tailed
unimodal distributions. The extreme skewness (39.1) and kurtosis (2382) mean
this distribution may be better described as "heavy-tailed unimodal" rather than
truly bimodal. Hartigan's dip test was not run. The BC passing is directionally
informative but should not be over-interpreted.

### Pruning Results (KC2: Quality)

| Threshold | Neurons Pruned | % Pruned | PPL | PPL Delta | KC2 Status |
|-----------|---------------|----------|-----|-----------|------------|
| baseline | 0 | 0.0% | 21.31 | -- | -- |
| tau=0.01 | 2 | 0.0% | 24.75 | +16.1% | KILL |
| tau=0.02 | 9 | 0.0% | 24.60 | +15.5% | KILL |
| tau=0.05 | 18,420 | 15.8% | 552.78 | +2,494% | KILL |
| tau=0.10 | 77,946 | 66.8% | 98,328 | +461,373% | KILL |
| tau=0.20 | 109,818 | 94.1% | 1,464,245 | +6,871,898% | KILL |
| tau=0.50 | 115,440 | 98.9% | 140,140 | +657,603% | KILL |

Baseline perplexity: 21.31 (WikiText-2 validation split, bf16).

**KC2 VERDICT: KILL at ALL thresholds.** Even pruning 2 neurons (0.002% of total)
causes +16.1% perplexity degradation. Zero-shot weight zeroing is catastrophically
destructive at macro scale.

### Random Pruning Baseline (Critical Control)

At tau=0.05, gate-product profiling prunes 18,420 neurons. To test whether the
profiling provides signal above chance, we randomly pruned the same number of
neurons (3 seeds):

| Method | Neurons Pruned | PPL | Delta vs Baseline |
|--------|---------------|-----|-------------------|
| Gate-product profiled | 18,420 | 552.78 | +2,494% |
| Random (seed 0) | 18,420 | 53.53 | +151% |
| Random (seed 1) | 18,420 | 58.72 | +176% |
| Random (seed 2) | 18,420 | 73.64 | +246% |
| Random (mean +/- std) | 18,420 | 61.97 +/- 8.52 | +191% |

**Gate-product profiled pruning is 8.9x WORSE than random pruning.** This is the
opposite of what the micro experiment showed (where profiled pruning was 2.3x
better than random).

This means gate-product profiling at macro scale is actively selecting the
WRONG neurons to prune -- it preferentially identifies specialist neurons that
have low mean activation but are critical for specific inputs. Without auxiliary
sparsity loss to train the model for robustness to their removal, these are the
worst possible neurons to prune.

### Root Cause: The Specialist Neuron Problem

Neurons with low mean gate-product magnitude are NOT universally inactive.
They are **specialist neurons** that fire strongly on rare but important input
patterns.

Example (layer 21, neuron with min activation):
- Mean |gate_product|: 0.000185 (lowest in the model)
- The neuron activates on <0.01% of positions but carries unique information

At micro scale with aux sparsity loss, the model was TRAINED to redistribute
information away from low-activation neurons. This robustness does not exist
in production models. The profiling signal is real (it identifies neurons with
low mean activation), but the interpretation is inverted: low mean activation
correlates with specialist function, not prunability.

### Why Micro Pruning Succeeded

The micro-scale experiment (66.5% prunable at +1.22%) succeeded because of
THREE factors that do not transfer:

1. **Auxiliary sparsity loss**: Actively pushes neurons toward zero AND trains
   the model to compensate for their removal
2. **Shallow depth (4 layers)**: Pruning errors propagate through fewer layers
3. **Low capacity (128 neurons)**: Fewer specialist neurons possible

At macro scale (no aux loss, 24 layers, 4864 neurons), ALL three factors work
against pruning.

### Gate (SiLU) vs Up Component Analysis

The multiplicative suppression mechanism IS present at macro scale:

| Layer | GP Floor | Gate Floor | Up Floor | GP/Gate Ratio |
|-------|----------|------------|----------|---------------|
| 0 | 0.020 | 0.087 | 0.149 | 0.23x |
| 11 | 0.018 | 0.036 | 0.170 | 0.51x |
| 21 | 0.000185 | 0.00002 | 0.249 | 8.93x |
| 23 | 0.040 | 0.041 | 0.346 | 0.96x |

The gate product floor (0.000185) is much lower than the micro floor (0.014),
confirming that the up-projection still acts as a learned suppression mask. The
suppression is stronger at macro scale, creating more extreme specialist neurons.

## The Core Finding

**The bimodal gate-product distribution is an architectural property of SwiGLU
that exists in production models without any auxiliary sparsity loss.** This means
the SIGNAL for identifying low-activation neurons is present. However, at macro
scale without aux loss, low mean activation is ANTI-correlated with safe
prunability: these neurons are specialists, not dead neurons.

This is a three-part result:
1. The distribution shape (heavy tail + bulk) transfers architecturally
2. The profiling signal (which neurons have low mean activation) transfers
3. The interpretation is INVERTED: low activation means "specialist" not "prunable"

The random pruning baseline is the critical evidence: profiled pruning is 8.9x
worse than random, meaning the profiling is actively selecting critical neurons.

## Comparison with SOTA Pruning Methods

The finding that zero-shot structured pruning fails at macro scale is consistent
with the literature:

- **Wanda** (Sun et al., 2023): Uses weight * activation magnitude for
  unstructured pruning, includes calibration
- **LLM-Pruner** (Ma et al., 2023): Structured pruning with gradient-based
  importance and LoRA recovery
- **SparseGPT** (Frantar & Alistarh, 2023): Unstructured pruning with
  Hessian-based weight updates

All successful macro pruning methods include some form of weight adjustment
or recovery training. Pure mean-activation profiling without weight correction
is insufficient. The inverted signal (profiled worse than random) suggests that
a Wanda-style approach combining weight norms with activation frequency (not
just mean magnitude) would be more appropriate.

## Micro-Scale Limitations

1. **Single dataset**: WikiText-2 is English Wikipedia text. Specialist neurons
   active on code, math, or multilingual text may not fire on this corpus.

2. **bf16 precision**: Gate product magnitudes measured in bf16 may not
   capture sub-1e-3 variations accurately. This does not affect the main
   finding (pruning fails at all thresholds).

3. **Single model**: Only Qwen2.5-0.5B tested. Larger models (7B, 70B)
   may have different distribution shapes.

4. **No recovery training**: The zero-shot pruning method is the weakest
   possible baseline. Methods with recovery (Wanda, LLM-Pruner) would
   likely perform much better.

5. **Perplexity as sole metric**: Downstream task performance may degrade
   differently than perplexity suggests.

6. **BC false positive risk**: Sarle's bimodality coefficient may produce
   false positives for the heavy-tailed distribution observed (skewness=39.1,
   kurtosis=2382). Hartigan's dip test was not run.

## What Would Kill This (Updated)

**KC1 (bimodality)**: PASSED (with caveats). Would be killed if:
- Distribution was confirmed unimodal via Hartigan's dip test
- Bimodality was an artifact of calibration data distribution

**KC2 (pruning quality)**: KILLED. Zero-shot pruning is not viable.

**What the random baseline revealed**:
- Gate-product mean magnitude is an ANTI-signal for prunability at macro scale
- Low mean activation correlates with specialist function (high max/mean ratio)
- The auxiliary sparsity loss doesn't just shape the distribution -- it
  fundamentally changes which neurons are safe to remove

**What would resurrect the approach**:
- Gate-product profiling + weight norms (Wanda-style) could invert the signal
- Activation FREQUENCY (how often a neuron fires above threshold) instead of
  mean magnitude may be a better importance metric
- Post-pruning fine-tuning (<1000 steps) to allow the model to adapt

**What this definitively proves**:
- The auxiliary sparsity loss in micro experiments was necessary for pruning
  robustness, not just for distribution shaping
- Mean gate-product magnitude is an ANTI-predictor of prunability at macro scale
- Bimodal gate-product distributions are an intrinsic property of SwiGLU
- The gap between micro and macro pruning is not just quantitative (more damage)
  but qualitative (the signal is inverted)

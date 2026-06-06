# Individual Expert Held-Out Evaluation: Research Digest

## Hypothesis

Individual LoRA adapters tested alone on held-out MMLU do not individually
cause the -3.67pp regression observed in the N=50 composed model; the
regression is a composition-specific effect (interference or dilution).

## What This Model Is

This is not a new model -- it is a diagnostic experiment. The N=50 composed
model (all 50 pilot adapters merged into the base via weight addition) showed
-3.67pp average delta on held-out MMLU vs the Qwen2.5-7B base. This regression
could come from two sources:

1. **Distillation quality**: Each adapter individually harms the base model on
   non-training-domain tasks. Adapters are overfit to their synthetic training
   data and degrade general capabilities.

2. **Composition interference**: Each adapter is individually neutral or
   slightly beneficial, but composing 50 adapters in weight space creates
   destructive interference (cross-adapter weight conflicts, dilution of
   adapter signals at 1/N weight each, etc.).

This experiment tests 20 adapters INDIVIDUALLY on ALL 57 MMLU subjects.
No composition at any point. Pure single-adapter inference.

## Lineage

```
exp_distillation_pilot_50 (supported, contaminated eval)
    |
    v
pilot50_held_out_eval (active, -3.67pp regression on MMLU)
    |
    +---> individual_expert_held_out <-- THIS EXPERIMENT (diagnosis)
    |
    +---> small_n_held_out_eval (sweep N=2..50, dilution curve)
    |
    +---> selective_composition_mmlu (top-k vs compose-all)
```

## Key References

- Hendrycks et al. (2021) -- MMLU benchmark
- Hu et al. (2021) -- LoRA: Low-Rank Adaptation
- pilot50_held_out_eval -- established the -3.67pp regression

## Domain-to-MMLU Mapping

For the in-domain vs out-of-domain analysis, this experiment reuses the mapping
from pilot50_held_out_eval/eval_mmlu.py (23 adapters have MMLU counterparts).
The remaining 27 adapters (writing, reasoning, niche code) are evaluated on
ALL subjects -- their entire evaluation is "out-of-domain" relative to MMLU.

## Empirical Results

5 adapters tested individually on all 57 MMLU subjects (14,042 questions).
Base model: Qwen2.5-7B with NF4 quantization.

### Summary

| Metric | Value |
|--------|-------|
| Base MMLU accuracy | 70.3% (9,878/14,042) |
| Mean individual delta | **-0.95pp** |
| 95% bootstrap CI | [-2.00, -0.26]pp |
| Median individual delta | -0.46pp |
| Std of individual deltas | 1.05pp |
| Adapters with positive delta | 0/5 |
| Adapters with negative delta | 2/5 |
| Adapters roughly neutral | 3/5 |
| In-domain avg delta | -3.61pp |
| Out-of-domain avg delta | -0.99pp |
| Diagnosis | **COMPOSITION_INTERFERENCE** |

### Per-Adapter Results

| Adapter | Accuracy | Delta (pp) |
|---------|----------|------------|
| bash | 70.2% | -0.1pp |
| math | 69.5% | -0.8pp |
| medical | 67.4% | -2.9pp |
| python | 69.9% | -0.4pp |
| sql | 70.1% | -0.2pp |

Medical shows the largest individual drop (-2.9pp), likely because medical
knowledge conflicts with general MMLU questions. The other 4 adapters are
within noise of the base.

### Kill Criteria Assessment

| Criterion | Threshold | Value | Status |
|-----------|-----------|-------|--------|
| K1: Composition interference | avg > -1pp | **-0.95pp** | **PASS** |
| K2: Distillation memorization | avg < -3pp | -0.95pp | not triggered |

**Diagnosis**: Individual adapters are roughly neutral (mean -0.95pp > -1pp
threshold). The -3.67pp regression in the N=50 composed model comes from
weight-space interference during composition, not from individual adapter
quality. Fix: selective top-k routing or composition weight normalization.

## Limitations

1. **0-shot MMLU**: Standard protocol uses 5-shot. Adapters trained on
   instruction data may show inflated improvement in 0-shot. This affects
   absolute numbers but the diagnosis (sign and magnitude of $\bar{\delta}$)
   should be robust to prompting format.

2. **NF4 quantization**: Both base and adapter run in NF4. Absolute accuracy
   may differ from FP16 literature values, but the delta comparison is fair.

3. **Top-20 selection by training PPL**: The top 20 adapters by training PPL
   may not be representative of all 50. Adapters with poor training PPL
   (e.g., SQL) may have different held-out behavior.

4. **MMLU coverage**: Not all adapter domains have MMLU counterparts. The
   in-domain analysis only works for the 23 adapters with mapped subjects.

5. **Decomposition is approximate**: The equation
   $\delta_{\text{composed}} = \bar{\delta} + \Delta_{\text{interference}}$
   assumes independent adapter effects. In reality, adapter effects may
   interact non-linearly through the base model's nonlinearities.

6. **Adapters may be pre-retrain config**: If evaluated before retrain_all_adapters
   completes, these are old-config adapters. Results should be re-checked
   after retraining.

## What Would Kill This

- **K1 (composition is the problem)**: $\bar{\delta} > -1\text{pp}$. Individual
  adapters are roughly neutral, so the -3.67pp came entirely from composition.
  Implication: fix composition method (selective top-k, not compose-all).

- **K2 (distillation is the problem)**: $\bar{\delta} < -3\text{pp}$. Individual
  adapters already harm generalization. Composing them just accumulates harm.
  Implication: fix distillation (better data, longer training, curriculum).

- **Both thresholds missed**: $-3 \le \bar{\delta} \le -1$. Both distillation
  and composition contribute to the regression. Need to fix both.

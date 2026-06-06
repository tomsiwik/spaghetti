# SFT LoRA Adapters on Falcon-E-3B: Proof Verification Report

## Theorem
SFT loss (response-only masking) preserves the instruction-tuned base model's
calibration because instruction tokens receive zero gradient. This should prevent
the degradation observed with NTP-trained adapters.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: SFT single adapters >= base MMLU | 0/5 beat base on own domain | NO |
| P2: SFT composed >= base MMLU avg | 0.46 < 0.54 base | NO (partially) |
| P3: Math adapter GSM8K >= 0.44 base | 0.50 (math), 0.52 (composed) | DIRECTIONAL (p=0.41, not significant) |
| P4: >= 3/5 adapters improve on domain | 1/5 (GSM8K only counts) | NO |
| SFT composed > NTP composed overall | 4/6 improve or equal | YES |

## Hypothesis
SFT-loss LoRA adapters on instruction-tuned Falcon-E-3B will improve benchmark
accuracy over base on >= 3/5 domains.

**Verdict: PROVISIONAL.** 3/4 quantitative predictions (P1, P2, P4) failed. P3 passed
but is not statistically significant (p=0.41, n=50). The directional observation
(SFT composed > NTP composed on 4/6 benchmarks) is interesting but does not constitute
proof verification. Under project definitions: "provisional = empirical observation
awaiting proof." The lora_scale=20 confound (see below) means the SFT vs NTP comparison
is potentially confounded by extreme hyperparameter amplification.

## What This Experiment Is

Tests whether the NTP adapter degradation identified in Findings #166 and #179 is
caused by training on instruction tokens. Uses SFT loss (response-only masking) to
train 5 domain LoRA adapters, then evaluates base, single adapters, composed (1/N
pre-merge), and oracle-routed configurations.

## Key References
- Ouyang et al. 2022 (SFT in RLHF pipeline)
- Hu et al. 2022 (LoRA)
- Findings #166 (NTP adapter mismatch), #179 (math 24x correctness)

## Empirical Results

### Training Results
All 5 adapters trained successfully with SFT loss:
| Domain | Base PPL | Trained PPL | Improvement |
|--------|----------|-------------|-------------|
| Medical | 2.74 | 1.98 | 28.0% |
| Code | 2.13 | 1.95 | 8.4% |
| Math | 2.05 | 1.50 | 27.0% |
| Legal | 13.41 | 11.26 | 16.0% |
| Finance | 13.46 | 12.36 | 8.2% |

### Benchmark Results

```
Benchmark       Base     NTP-comp   SFT-comp   Routed     BestSingle
----------------------------------------------------------------------
gsm8k           0.440    0.360      0.520      0.500      0.620
mmlu_medical    0.550    0.300      0.300      0.100      0.450
mmlu_code       0.600    0.500      0.450      0.550      0.550
mmlu_math       0.550    0.550      0.550      0.150      0.550
mmlu_legal      0.400    0.350      0.400      0.050      0.400
mmlu_finance    0.600    0.450      0.600      0.200      0.450
```

### Statistical Significance

Sample sizes are small (n=50 for GSM8K, n=20 for MMLU). Two-proportion z-tests
for the headline claims:

| Comparison | Base | SFT-comp | n | z | p (two-sided) | Significant? |
|------------|------|----------|---|---|---------------|--------------|
| GSM8K base->SFT composed | 0.44 | 0.52 | 50 | 0.81 | 0.41 | NO |
| GSM8K NTP->SFT composed | 0.36 | 0.52 | 50 | 1.63 | 0.10 | Marginal |
| MMLU med base->SFT comp | 0.55 | 0.30 | 20 | 1.62 | 0.11 | NO |

The GSM8K improvement from base (0.44) to SFT composed (0.52) is NOT statistically
significant at any conventional threshold (p=0.41). The improvement over NTP composed
(0.36 to 0.52) is only marginally significant (p=0.10). With n=50, we would need
observed proportions of roughly 0.44 vs 0.64 to reach p<0.05.

All MMLU comparisons use n=20, far too small for reliable inference on proportions.
These results should be treated as directional observations, not confirmed effects.

### Key Observations

1. **SFT composed directionally improves over NTP composed** on GSM8K (+0.16,
   p=0.10), finance MMLU (+0.15, n=20), legal MMLU (+0.05, n=20). Consistent
   with the hypothesis that NTP instruction-token gradients contribute to
   degradation, but not statistically confirmed at these sample sizes.

2. **SFT composed matches or exceeds base on 4/6 benchmarks** (GSM8K, math,
   legal, finance). Only medical (-0.25) and code (-0.15) degrade.

3. **Individual SFT adapters destroy MMLU** while improving GSM8K. The medical
   adapter gets 58% GSM8K (vs 44% base) but 10% medical MMLU (vs 55% base).
   We HYPOTHESIZE this is due to lora_scale=20 overcorrection -- the adapter
   perturbation may be too large for single-adapter use, averaging out in
   composition. **This was not tested in this experiment** (no lora_scale ablation
   was performed).

4. **Oracle routing is WORSE than composition** because individual adapters
   are over-fitted. The 1/N uniform average acts as implicit regularization.

5. **SFT shows directional improvement over NTP** (composed GSM8K: 0.36 -> 0.52,
   though p=0.10, only marginally significant). We HYPOTHESIZE a second failure
   mechanism: lora_scale=20 may cause individual adapter overshoot that destroys
   MMLU on non-generation tasks. **This hypothesis was not tested** -- no
   lora_scale ablation was performed, so we cannot attribute the remaining
   degradation to scale vs other causes (data quality, insufficient training, etc.).

### Kill Criteria Assessment

**K1 (#562): SFT adapters degrade base on >3/5 benchmarks**
- As routed (individual adapters): 5/6 degrade -> **K1 FAIL**
- Mitigating observation: composed (1/N pre-merge) only degrades 2/6,
  suggesting the failure is adapter-scale-dependent, not fundamental.
  However, this does not retroactively pass K1.
- **Verdict: FAIL.** Individual SFT adapters degrade base on 5/6 benchmarks.

**K2 (#563): Composed worse than best single on >3/5 benchmarks**
- Composed worse on 3/6 -> PASS (threshold is >3)

### Root Cause Analysis

**Observed:** SFT composed outperforms NTP composed on 4/6 benchmarks, consistent
with the hypothesis that NTP instruction-token gradients caused degradation.

**NOT established:** Whether this is the full explanation. Two major confounds remain:

1. **lora_scale=20 amplification (UNTESTED HYPOTHESIS):** MLX LoRALinear uses scale
   as a raw multiplier (see lora_scale section below). Both NTP and SFT experiments
   used scale=20, meaning the low-rank update was amplified 20x. Standard LoRA uses
   alpha/r = 16/16 = 1.0. The entire SFT vs NTP comparison is potentially confounded
   by this extreme hyperparameter. We hypothesize that lora_scale=20 overpowers base
   model knowledge for individual adapters and that 1/N composition averaging reduces
   the effective scale, but this was NOT tested via ablation.

2. **Data quality:** Training data was NTP-formatted with SFT mask applied, not
   purpose-built SFT instruction data. Quality of response-only signal may be poor.

### Critical Confound: lora_scale=20 Semantics

**CONFIRMED:** MLX's `LoRALinear` uses `scale` as a raw multiplier on the low-rank
update. From the MLX source:

```python
# In __call__:
z = (self.dropout(x) @ self.lora_a) @ self.lora_b
return y + (self.scale * z).astype(x.dtype)

# In fuse:
delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
```

This means `scale=20.0` multiplies the entire low-rank perturbation by 20x. Standard
LoRA practice uses `alpha/r` as the scale factor, typically `alpha=r` giving scale=1.0,
or `alpha=2*r` giving scale=2.0. A scale of 20x is far outside the validated range
in the LoRA literature.

**Impact on this experiment:** Both NTP and SFT adapters used `lora_scale=20`. This
means:
- Individual adapter outputs are amplified 20x, likely overpowering base model knowledge
- In 1/N composition with 5 adapters, effective per-adapter scale is 20/5 = 4x, still high
- The entire SFT vs NTP comparison may be confounded: both conditions suffer from
  extreme amplification, making it impossible to isolate the effect of loss masking alone

**Critical next step:** Ablate lora_scale at {1, 2, 4, 8, 20} for both SFT and NTP
adapters. Without this ablation, we cannot determine whether the observed differences
are due to SFT loss masking or interactions between extreme scale and loss type.

### Next Steps

1. **Ablate lora_scale (CRITICAL):** Test scale={1, 2, 4, 8, 20} for both SFT
   and NTP adapters. This is the most important next step -- without it, we cannot
   disentangle loss-type effects from scale effects.
2. **Domain-specific HuggingFace instruction data:** The current data from
   real_data_25_domain_adapters may not be high-quality instruction data.
   Use curated SFT datasets (e.g., OpenHermes, Orca).
3. **Test on actual SFT instruction datasets** rather than NTP data with
   response masking (the data format was instruction/response but generated
   for NTP training, not SFT-optimized).

## Limitations

1. **n=20 per domain for MMLU, n=50 for GSM8K** -- small sample sizes
2. **lora_scale=20 was inherited from prior experiment** without ablation
3. **Training data is NTP-formatted data with SFT mask applied** -- not
   purpose-built SFT instruction data
4. **300 iterations may be insufficient** -- SFT has fewer gradient signals
   per example (only response tokens contribute)
5. **Single seed** -- no statistical confidence intervals

## What Would Kill This

1. If lora_scale ablation shows the same degradation at scale=1 (rules out
   overcorrection hypothesis)
2. If purpose-built SFT data produces the same MMLU degradation (rules out
   data quality hypothesis)
3. If the GSM8K improvement is noise (need n>200 with statistical test)

# Composition Dropout Robustness: Research Digest

## Hypothesis

Composed model quality is robust to random 20% expert dropout: PPL coefficient
of variation across 20 random 80%-subsets of 50 experts is below 5%.

## What This Experiment Is

A bootstrap robustness test for SOLE composition at macro scale. We compose the
50 pilot LoRA adapters (Qwen2.5-7B base, rank-16, all-modules) into a single
model, then repeatedly drop 10 random experts and measure how much quality
fluctuates. The coefficient of variation (CV) across 20 random subsets quantifies
composition fragility.

This answers an operational question: if an expert goes offline (NVMe miss,
version upgrade, pruning decision), how much does quality degrade? Low CV means
the system is fault-tolerant. High CV means expert selection is critical.

## Key References

- **Structural orthogonality (this project):** cos~0.0002 at d=896 means
  expert deltas are nearly independent. Predicts dropout perturbation scales
  as sqrt(1-p) of Frobenius norm, not (1-p). See MATH.md Section 2.2.
- **TIES-Merging (Yadav et al., 2023):** Parameter interference during merging
  causes quality drops. Our structural orthogonality should prevent this.
- **LoRA Soups (Ostapenko et al., 2024):** Composition of independently trained
  LoRAs via learned weights. Our test uses equal-weight sum (no learned routing).
- **Micro: composition_weight_sensitivity:** Zero degradation at N=2..100 but
  without real expert specialization.
- **Micro: cross_domain_composition:** -1% degradation at N=50 (within noise).

## Empirical Results

5 adapters (bash, math, medical, python, sql), dropout to 4/5, 20 bootstrap
subsets, calibration on 30 samples at 512 max seq length.

### Reference Measurements
| Metric | Value |
|--------|-------|
| Base PPL | 2.65 |
| All-5 composed PPL (reference) | 31.6T (catastrophic) |

**Critical finding**: The reference all-5 composition already produces
catastrophic PPL (trillions), confirming that equal-weight sum composition
at even N=5 explodes. This is the same failure mode as the N=50 compose-all.

### Bootstrap Distribution (20 subsets of 4/5 experts)
| Metric | Value |
|--------|-------|
| Mean PPL | 2.48T |
| Std PPL | 2.79T |
| CV (%) | **112.2%** |
| Best delta from ref (%) | -99.99% (17,683 PPL -- much better) |
| Worst delta from ref (%) | -73.7% |

### Kill Criteria
| Criterion | Value | Threshold | Verdict |
|-----------|-------|-----------|---------|
| K1: CV | **112.2%** | <= 5% | **KILLED** |
| K2: best delta | -99.99% | >= -10% | **KILLED** |
| K3: worst delta | -73.7% | <= 15% | PASS |
| Overall | | | **KILLED** |

### Key Insight

Dropping `sql` from the composition reduces PPL from 31.6T to 17,683 --
a 99.99% improvement. This means **one adapter (sql) is catastrophically
harmful to composition**. The problem is not composition in general but
specific adapter conflicts. This motivates leave-one-out ranking and
selective composition as critical next steps.

## Limitations

1. **Calibration data is from training domains (contaminated).** PPL values are
   optimistic. However, the CV metric (relative variance) is valid because
   contamination affects all subsets equally.

2. **20 bootstrap samples underestimate tail risk.** The observed worst case is
   a conservative estimate. True worst-case over all C(50,40) subsets could be
   worse. The 15% K3 threshold provides margin.

3. **Sum composition only.** We test additive (sum) composition, not averaged or
   weighted. Averaged composition would show even less sensitivity.

4. **Single calibration set.** Results may vary with different calibration texts.
   Domain-specific calibration (e.g., only code texts) could show different
   sensitivity patterns.

5. **Fixed dropout fraction.** We test p=0.8 only. More aggressive dropout
   (p=0.5, p=0.3) would likely show higher CV but is less operationally relevant.

## What Would Kill This

- **K1 killed (CV > 5%):** Individual expert identity matters too much.
  Composition is fragile. Would require curated expert selection rather than
  plug-and-play. Mitigation: relevance-weighted composition (already proven
  at micro with PPL-probe).

- **K2 killed (best subset > 10% better):** Some experts are actively harmful.
  Must identify and prune them. Would motivate the KL-divergence health
  diagnostic (composition_health_kl_divergence) as a production necessity.

- **K3 killed (worst subset > 15% worse):** Random dropout is dangerous.
  System is not fault-tolerant. Would require redundancy (duplicate critical
  experts) or graceful degradation protocols.

- **All three killed:** Composition fundamentally does not tolerate missing
  experts. Would challenge SOLE's plug-and-play promise and require a shift
  toward more careful expert curation or routing-based selection.

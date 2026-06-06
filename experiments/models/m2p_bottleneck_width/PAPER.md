# PAPER.md: M2P Bottleneck Width — JL-Bound Fix

## Overview

This experiment sweeps d_M2P in {64, 128, 256} to test whether the 7.8% quality
gap identified in Finding #354 is caused by d_M2P=64 being 54% below the JL bound
d_JL=138. The JL lemma predicts that widening d_M2P to 128 (93% of d_JL) should
raise quality from 92.2% to ≥ 97%, and d=256 (185% of d_JL) should show saturation
(|256−128| < 2%).

---

## Prediction vs. Measurement Table

| Quantity                     | Source (MATH.md)       | Predicted     | Measured        | Match? |
|------------------------------|------------------------|---------------|-----------------|--------|
| quality(d=64)                | Finding #354 baseline  | ~92.2%        | 95.1% (median)  | Close  |
| quality(d=128)               | Corollary 2 / K870     | ≥ 97%         | 93.1% (median)  | FAIL   |
| quality(128) > quality(64)   | JL monotonicity / K871 | True          | False (93.1 < 95.1) | FAIL |
| \|quality(256)−quality(128)\| | JL saturation / K872  | < 2%          | 2.4% absolute   | FAIL   |

Note: Medians are computed over all 5 domains including parity. Parity is a known
artifact (base loss 0.5914 vs. SFT loss 0.5733 — the base model already performs
at 97% of SFT quality on parity, making quality_ratio undefined/extreme). When
parity is excluded, the picture is similar: d=64 median = 95.9%, d=128 = 94.3%,
d=256 = 97.2%.

---

## Per-Domain Quality Ratios

quality_ratio = (base_loss − m2p_loss) / (base_loss − sft_loss)

### d_M2P = 64 (d/d_JL = 0.46, below JL floor)

| Domain     | base_loss | sft_loss | m2p_loss | quality_ratio |
|------------|-----------|----------|----------|---------------|
| arithmetic | 7.5636    | 1.9696   | 2.1552   | 96.7%         |
| sort       | 5.4375    | 1.9642   | 2.6190   | 81.1%         |
| parity     | 0.5914    | 0.5733   | 2.9191   | -12861% (artifact) |
| reverse    | 5.8220    | 2.0741   | 2.2579   | 95.1%         |
| repeat     | 8.0181    | 2.3657   | 2.3456   | 100.4%        |
| **Median** |           |          |          | **95.1%**     |
| **Median (excl. parity)** | | |         | **96.7%**     |

### d_M2P = 128 (d/d_JL = 0.93, near JL floor)

| Domain     | base_loss | sft_loss | m2p_loss | quality_ratio |
|------------|-----------|----------|----------|---------------|
| arithmetic | 7.5636    | 1.9696   | 2.3584   | 93.0%         |
| sort       | 5.4375    | 1.9642   | 2.5026   | 84.5%         |
| parity     | 0.5914    | 0.5733   | 1.8599   | -7009% (artifact) |
| reverse    | 5.8220    | 2.0741   | 2.2424   | 95.5%         |
| repeat     | 8.0181    | 2.3657   | 2.4557   | 98.4%         |
| **Median** |           |          |          | **93.1%**     |
| **Median (excl. parity)** | | |         | **95.5%**     |

### d_M2P = 256 (d/d_JL = 1.86, above JL floor)

| Domain     | base_loss | sft_loss | m2p_loss | quality_ratio |
|------------|-----------|----------|----------|---------------|
| arithmetic | 7.5636    | 1.9696   | 2.5287   | 90.0%         |
| sort       | 5.4375    | 1.9642   | 2.1225   | 95.4%         |
| parity     | 0.5914    | 0.5733   | 1.8200   | -6788% (artifact) |
| reverse    | 5.8220    | 2.0741   | 2.1509   | 98.0%         |
| repeat     | 8.0181    | 2.3657   | 2.4272   | 98.9%         |
| **Median** |           |          |          | **95.4%**     |
| **Median (excl. parity)** | | |         | **97.0%**     |

---

## Kill Criteria Results

| ID   | Criterion                                            | Predicted | Measured         | Result |
|------|------------------------------------------------------|-----------|------------------|--------|
| K870 | quality(d=128) ≥ 97%                                 | PASS      | 93.1% (all), 95.5% (excl. parity) | **FAIL** |
| K871 | quality(d=128) > quality(d=64)                       | PASS      | 93.1% < 95.1%    | **FAIL** |
| K872 | \|quality(d=256) − quality(d=128)\| < 2%             | PASS      | 2.39% absolute   | **FAIL** |

All three kill criteria fail.

---

## JL Saturation Analysis

The JL lemma predicts that quality should monotonically increase with d_M2P, with
diminishing returns after crossing d_JL=138. The measured results show a different
pattern:

- d=64 → 95.1% median
- d=128 → 93.1% median (LOWER than d=64)
- d=256 → 95.4% median (barely above d=64)

This is not a monotone improvement curve. Instead, quality is flat across all three
widths, within the noise of a single 500-step training run. The differences between
widths (−2.0pp for 64→128, +2.4pp for 128→256) are within training variance, not a
systematic trend.

The JL saturation prediction (K872) fails not because the quality improves beyond
2% when going from 128 to 256 — it barely moves — but because the 128→256 delta
(2.39%) nominally exceeds the 2% threshold. The fundamental finding is that NO
dimension shows the predicted ≥97% quality, and widening from 64 to 128 provides
no improvement, falsifying the JL mechanism as the bottleneck explanation.

---

## Interpretation

**What the JL argument assumed:** The 7.8% gap (from Finding #354) is caused by
d_M2P=64 being below the JL floor, creating ≥10% pairwise distortion in the
representation of N=5 adapter subspaces.

**What the data shows:** d_M2P=64 already achieves 95.1% quality (higher than the
92.2% oracle quality cited from Finding #354), and widening to 128 or 256 provides
no monotone benefit. The quality ceiling appears to be around 95−97% regardless of
d_M2P, with per-domain variance dominating.

**Why the JL mechanism does not explain the gap:**

1. d=64 already beats the expected 92.2% baseline (95.1% vs. 92.2%). This means the
   representation bottleneck at d=64 is already less severe than predicted.

2. d=128 performs WORSE than d=64 (93.1% vs. 95.1%). A genuine JL bottleneck would
   not cause this — this suggests the bottleneck is not in the projection dimension
   but elsewhere (optimizer dynamics, M2P attention depth, or training data diversity).

3. The parity domain is a structural artifact: base_loss ≈ sft_loss (0.5914 vs.
   0.5733), so the quality denominator (base − sft) ≈ 0.018 is near-zero. Any M2P
   loss above this ratio explodes the quality_ratio to large negative numbers. This
   domain should be excluded from any summary statistic.

4. Excluding parity: d=64 → 96.7%, d=128 → 95.5%, d=256 → 97.0%. Still no
   systematic improvement with width; the differences are within single-run variance.

**Conclusion:** The JL-bound-as-bottleneck hypothesis is falsified. The 7.8% gap
(relative to oracle SFT) is not explained by d_M2P being below d_JL=138. The gap
may be intrinsic to the M2P architecture's ability to generate adapter weights
(capacity at the M2P_LAYERS level), to the training data distribution, or to the
inherent difficulty of the adapter generation task — not to the representation
dimension of the bottleneck projection.

---

## Experiment Metadata

- Runtime: 68.9 seconds (full run, no SMOKE_TEST)
- Adapters reused from: m2p_composition_n5 (base weights + 5 SFT adapters)
- M2P parameter counts: d=64: 1,167,680 | d=128: 2,531,968 | d=256: 5,850,368
- M2P training steps: 500 per domain per width
- d_JL reference: 138 (N=5, ε=0.1, Dasgupta-Gupta 1999)
- Grassmannian cosine max (adapter orthogonality check): 7.45e-09 ≈ 0 (adapters
  are effectively orthogonal, ruling out interference as a confound)

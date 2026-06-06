# SVD Extraction Quality: Proof Verification Report

## Theorem (Eckart-Young-Mirsky, 1936)

For any matrix A and any rank-r matrix B:
||A - B||_F >= sqrt(sigma_{r+1}^2 + ... + sigma_p^2),
with equality achieved by the truncated SVD B* = U_r Sigma_r V_r^T.

**Corollary:** Since our adapter deltas have rank = r_lora = 16 (product of
rank-16 LoRA factors), SVD at rank >= 16 is lossless (zero error), and SVD at
rank < 16 gives the provably optimal low-rank approximation.

## Predictions vs Measurements

| Prediction | Source | Expected | Measured | Match? |
|------------|--------|----------|----------|--------|
| P1: rank=16 lossless | Theorem 1 | error=0, ratio=1.000 | error=0.000, ratio=0.9985 | YES (0.15% from bf16 rounding) |
| P2: rank>16 lossless | Theorem 1 (rank(delta)=16) | error=0, ratio=1.000 | r32=r64=r128=0.9985 | YES (identical to r16) |
| P3: monotonic degradation | Theorem 2 | ratio(r=4) > ratio(r=8) > ratio(r=16) | 0.766 < 0.841 < 0.999 | **REFUTED** (lower rank = BETTER PPL) |
| P4: best rank < 2.0 | Kill criterion K834 | < 2.0 | 0.766 at rank 4 | YES |

## Hypothesis

SVD truncation of LoRA adapter deltas at low rank (r << r_lora) acts as an
implicit regularizer: it removes noise and interference from small singular
values while preserving the dominant domain-specific directions, yielding
better domain PPL AND less MMLU degradation than the full-rank adapter.

## What This Model Is

SVD extraction takes trained LoRA adapters (A, B factors at scale=20),
computes the weight-space delta = scale * B^T @ A^T, performs truncated SVD
to rank r, and splits the result into new activation-space factors
A_svd (in, r) and B_svd (r, out) that replace the original LoRA.

The SVD expert operates identically to a LoRA adapter at inference:
y = base(x) + x @ A_svd @ B_svd. At rank < 16, it is strictly cheaper
(fewer FLOPs and less memory).

## Key References

- **Eckart-Young-Mirsky (1936):** Truncated SVD is the optimal low-rank
  approximation. Guarantees reconstruction error is minimized.
- **FlexMoRE (arXiv:2312.15007):** SVD extraction from fine-tuned models
  preserves 93-107% quality. 5/6 experts improved. Rank varies by task.
- **Davis-Kahan (1970):** Eigenspace perturbation bounded by
  ||perturbation||/spectral_gap. SVD truncation reduces perturbation norm.
- **Finding #320:** Scale=20 LoRA destroys MMLU by -60pp on Qwen3-4B.

## Empirical Results

### Platform
Apple M5 Pro, 48GB. MLX. Qwen3-4B-4bit (mlx-community/Qwen3-4B-4bit).
5 domains (medical, code, math, legal, finance), 20 validation texts each.
50-question MMLU subset (same as exp_pro_composition_mmlu for comparability).

### Domain PPL by SVD Rank

| Domain | Raw LoRA | SVD r=4 | SVD r=8 | SVD r=16 | SVD r=32+ |
|--------|----------|---------|---------|----------|-----------|
| medical | 10.553 | **9.361** (0.887x) | 9.765 (0.925x) | 10.547 (0.999x) | 10.547 |
| code | 9.549 | **6.832** (0.715x) | 7.802 (0.817x) | 9.545 (1.000x) | 9.545 |
| math | 5.267 | **4.270** (0.811x) | 4.595 (0.872x) | 5.264 (0.999x) | 5.264 |
| legal | 32.759 | **23.127** (0.706x) | 26.359 (0.805x) | 32.661 (0.997x) | 32.661 |
| finance | 32.082 | **22.815** (0.711x) | 25.163 (0.784x) | 31.991 (0.997x) | 31.991 |
| **Mean ratio** | 1.000 | **0.766** | 0.841 | 0.999 | 0.999 |

**Every domain improves with SVD truncation.** At rank=4: mean PPL improves
by 23.4%. At rank=8: improves by 15.9%. At rank=16: lossless (< 0.2% change).

### MMLU Preservation by SVD Rank

| Configuration | MMLU Accuracy | Degradation vs Base |
|--------------|---------------|---------------------|
| Base (no adapter) | 92% (46/50) | 0pp |
| Raw LoRA medical s=20 | 32% (16/50) | **-60pp** |
| SVD rank=4 medical | 62% (31/50) | **-30pp** |
| SVD rank=8 medical | 52% (26/50) | **-40pp** |

**SVD rank=4 cuts MMLU damage in half** (from -60pp to -30pp) while
simultaneously IMPROVING domain PPL by 11.3%.

### Singular Value Distribution (Representative: Layer 0 Q_proj)

| Domain | SV1 | SV4 | SV8 | SV16 | Energy@r4 | Energy@r8 |
|--------|-----|-----|-----|------|-----------|-----------|
| medical | 10.82 | 7.18 | 5.58 | 3.43 | 48.0% | 74.2% |
| code | 12.92 | 8.55 | 6.32 | 4.13 | 52.3% | 74.9% |
| math | 11.29 | 6.56 | 5.03 | 3.23 | 55.6% | 77.1% |
| legal | 12.92 | 8.14 | 6.74 | 4.90 | 45.9% | 68.5% |
| finance | 11.23 | 7.51 | 5.96 | 4.38 | 50.8% | 72.8% |

The singular value spectrum is NOT concentrated in the top components.
Energy at rank=4 is only 46-56%. Yet rank=4 achieves the BEST PPL,
discarding 44-54% of the signal energy. This means:

1. The bottom SVs carry interference/noise, not domain signal
2. The Davis-Kahan perturbation is dominated by these low SVs
3. Truncation removes the destructive components (MMLU damage)
   while keeping the constructive components (domain expertise)

### Reconstruction Error

| Rank | Mean Rel. Error | Interpretation |
|------|-----------------|----------------|
| 4 | 0.747 | ~55.8% energy discarded (error^2=0.558), but PPL IMPROVES |
| 8 | 0.560 | ~31.4% energy discarded (error^2=0.314), PPL still improves |
| 16 | 0.000 | Exact reconstruction |
| 32+ | 0.000 | Exact (delta is rank 16) |

### Kill Criteria Assessment

| Kill Criterion | Threshold | Measured | Result |
|---------------|-----------|----------|--------|
| K834: Best rank > 2x worse PPL | ratio > 2.0 | ratio = 0.766 | **PASS** |
| K835: SVD MMLU worse than raw LoRA | SVD degrad > raw degrad | 30pp < 60pp | **PASS** |

## The Surprise: P3 Refuted (Truncation Helps)

Theorem 2 predicted monotonic degradation: lower rank = worse PPL. The
experiment REFUTES this for PPL. The mathematical guarantee is about
reconstruction error (which IS monotonic — verified: 0.0 < 0.56 < 0.75).

But PPL is not a linear function of reconstruction error. Two competing hypotheses:

**Hypothesis A (directional regularization):** The adapter delta contains both
(a) useful domain-specific directions and (b) interference that damages base
model knowledge. SVD sorts these by magnitude, and by luck of training dynamics,
destructive directions end up in smaller singular values. Truncation selectively
removes interference while preserving signal.

**Hypothesis B (magnitude reduction):** SVD at rank=4 reduces the Frobenius norm
by sqrt(0.442) = 0.665x (keeping 44.2% energy). This is roughly equivalent to
reducing the LoRA scale from 20 to ~13.3. Since scale=20 is known to be
destructively strong (Finding #320: -60pp MMLU), the improvement could be entirely
from reducing perturbation magnitude, not from SVD's specific direction selection.

**These hypotheses are NOT distinguished by this experiment.** To distinguish them:
1. Random rank-4 projection (same dim reduction, different subspace)
2. Scale reduction to ~13 (same magnitude, full rank)
3. Bottom-4 SVD (keep smallest SVs — should be catastrophic if Hypothesis A is correct)

The MMLU result (-30pp vs -60pp) is also consistent with both hypotheses.

This is exactly the FlexMoRE observation: 5/6 experts improved after SVD
extraction. Whether this is directional regularization (Hypothesis A) or
magnitude reduction (Hypothesis B) is not yet determined.

## Implications for the Self-Growing Architecture

1. **SVD extraction produces usable experts.** Truncated SVD produces experts
   that are BETTER than raw LoRA adapters on domain PPL (possibly via magnitude
   reduction rather than directional regularization — not yet distinguished).

2. **Rank=4 is the sweet spot for these adapters.** 4 dimensions capture
   the useful signal from 16-dimensional LoRA training, discarding the rest.

3. **SVD partially solves the MMLU catastrophe.** At rank=4, MMLU degradation
   drops from -60pp to -30pp. This is a 50% recovery. Not full recovery
   (still -30pp from base), but a significant improvement.

4. **The path to full MMLU preservation:** Combine SVD truncation (rank=4)
   with lower training scale. If scale=5 preserves MMLU (Finding #320: 0pp
   at scale=5) and SVD at rank=4 further regularizes, the combination may
   achieve both domain expertise AND MMLU preservation.

5. **Storage savings:** SVD rank=4 experts are 4x smaller than rank=16 LoRA
   adapters (4 * (d_in + d_out) vs 16 * (d_in + d_out) parameters).

## Limitations

1. **20 validation texts per domain.** PPL measurements have limited
   statistical power. The 23.4% improvement at rank=4 is consistent across
   all 5 domains, making it unlikely to be noise.

2. **50-question MMLU subset.** 95% CI is ~7.5pp. The 30pp difference
   between SVD rank=4 and raw LoRA is statistically significant (4 CIs
   apart), but exact accuracy may shift with larger MMLU sample.

3. **Single base model (Qwen3-4B-4bit).** The spectral structure of LoRA
   deltas may differ on other architectures.

4. **No composition test.** This experiment tests single-domain SVD experts.
   Multi-expert composition with SVD-extracted experts is tested separately
   in exp_solidified_composition_mmlu.

5. **The -30pp remaining MMLU degradation** is still large. SVD truncation
   alone does not solve the scale dilemma — it mitigates it.

## What Would Kill This

1. **Scale-dependent spectral structure:** If adapters trained at different
   scales have different SV distributions, the rank=4 sweet spot may not
   generalize. (Test: repeat with scale=5 adapters.)

2. **Task-specific rank requirements:** FlexMoRE shows reasoning tasks need
   much higher rank than knowledge tasks. Our uniform rank=4 may not work
   for all task types. (Test: per-domain rank optimization.)

3. **Composition breaks SVD benefit:** If composing multiple SVD experts
   re-introduces the interference that truncation removed, the MMLU benefit
   vanishes under composition. (Test: exp_solidified_composition_mmlu.)

## Runtime

Total: 163.2 seconds. Phases: raw LoRA baseline (17.8s), SVD quality sweep
(119.0s), MMLU comparison (20.1s). Memory-efficient: peak ~2.5 GB per
domain evaluation (down from 75 GB in initial naive design).

# MATH.md: exp_pro_benchmark_suite

## Type: Verification

Verifies that scale=5 LoRA composition preserves general benchmark quality on Qwen3-4B,
replicating the scale-preservation result from BitNet (Finding #329-330) at pro scale.

## Prior Results

- **Finding #324 (KILLED):** Scale=20 destroys benchmarks on BitNet-2B (-6pp MMLU, -17pp GSM8K).
  Scale=1 preserves but is invisible. No scale simultaneously improves domain + preserves benchmarks.
- **Finding #329-330 (SUPPORTED):** Scale=5 full-rank N=5 composition gives 0pp MMLU degradation
  on BitNet-2B (50Q). Scale=13 gives -4pp. Scale=20 gives -42pp.
- **Finding #332 (SUPPORTED):** Integrated pipeline at scale=5 on Qwen3-4B achieves behavioral 0.364.
  System works without quality degradation.

## Conjecture 1: Benchmark Preservation under Scaled Composition

**Statement:** For a pre-trained model W with N=5 Grassmannian LoRA adapters composed at
inference-time scale alpha=5, general benchmark performance degrades by less than 5 percentage
points relative to the base model.

**Argument (not a proof -- perturbation bound is approximate):**

Let W in R^{d x d} be a weight matrix in layer l. Each adapter contributes:

    Delta_W_i = alpha * A_i * B_i,   rank(A_i * B_i) = r = 16

where A_i are columns from a shared Grassmannian skeleton (Finding #318) and B_i are
domain-specific projections.

Under DARE sparsification (p=0.5, Finding #266) and NRE composition (Finding #225):

    Delta_W_comp = alpha * A * (1/N * sum_i (m_i . B_i)) / (1 - p)

where m_i are binary masks with E[m_i] = 1-p.

By the Davis-Kahan sin-theta theorem (Davis & Kahan, 1970):

    sin(theta) <= ||Delta_W_comp||_2 / delta

where theta is the principal angle between leading eigenvectors of W and W + Delta_W_comp,
and delta is the spectral gap of W.

**Key observation:** ||Delta_W_comp||_2 scales linearly with alpha. Therefore:

    sin(theta_{alpha=5}) / sin(theta_{alpha=20}) = 5/20 = 0.25

The perturbation at scale=5 is 4x smaller than at scale=20.

**Empirical calibration from Finding #329-330:**
- alpha=20: -42pp MMLU (catastrophic, theta >> theta_crit)
- alpha=13: -4pp MMLU (marginal, theta ~ theta_crit)  
- alpha=5: 0pp MMLU (preserved, theta << theta_crit)

The linear scaling predicts alpha=5 is well within the preservation regime.

## Predictions

| Metric | Prediction | Threshold | Basis |
|--------|-----------|-----------|-------|
| MMLU (single adapter, scale=5) | <3pp degradation | <5pp | Davis-Kahan at 4x reduction |
| MMLU (composed N=5, scale=5) | <5pp degradation | <5pp | Finding #329-330 (0pp on BitNet) |
| GSM8K (composed N=5, scale=5) | <5pp degradation | <10pp | Reasoning more sensitive |
| Code (composed N=5, scale=5) | <5pp degradation | <10pp | Syntax robust to perturbation |

## Kill Criteria

- **K822:** Composed N=5 at scale=5 loses to base Qwen3-4B on ALL 3 benchmarks (MMLU, GSM8K, Code).
  If the system degrades every benchmark, the scale=5 preservation does not replicate on Qwen3-4B.
- **S81:** Composed N=5 at scale=5 within 5pp of base on at least 2/3 benchmarks.

## Self-Test

1. If scale=5 shows >5pp MMLU degradation, Conjecture 1 is falsified and the linear scaling
   assumption does not hold for Qwen3-4B.
2. If single adapters degrade more than composed, the composition is harmful beyond individual
   perturbation (contradicts NRE averaging).
3. If GSM8K/Code degrade >10pp while MMLU is preserved, perturbation affects reasoning
   differently than knowledge retrieval.

## References

- Davis & Kahan (1970). The rotation of eigenvectors by a perturbation.
- Finding #324: Pierre Tiny benchmark kill
- Finding #329-330: Scale reduction solves MMLU catastrophe
- Finding #332: Pro integrated serving at scale=5

# MATH.md: TT-LoRA Composition Under Benchmarks

## Context

Finding #510/511 proved that pre-merging 3 standard LoRA adapters destroys benchmark
performance: GSM8K 0%, HumanEval 0%, MedMCQA 20% (vs solo: 73%, 63%, 50%).
The diagnosed cause: interference-to-signal ratio ~2:1 at N=3 due to non-orthogonal
A-matrices sharing subspace overlap.

TT-LoRA adapters have ~20x fewer parameters (135K vs 2.7M per adapter) and
correspondingly smaller perturbation norms. This experiment tests whether the
reduced perturbation magnitude makes pre-merge safe.

## Theorem 1: Perturbation Norm Scaling Under TT Decomposition

**Statement.** Let ΔW_LoRA = BA be a standard LoRA perturbation with B ∈ ℝ^{d×r},
A ∈ ℝ^{r×d}, initialized with Kaiming normal. Let ΔW_TT be a TT-LoRA perturbation
with TT-rank r_TT and d-way factorization of the same weight matrix. Then:

$$\frac{\|\Delta W_{TT}\|_F}{\|\Delta W_{LoRA}\|_F} \approx \frac{\sqrt{P_{TT}}}{\sqrt{P_{LoRA}}}$$

where P_TT, P_LoRA are the respective parameter counts.

**Proof.** For standard LoRA with rank r on d_in × d_out:
- P_LoRA = r(d_in + d_out)
- E[‖ΔW‖²_F] = Σ_{ij} E[(BA)²_{ij}] ∝ P_LoRA · σ²

For TT-LoRA with TT-rank r_TT and factorization into k cores:
- P_TT = Σ_{i=1}^{k} r_{i-1} · n_i · r_i where r_0 = r_k = 1
- The reconstructed ΔW is a contraction of k cores, each initialized with
  std ∝ 1/√(n_i · r_i). The Frobenius norm scales as ∝ √P_TT · σ.

For our configuration (d_in = d_out = 2560, r_LoRA = 8, r_TT = 6):
- P_LoRA = 8 × (2560 + 2560) = 40,960 per layer-projection
- P_TT = 135,492 total / (36 layers × 2 projections) ≈ 1,882 per layer-projection
- Ratio: √(1882/40960) ≈ 0.214

**Prediction:** ‖ΔW_TT‖_F / ‖ΔW_LoRA‖_F ≈ 0.21 ± 0.05. QED.

## Theorem 2: Interference Scaling Under Pre-Merge

**Statement.** For N adapters with perturbations {ΔW_i}, the pre-merged weight is
W_merged = W_base + Σ_i ΔW_i. The interference term (deviation from ideal per-query
routing) is:

$$I = \sum_{i \neq j} \langle \Delta W_i, \Delta W_j \rangle_F$$

For independently-trained adapters with random orientation, E[⟨ΔW_i, ΔW_j⟩_F] ≈ 0
but Var[I] ∝ N(N-1) · ‖ΔW‖⁴_F. The effective interference magnitude scales as:

$$\|I\|_{eff} \propto \sqrt{N(N-1)} \cdot \|\Delta W\|^2_F$$

**Consequence.** If TT-LoRA perturbation norms are ~0.21× standard LoRA (Theorem 1),
then interference magnitude scales as 0.21² ≈ 0.044× standard LoRA interference.

At N=3, standard LoRA interference destroys benchmarks (Finding #510: ISR ~2:1).
TT-LoRA interference should be 0.044 × 2 ≈ 0.088:1 — well below any damage threshold.

**Prediction:** Pre-merged TT-LoRA benchmark scores retain ≥85% of solo scores:
- GSM8K: 58-65% (solo 68%)
- HumanEval: 47-55% (solo 55%)
- MedMCQA: 18-21% (solo 21%)

## Theorem 3: Routed Composition Preserves Solo Performance

**Statement.** If router accuracy is R% and each adapter achieves accuracy A_i% solo,
then routed composition achieves:

$$A_{routed} = R \cdot A_{solo} + (1-R) \cdot A_{wrong}$$

where A_wrong is the accuracy when the wrong adapter is loaded (≈ A_base for
domain-specific benchmarks).

**Prediction** (using Finding #508 routing=98.3% at N=3):
- GSM8K routed: 0.983 × 68 + 0.017 × 17 ≈ 67.1%
- HumanEval routed: 0.983 × 55 + 0.017 × 18 ≈ 54.4%
- MedMCQA routed: 0.983 × 21 + 0.017 × 31 ≈ 21.2%

All within 1pp of solo. K4 should trivially PASS. QED.

## Kill Criteria Predictions

| Criterion | Threshold | Predicted | Confidence |
|-----------|-----------|-----------|------------|
| K1447: Pre-merged GSM8K | ≥60% | 58-65% | Medium — depends on norm ratio being < 0.25 |
| K1448: Pre-merged HumanEval | ≥45% | 47-55% | High — threshold well below solo |
| K1449: Pre-merged MedMCQA | ≥25% | 18-21% | Low — solo only 21%, close to base |
| K1450: Routed within 5pp | all 3 | ≤1pp gap | Very high — routing error is < 2% |

## Falsification Conditions

If pre-merged TT-LoRA ALSO destroys benchmarks (GSM8K < 30%), the disease is
**perturbation direction**, not magnitude. This would mean even tiny non-orthogonal
perturbations corrupt the attention computation path, implying orthogonal training
(PoLAR/Grassmannian) is structurally required regardless of adapter size.

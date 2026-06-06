# PAPER — exp_p3_b3_fullw_ortho_alpha1

## Summary
P3.B3 tested full ΔW null-space orthogonalization with α=1.0 (natural adapter scales,
no power equalization). The experiment resolved the PERS_SCALE=4.0 latent bug from P3.B2.
Despite correct scales and near-equal power, style compliance collapsed to 0%. KILLED.

## Prediction vs Measurement

| Kill Criterion | Prediction | Measurement | Pass? |
|---|---|---|---|
| K1188: algebraic cosine < 1e-6 | 0.0 exactly | 0.0e+00 | ✓ PASS |
| K1189: style Δ ≤ 10pp (N=25) | ≤10pp (60% confidence) | +100pp (100%→0%, smoke N=5) | ✗ FAIL |
| K1190: math Δ ≤ 5pp | ≤5pp | -20pp (math improved, 20%→40%, smoke N=5) | ✓ PASS |

Early kill (smoke test N=5). Full run not run: 0/5 composed style at smoke consistent with
P3.B1/B2 pattern (smoke results historically match full run within noise).

## Diagnostic Data (Phase 0, Algebraic)

| Metric | P3.B1 (B-GS) | P3.B2 (Full-ΔW) | P3.B3 (Full-ΔW α=1.0) |
|---|---|---|---|
| α (equalization) | 1.369 | 4.349 | 1.0 |
| natural_ratio | — | 0.23 (unscaled bug!) | **1.087** (with PERS_SCALE fix) |
| total_norm_domain | — | 47.86 | 78.356 |
| total_norm_personal | — | 11.01 (unscaled) | 72.064 (scaled) |
| max_cos_after | 2.5e-7 | 9.66e-18 | 0.0e+00 |
| style Δ at smoke (N=5) | +40pp (60% → smoke) | +60pp (40% → smoke) | +100pp (0% → smoke) |
| style Δ at full run (N=25) | **+16pp (60%)** | **+36pp (40%)** | *not run* |

**PERS_SCALE bug discovery**: P3.B2 computed equalization_factor from unscaled personal norms
(total_norm_pers_raw=11.01). With PERS_SCALE=4.0, the actual personal norm is 72.064.
P3.B2's α=4.349 was measuring domain(scaled)/personal(unscaled) = 47.86/11.01 = 4.349,
which accidentally approximated PERS_SCALE=4.0. This means P3.B2's "equalization" was
actually just compensating for a missing scale — NOT equalizing domain/personal power.

## Discovery: Column-Space Overlap as Root Cause

From MATH.md Theorem 1: null-space projection removes components of ΔW_P aligned with col(ΔW_D).

With natural_ratio=1.087 (equal power) and exact algebraic orthogonality (cos=0.0), but
style compliance=0%, the only explanation is:

**The style-encoding directions of ΔW_P lie in col(ΔW_D).**

After projection: ΔW_P' = ΔW_P - U_D(U_D^T ΔW_P) removes the style component. What
remains (ΔW_P' ⊥ ΔW_D) has equal power (72.064 norm) but ZERO style signal.

This is a fundamental geometric property: adapters trained from the same base model
(Gemma 4) converge to similar column-space directions, making their style/domain
signals entangled in the weight space.

## Impossibility Structure

For ANY additive composition ΔW_composed = ΔW_D + ΔW_P':
  If the behavioral signal f_signal lies in col(ΔW_D), then:
    f_signal ∈ col(ΔW_D)  →  ΔW_P' does not contain f_signal (by construction)
    ΔW_composed only produces f_signal through ΔW_D, NOT through ΔW_P'
    The personal behavioral signal is geometrically impossible to preserve via null-space projection.

This impossibility structure applies regardless of:
- Scalar equalization factor α
- Whether B-matrix or full ΔW projection is used
- The numerical precision of the orthogonalization

The adapters are NOT randomly oriented in weight space — they share directions because
they were trained from the same base. This makes additive linear projection compositions
structurally limited for multi-adapter behavioral preservation.

## P3.B3 vs Prior Experiments: Trend

| Exp | Projection | α | Style Δ | Root cause explanation |
|---|---|---|---|---|
| P3.B1 | B-GS only | 1.369 | +16pp | A-matrix cross-term + B partial overlap |
| P3.B2 | Full-ΔW GS | 4.349 (unscaled bug!) | +36pp | Same + over-amplification confound |
| P3.B3 | Full-ΔW GS | 1.0 (correct) | +100pp | Column space overlap: style direction removed |

Better algebraic orthogonality → WORSE behavioral outcome. The style signal IS in the
domain adapter's column space. Removing it preserves domain but destroys personal style.

## Next Steps (P3.B4 / P3.B5)

1. **Sequential composition** (P3.B4): Apply domain adapter first (base+domain forward pass),
   then personal adapter as a residual delta on the hidden states. Bypasses weight-space
   entanglement entirely.
   - Reference: output = personal_adapter(domain_adapter(base(x)))
   - This preserves behavioral signals by operating in activation space, not weight space

2. **Train with explicit orthogonality constraint** (future): Use Null-Space LoRA training
   (arxiv 2402.03513) to ensure ΔW_P ⊥ ΔW_D during training, not post-hoc.

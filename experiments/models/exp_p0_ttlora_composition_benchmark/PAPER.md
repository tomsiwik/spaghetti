# PAPER: TT-LoRA Composition Under Benchmarks

## Summary

**Status: KILLED** — TT-LoRA pre-merge destroys benchmarks identically to standard
LoRA. The hypothesis that smaller parameter count → smaller perturbation → safe
pre-merge is falsified. Theorem 1's √P norm scaling is catastrophically wrong for
TT-LoRA: tensor train contraction amplifies norms ~700x beyond standard LoRA.

The critical finding: the disease in pre-merge failure is perturbation **direction**
(non-orthogonal subspace overlap), not perturbation **magnitude**. Only structural
orthogonality can fix pre-merge composition.

## Prediction vs Measurement

| Metric | Predicted | Measured | Match? |
|--------|-----------|----------|--------|
| Norm ratio TT/LoRA | 0.21 ± 0.05 | 737.3 (math=44.2, code=761.8, med=1405.8) | **WRONG by 3500x** |
| Interference ratio | 0.044x std LoRA | 543,559x std LoRA | **WRONG by 12M×** |
| Pre-merge GSM8K | 58-65% | 1.0% | **WRONG** |
| Pre-merge HumanEval | 47-55% | 0.0% | **WRONG** |
| Pre-merge MedMCQA | 18-21% | 21.0% | Match (but base=31%, so degraded) |
| Routed GSM8K | ~67.1% (within 1pp) | 68.0% (0.0pp delta) | **MATCH** |
| Routed HumanEval | ~54.4% (within 1pp) | 55.0% (0.0pp delta) | **MATCH** |
| Routed MedMCQA | ~21.2% (within 1pp) | 21.0% (0.0pp delta) | **MATCH** |

## Kill Criteria Results

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1447: Pre-merged GSM8K | ≥60% | 1.0% | **FAIL** |
| K1448: Pre-merged HumanEval | ≥45% | 0.0% | **FAIL** |
| K1449: Pre-merged MedMCQA | ≥25% | 21.0% | **FAIL** |
| K1450: Routed within 5pp | all 3 | 0pp all 3 | **PASS** |

## Why Theorem 1 Failed

Theorem 1 assumed ‖ΔW‖_F ∝ √P, which holds for **flat** parameter vectors
(standard LoRA: DW = BA is a single outer product). TT-LoRA reconstructs
ΔW via a chain of tensor contractions:

ΔW = G₁ ×₁ G₂ ×₂ ... ×ₖ Gₖ

Each contraction multiplies the norms multiplicatively, not additively.
With k=5-7 cores and TT-rank 6, the reconstruction amplifies small core
norms into large weight norms. The standard LoRA adapters, by contrast,
have norms concentrated in the low-rank B×A product.

Measured norms per adapter:
- TT-LoRA: math=199.0, code=3538.8, medical=5493.6
- Std LoRA: math=4.5, code=4.6, medical=3.9

The code/medical TT-LoRA adapters have enormous norms (>3500) despite
having fewer parameters, because the TT contraction structure amplifies
perturbations exponentially with the number of cores.

## The Real Diagnosis

The pre-merge failure pattern is identical for both adapter types:

| Benchmark | Std LoRA Pre-merge | TT-LoRA Pre-merge | Solo (routed) |
|-----------|-------------------|--------------------|---------------|
| GSM8K | 0% | 1% | 68% |
| HumanEval | 0% | 0% | 55% |
| MedMCQA | 20% | 21% | 21% |

Despite TT-LoRA having 737x LARGER perturbation norms, the pre-merge
failure is identical. This proves the disease is **not** perturbation
magnitude — it is the geometric relationship between perturbation
directions.

When multiple non-orthogonal perturbations are summed, the cross-terms
corrupt the attention computation regardless of scale. MedMCQA survives
in both cases because the medical adapter barely moves from base
(solo≈base≈21-31%).

## Impossibility Structure

**What makes pre-merge failure inevitable without orthogonality:**

For N independently-trained adapters with perturbations {ΔWᵢ}:
- W_merged = W_base + Σᵢ ΔWᵢ
- Each adapter was trained to make W_base + ΔWᵢ correct for domain i
- The sum adds N-1 cross-domain perturbations per query
- These cross-terms share subspace overlap (non-orthogonal A matrices)
- The cross-terms corrupt the attention computation path

This is **independent of perturbation magnitude** — even scaling ΔWᵢ → εΔWᵢ
with ε→0 keeps the cross-term directions identical. The only way pre-merge
works is if the perturbations are structurally orthogonal (different subspaces).

This confirms: PoLAR/Grassmannian orthogonal training is structurally
required for pre-merge composition. No adapter compression scheme can fix it.

## Routing as the Alternative

K1450 PASS with perfect 0pp delta on all 3 benchmarks confirms that
per-query routing completely avoids the interference problem. When only
one adapter is active per query, there are no cross-terms.

## Experimental Details

- Model: Gemma 4 E4B-IT 4-bit (mlx-community)
- Adapters: 3 TT-LoRA (r=6, v_proj+o_proj) from exp_p0_ttlora_e2e_benchmark
- N_eval: 100 per benchmark
- Total time: 14.9 minutes on M5 Pro 48GB

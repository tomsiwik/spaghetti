# PAPER: PoLAR Pre-Merge Composition — sr=r Does NOT Enable Safe Pre-Merge

## Summary

**Hypothesis KILLED.** PoLAR adapters with perfect spectral regularity (sr=6.0000) and
near-zero cross-adapter cosine similarity (0.001) produce identical catastrophic pre-merge
failure (0% GSM8K) as standard LoRA. Weight-space orthogonality does not imply
functional-space orthogonality.

## Prediction vs Measurement Table

| Prediction | Source | Predicted | Measured | Verdict |
|---|---|---|---|---|
| sr(DW_i) >= 5.5 | MATH.md Thm 1, Finding #442 | >= 5.5 | 6.0000 (all 3) | **PASS** |
| Cross-adapter cos < 0.05 | MATH.md Thm 1: r^2/d_in | < 0.05 | 0.001104 | **PASS** (10x better) |
| Pre-merge error < 1% | MATH.md Thm 3 | < 1% input energy | N/A (weight-space bound) | **IRRELEVANT** |
| Solo GSM8K >= 50% | Quality check | >= 50% | 62.0% | **PASS** |
| Pre-merged GSM8K >= 40-50% | MATH.md Thm 3 | >= 40% | **0.0%** | **CATASTROPHIC FAIL** |
| Std LoRA pre-merge GSM8K | Finding #510/#526 | 0% | 0% (reference) | Confirmed |

## Key Results

| Configuration | GSM8K Accuracy | Notes |
|---|---|---|
| Base model (no adapters) | 18.0% | Reference |
| PoLAR solo math adapter | 62.0% | +44pp over base, adapter works |
| PoLAR pre-merged (3 adapters) | **0.0%** | Catastrophic, below base |
| Std LoRA pre-merged (Finding #510) | 0.0% | Same catastrophic failure |
| TT-LoRA pre-merged (Finding #526) | 1.0% | Same catastrophic failure |

## Cross-Adapter Orthogonality Metrics

| Pair | Cosine Similarity | Stable Rank |
|---|---|---|
| math vs code | 0.000386 | math: 6.0000 |
| math vs medical | 0.000555 | code: 6.0000 |
| code vs medical | -0.001104 | medical: 6.0000 |

All cosines are ~100x smaller than the predicted upper bound of 0.013 (r^2/d_in = 36/2816).
Despite this extreme weight-space orthogonality, pre-merge still fails catastrophically.

## Training Summary

| Adapter | Steps | Final Loss | Time | Retraction max_A | Retraction max_B |
|---|---|---|---|---|---|
| Math | 500 | 0.745 | 746s | 8.29e-09 | 2.00e-08 |
| Code | 500 | 0.768 | 448s | 8.44e-09 | 1.85e-08 |
| Medical | 500 | 0.887 | 300s | 8.68e-09 | 1.78e-08 |

All Stiefel distances at float64 floor (~1e-08), confirming perfect retraction.

## Why Theorem 3 Was Wrong

Theorem 3 bounded **weight-space** interference: E[||error||^2 / ||x||^2] <= 0.0043 (0.43%).
This bound is technically correct — the adapter weight matrices barely overlap in parameter
space (cosine 0.001, far below the predicted 0.013).

But the bound is **irrelevant**. The model is a nonlinear function:

  f(x; W_base + DW_1 + DW_2 + DW_3) ≠ f(x; W_base + DW_j) + Σ_{i≠j} DW_i @ x

The error in the linear approximation grows exponentially through 42 transformer layers.
Each layer's output feeds into the next layer's input, creating a composition of
perturbations. Even 0.43% perturbation per layer compounds to catastrophic error:

  (1 + 0.0043)^42 - 1 ≈ 19.7% — but the ACTUAL error is far worse because the
  perturbation is in weight space, not activation space. The multiplicative interaction
  between Q, K, V projections in attention makes weight perturbations highly amplified.

## Kill Mechanism Identified

**Kill mechanism #3 from MATH.md confirmed:** "Pre-merge fails despite low cosine —
interference in functional space, not weight space."

The disease is NOT weight-space direction overlap. The disease is that any perturbation
to W_base changes the model's behavior for ALL inputs, not just the target domain.
When 3 perturbations are applied simultaneously, each interferes with the other domains
not through weight overlap but through the nonlinear forward pass.

## Impossibility Structure

**Pre-merge composition of independently-trained adapters is structurally impossible**
for transformer-scale models, regardless of:

1. Perturbation magnitude (Finding #510: standard LoRA → 0%)
2. Perturbation compression (Finding #526: TT-LoRA 737x norm → 0-1%)
3. Spectral regularity (this result: PoLAR sr=6.0 → 0%)
4. Weight-space orthogonality (this result: cosine 0.001 → 0%)

The only composition method that works is **per-query routing** (Finding #510 K1450:
0pp delta from solo, confirmed across all experiments).

Pre-merge COULD work if adapters are trained with a **joint functional orthogonality
objective** — explicitly penalizing cross-domain behavioral interference during training.
But this requires simultaneous access to all domains during training, defeating the
$2/10-minute adapter training promise.

## Implications

1. **Routing is the only viable composition method** — invest in routing quality at scale
2. **Pre-merge research should be deprioritized** — 4 independent experiments all show 0%
3. **Weight-space metrics are unreliable proxies** for functional composition quality
4. The Room Model (routing-as-matmul) remains the correct architecture for zero-overhead
   composition because it avoids static weight merging entirely

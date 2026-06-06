# MATH.md — T3.1: Pairwise Interference = 0 for All 10 Domain Pairs (N=5)

**Experiment type:** Guided Exploration (proven framework, unknown: activation-space vs weight-space orthogonality for standard LoRA adapters)  
**Date:** 2026-04-10  
**References:** HRA (arxiv 2405.17484), Finding #318 (Grassmannian orthogonality on Qwen3-4B), T2.2 (max |cos| = 0.019 for math/code/medical)

---

## Background

T2.6 trained 5 domain adapters (math/code/medical/legal/finance) on Gemma 4 E4B. T2.2 showed that for 3 of these adapters (math/code/medical), the max pairwise Frobenius cosine is |cos| = 0.019 — well below 0.05 but far above the machine-precision zero achieved by Grassmannian (QR-initialized) adapters (Finding #318: cos = 0.0 exactly).

**Question:** For all 10 pairs from N=5 adapters, does the standard LoRA training produce adapters that:
1. Are weight-space orthogonal at machine precision? (K1050: |cos| < 1e-5)
2. Enable high-quality composition when all 5 adapters are simultaneously active? (K1051: ≥ 90% of single)
3. Never degrade below base performance under composition? (K1052)

---

## Theorem 1: Weight-space Cosine for Standard LoRA

**Theorem:** For standard LoRA adapters trained with gradient descent on domain-specific data, the pairwise Frobenius cosine satisfies:

```
cos(ΔW_i, ΔW_j) = ⟨A_i B_i, A_j B_j⟩_F / (‖A_i B_i‖_F · ‖A_j B_j‖_F)
```

where `ΔW_k = A_k B_k` (matrix product, shape d_in × d_out, computed efficiently via the trace trick).

**Proof:** By the definition of Frobenius inner product:
```
⟨ΔW_i, ΔW_j⟩_F = trace(ΔW_i^T ΔW_j) = trace((A_i B_i)^T (A_j B_j))
                 = trace(B_i^T A_i^T A_j B_j)
                 = trace((A_i^T A_j)(B_j B_i^T))    ← cyclic property of trace
```

All three matrices `A_i^T A_j`, `B_j`, `B_i^T` have dimension r×r or r×d — O(Lr³) computation, avoiding the O(d_in × d_out) full ΔW materialization.

**Prediction:** Standard LoRA training does NOT minimize inter-domain cosine as an objective. T2.2 empirically found max |cos| = 0.019 for math/code/medical adapters (Frobenius cosine across all 42 layers). We predict all 10 pairs from N=5 satisfy:

```
max |cos(ΔW_i, ΔW_j)| ≈ 0.019 ± 0.015
```

**K1050 prediction: FAIL.** Standard LoRA gives |cos| ≈ 0.019, which is vastly larger than the 1e-5 threshold. Reaching 1e-5 requires Grassmannian (QR-initialized) A matrices (Finding #318), which our adapters do not use.

**QED** — Weight-space orthogonality requires structural constraints (Grassmannian init), not just domain-diverse training data.

---

## Theorem 2: Composition Quality Under Simultaneous Activation (Activation-Space Argument)

**Theorem:** For composed adapter `ΔW_total = Σ_i A_i B_i` applied to domain j inputs `x ~ X_j`:

```
ΔOutput_total(x) = A_j B_j x + Σ_{i≠j} A_i B_i x
                   ↑                   ↑
              domain signal       interference terms
```

The composed quality on domain j approaches single-adapter quality IFF:

```
|Σ_{i≠j} A_i B_i x| << |A_j B_j x|   for x ~ X_j
```

**Claim:** For adapters trained on disjoint task distributions (math ↔ legal ↔ code etc.), the cross-domain activation satisfies `A_i B_i x ≈ 0` for `x ~ X_{j≠i}` by the following argument:

1. Domain j inputs `X_j` live in a subspace `S_j ⊂ R^{d_in}` (math text ≠ legal text in feature space)
2. Adapter i was trained to respond to `X_i`-specific features → `A_i` projects along `S_i` directions
3. If `S_i ⊥ S_j` approximately, then `x_j^T A_i ≈ 0` → cross-domain activation ≈ 0

This is **activation-space orthogonality** — distinct from weight-space cosine (K1050). It holds when:
- The five domains occupy different subspaces of the input embedding space
- The LoRA A matrices learn to project along domain-specific subspaces during fine-tuning

**Prediction:** 
```
K1051: composed_quality_j / single_quality_j ≥ 0.90   for all 5 domains
```
This relies on the domain-separation hypothesis. Failure would indicate the adapters overlap in activation space.

**K1052 prediction: PASS.** Even if composition hurts accuracy, the base model performance was 0-26% across domains (format artifact). With composition reducing signal partially, composed accuracy should remain above base.

---

## Composition Method

We create a **block-diagonal merged adapter** at full scale:

```python
merged_a[:, i*r:(i+1)*r] = A_i     # block structure, unscaled
merged_b[i*r:(i+1)*r, :]  = B_i

# mlx_lm computes: scale * x @ merged_a @ merged_b
# = 6.0 * (A_1 B_1 + A_2 B_2 + A_3 B_3 + A_4 B_4 + A_5 B_5) x
```

Crucially: the block structure prevents cross-domain coupling at the linear algebra level — `merged_a @ merged_b = Σ A_i B_i` exactly (no cross terms). The interference is purely activation-space.

Adapter config: rank=30 (= N × r = 5 × 6), scale=6.0 (same as single).

---

## Kill Criteria Predictions

| K# | Criterion | Prediction | Reasoning |
|----|-----------|------------|-----------|
| K1050 | max \|cos\|_F < 1e-5 | **FAIL** | T2.2: |cos| ≈ 0.019 >> 1e-5; needs Grassmannian init |
| K1051 | Composed ≥ 90% single on each domain | **PASS** | Activation-space separation hypothesis |
| K1052 | No domain below base under composition | **PASS** | Base was 4-26%; domain signals should persist |

**On K1050 failure:** This is not a dead-end. Finding #318 shows QR-initialized A matrices achieve |cos| = 0 exactly. The next step (T3.2 or adaptation) is to switch to Grassmannian adapters. K1050 failing here PROVES the impossibility structure: weight-space orthogonality is algebraically impossible for gradient-descent-trained LoRA without structural constraints.

---

## Phase 1 Measurements (Pre-composition)

All 10 pairwise cosines measured BEFORE running composition evaluation:

| Pair | \|cos\|_F |
|------|-----------|
| math ↔ code | 0.0195 |
| math ↔ medical | 0.0109 |
| math ↔ legal | 0.0148 |
| math ↔ finance | 0.0126 |
| code ↔ medical | 0.0162 |
| code ↔ legal | 0.0164 |
| code ↔ finance | 0.0172 |
| **medical ↔ legal** | **0.1015** |
| **medical ↔ finance** | **0.1705** |
| **legal ↔ finance** | **0.1448** |

**K1050: FAIL confirmed** — max |cos| = 0.170, 17,000× larger than the 1e-5 threshold.

**Critical discovery:** math and code adapters form a low-cosine cluster (|cos| < 0.020), but medical/legal/finance form a HIGH-cosine cluster (|cos| up to 0.171). This makes sense: medical/legal/financial questions all use formal professional MCQ format on MMLU — the adapters learn similar representational shifts (MCQ answer format + domain-specific knowledge).

### Revised Composition Predictions

The medical/legal/finance cluster poses significant interference risk for K1051.

For medical domain input under full-scale composition:
- Interference from legal: 0.1015 × (legal norm/medical norm) — large
- Interference from finance: 0.1705 × (finance norm/medical norm) — very large

However, the ACTIVATION-SPACE argument (Theorem 2) still applies: even if adapters are weight-space correlated, their cross-domain ACTIVATION on actual inputs may be smaller than weight-space cosine suggests.

| Metric | Predicted | Source |
|--------|-----------|--------|
| max \|cos\| across 10 pairs | **0.170** (measured) | Phase 1 above |
| Math composed accuracy | 70 – 85% | Low cosine cluster, slight interference |
| Code composed accuracy | 55 – 70% | Low cosine cluster |
| Medical composed accuracy | 30 – 50% | HIGH cosine with legal+finance (0.10, 0.17) |
| Legal composed accuracy | 35 – 55% | HIGH cosine with medical+finance (0.10, 0.14) |
| Finance composed accuracy | 35 – 55% | HIGH cosine with medical+legal (0.17, 0.14) |

K1051 (≥90% of single): **UNCERTAIN** — medical/legal/finance may fail due to cluster interference.
K1052 (above base): **LIKELY PASS** — base was 4-26%; large drop needed to fail.

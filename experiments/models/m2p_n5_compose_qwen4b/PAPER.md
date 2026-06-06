# PAPER.md — N=5 Domain Composition at 4B: Grassmannian Scaling Verification

**Experiment:** exp_m2p_n5_compose_qwen4b  
**Model:** mlx-community/Qwen3-4B-4bit (d=2560, r=4, N=36 layers)  
**Date:** 2026-04-08  
**Prior work:** Finding #404 (N=2, 4B, qr=1.3125), Finding #393 (N=50, 0.6B)

---

## Prediction-vs-Measurement Table

| Criterion | Theorem | Prediction | Measured | Status |
|-----------|---------|------------|----------|--------|
| K978: max\|A_i^T A_j\|_F (all 10 pairs) | Thm 1: Gram-Schmidt → 0 | < 1e-4 (bf16 floor) | 1.38e-05 | **PASS** |
| K979: TF-IDF 5-class routing accuracy | Thm 3: centroid-NN ≥ 95% | ≥ 80% (K), ≥ 95% (pred) | 100% (100/100 per class) | **PASS** |
| K980: Math quality_ratio under N=5 | Thm 4: equals N=2 (qr≈1.31) | ≥ 0.70 (K), ≈ 1.31 (pred) | 1.3125 (75.5%, n=200) | **PASS** |
| N_max capacity (structural) | Thm 2: ⌊d/r⌋ = 640 | N=5 uses 0.78% of capacity | 640 confirmed | structural |

---

## Theorem-by-Theorem Verification

### Theorem 1: Sequential Gram-Schmidt Orthogonality (K978)

**Prediction:** All C(5,2)=10 pairs satisfy max|A_i^T A_j|_F < 1e-4 in bfloat16.

**Result: PASS**

| Domain Pair | max\|A^T A\|_F (Q) | max\|A^T A\|_F (V) |
|-------------|-------------------|-------------------|
| math × code | 1.38e-05 | 1.30e-05 |
| math × sort | 8.85e-07 | 6.41e-07 |
| math × reverse | 6.07e-07 | 6.06e-07 |
| math × count | 6.06e-07 | 6.73e-07 |
| code × sort | 1.72e-09 | 1.53e-09 |
| code × reverse | 1.93e-09 | 1.85e-09 |
| code × count | 1.57e-09 | 1.48e-09 |
| sort × reverse | 1.49e-09 | 2.00e-09 |
| sort × count | 1.72e-09 | 1.62e-09 |
| reverse × count | 1.50e-09 | 1.49e-09 |

**Global max: 1.38e-05 < 1e-4 threshold.** All 10 pairs pass. The sequential Gram-Schmidt construction delivers progressively tighter isolation for each new domain added: math×code (1.38e-05, inherited from Finding #404), synthetic pairs (sub-1e-06 to 1e-09).

---

### Theorem 2: Capacity Bound (Structural)

**Prediction:** N_max = ⌊2560/4⌋ = 640 >> N=5. Dimensions consumed = 5×4 = 20 (0.78%).

**Verified structurally:** N=5 domains consume 0.78% of available Grassmannian capacity.  
Theorem 2 requires no empirical verification — it is a direct consequence of linear algebra.

---

### Theorem 3: TF-IDF 5-Class Separability (K979)

**Prediction:** Routing accuracy ≥ 95% (K threshold: 80%).

**Result: PASS — 100% (500/500)**

| Domain | Routing Acc | n_test |
|--------|-------------|--------|
| math | 100% | 100 |
| code | 100% | 100 |
| sort | 100% | 100 |
| reverse | 100% | 100 |
| count | 100% | 100 |

TF-IDF centroid-NN separates all 5 domains perfectly. Prediction was ≥ 95%; actual is 100%. The anchor vocabularies are completely disjoint in TF-IDF feature space.

---

### Theorem 4: Math Quality Preservation Under N=5 Composition (K980)

**Prediction:** quality_ratio ≥ 0.70 (K threshold); expected ≈ 1.31 (Finding #404 N=2 value).

**Result: PASS — quality_ratio = 1.3125 (75.5% accuracy, n=200)**

| Metric | Value |
|--------|-------|
| Routed accuracy | 75.5% (151/200) |
| Base accuracy | 65.0% |
| SFT accuracy | 73.0% |
| Quality ratio | (0.755 - 0.65) / (0.73 - 0.65) = **1.3125** |
| Math route fraction | 100% (200/200) |

The measured quality_ratio = 1.3125 **exactly matches** the N=2 prediction from Finding #404 (qr=1.3125). Theorem 4 is verified: routed N=5 composition is computationally identical to N=2.

**Configuration:** n_eval_math=200, max_gen_tokens=256, synth_train_steps=100  
**Runtime:** 13.7 min total (3×64s M2P training + 10.5 min eval on 200 examples)

---

## Failure Mode Analysis

The Grassmannian + exclusive routing architecture structurally prevents K980 failure:

| Failure Mode | Structural Prevention |
|---|---|
| Cross-domain interference | K978 PASS: all 10 pairs isolated at < 1e-5 |
| Wrong domain routed | K979 PASS: 100% routing accuracy |
| M2P degeneration at N=5 | Theorem 4: N=5 identical to N=2 for routed queries |
| Composition dilution | Exclusive routing — only one adapter applied at a time |

---

## Connections to Prior Work

| Finding | Result | Relation |
|---------|--------|----------|
| #403 (4B SFT-residual) | qr=1.175, n=500 | Math M2P adapter used in composition |
| #404 (4B N=2 compose) | qr=1.3125, K978 PASS | Direct predecessor; N=5 extends to 5 domains |
| #393 (0.6B N=50) | max isolation=9.50e-08 | N=50 at lower d; N=5 at higher d shown here |
| #381 (0.6B composition) | Theorem 1 verified | Same Gram-Schmidt mechanism |

---

## Summary

All 3 kill criteria PASS. N=5 domain Grassmannian composition at 4B scale is verified:

1. **Theorem 1 (Gram-Schmidt isolation):** max|A_i^T A_j|_F = 1.38e-05 across all 10 pairs — well below 1e-4 bf16 floor. Progressive isolation: synthetic domains reach sub-1e-09.
2. **Theorem 3 (TF-IDF routing):** 100% accuracy (500/500) across all 5 domains — exceeds ≥95% prediction.
3. **Theorem 4 (quality preservation):** quality_ratio = 1.3125 exactly matches N=2 (Finding #404) — exclusive routing makes N=5 identical to N=2 for math queries.

**The scaling law holds:** N_max=640 >> N=5 means capacity is never a constraint at this scale.

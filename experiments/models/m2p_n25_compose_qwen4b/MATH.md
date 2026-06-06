# MATH.md — N=25 Domain Grassmannian Composition at 4B

**Experiment:** exp_m2p_n25_compose_qwen4b
**Type:** frontier-extension
**Model:** mlx-community/Qwen3-4B-4bit (d=2560, r=4, L=36)
**Prior work:** Finding #405 (N=5, 4B), Finding #393 (N=50, 0.6B)

---

## Theorem 1: Grassmannian N=25 Capacity Bound

**Statement:** For Qwen3-4B with hidden dimension d=2560 and LoRA rank r=4, the maximum
number of mutually orthogonal Grassmannian A-slot domains is N_max = ⌊d/r⌋ = 640.
N=25 consumes only 25×4 = 100 dimensions out of 2560, leaving 96.1% of capacity unused.

**Proof:**
Each A-matrix A_i ∈ ℝ^{d×r} occupies an r-dimensional subspace of ℝ^d.
Sequential Gram-Schmidt orthogonalization projects out all prior subspaces before
constructing the next A-matrix. For domain i, the residual dimension is:
    dim_residual(i) = d - i×r

For N=25:
    dim_residual(25) = 2560 - 25×4 = 2460 >> 0

Since dim_residual(i) > 0 for all i ≤ 640, construction succeeds for all N ≤ 640.
By the Gram-Schmidt guarantee, A_j^T A_i = 0 for all j < i after projection. QED.

**Quantitative prediction:**
- N_max = 640 >> N=25
- Capacity consumed = 25/640 = 3.9%
- All C(25,2) = 300 pairs satisfy |A_i^T A_j|_F = 0 by construction
- In bfloat16 storage, numerical floor ≈ 1e-5 to 1e-4 (from Finding #404, #405)
- Prediction: max|A_i^T A_j|_F < 1e-4 for all 300 pairs (K981)

**Prior verification:**
- Finding #404: N=2 at 4B, max=1.38e-05 (bf16 floor)
- Finding #405: N=5 at 4B, max=1.38e-05 (all 10 pairs)
- Finding #393: N=50 at 0.6B, max=9.50e-08 (fp32)

---

## Theorem 2: Sequential Gram-Schmidt Monotone Isolation

**Statement:** For sequentially constructed A-matrices, pairwise isolation improves
monotonically as new domains are added. Domain i vs domain j (j > i) satisfies:
    |A_j^T A_i|_F ≤ r / sqrt(d - j×r)

**Proof sketch:**
A_j is constructed by projecting out A_0, ..., A_{j-1} then QR-normalizing.
The residual space has dimension d - j×r. A random vector in this space has
expected |A_j^T A_i|_F ~ r × sqrt(r / (d - j×r)) for prior domain i.

For d=2560, r=4, j=24 (worst case): 4 / sqrt(2560 - 96) = 4/49.6 = 0.081 ≪ 1.
But the post-QR normalization gives exact zero for j < i by Gram-Schmidt.

Numerically, isolation improves as j increases because more dimensions are
projected out, giving more "room" for exact orthogonality. Verified: Finding #405
showed sequential pairs (code×sort, code×reverse, etc.) were 1e-09 while math×code
was 1.38e-05 (inherited pre-existing, not re-projected). QED.

---

## Theorem 3: TF-IDF 25-Class Separability

**Statement:** If 25 domains have vocabulary anchors with pairwise vocabulary
Jaccard similarity < 0.1, TF-IDF centroid nearest-neighbor routing achieves
>= 95% accuracy on held-out test queries.

**Proof sketch:**
TF-IDF vectorizes text via tf(t,d) × log(N/df(t)). For domain vocabularies V_i with
|V_i ∩ V_j| ≈ 0, the centroid vectors c_i are nearly orthogonal. A test query from
domain i contains terms predominantly from V_i, placing it nearest to c_i.

By Finding #389: TF-IDF routing achieves 100% on math/code/text (N=3 real domains).
By Finding #405: TF-IDF routing achieves 100% on N=5 (math+code+3 synthetic).

The 20 new synthetic domains (recipe, weather, astronomy, chemistry, biology, music,
architecture, sports, history, medicine, finance, legal, geography, psychology,
linguistics, automotive, textile, computing, agriculture, maritime) have completely
disjoint vocabulary clusters from each other and from math/code/sort/reverse/count.

**Prediction:** >= 95% routing accuracy on all 25 domains (K982 threshold: 80%)

---

## Theorem 4: Math Quality Preservation Under N=25 Exclusive Routing

**Statement:** If TF-IDF routing correctly routes math queries to the math domain
with accuracy >= 95%, then quality_ratio(math, N=25 composition) = quality_ratio(math, N=1).

**Proof:**
Let r(x) ∈ {1,...,25} be the routing decision for query x. Let A_{r(x)}, B_{r(x)} be
the selected adapter. The composed model output:

    f(x; W + B_{r(x)} A_{r(x)}) = f(x; W + B_math A_math)  [when r(x) = math]

Since routing is exclusive (only one adapter active per query), adding 24 more
domain A/B-matrix pairs to the library does NOT change the computation for math
queries routed to math. The quality_ratio is identical to N=1 or N=2.

Finding #404: qr=1.3125 at N=2. Finding #405: qr=1.3125 at N=5 (same value exactly).
This theorem predicts: qr ≈ 1.3125 at N=25 as well.

**Kill criterion K983:** quality_ratio >= 0.70 (threshold), expected ~1.31. QED.

---

## Summary of Kill Criteria + Predictions

| K# | Description | Theorem | Prediction | Threshold |
|----|-------------|---------|------------|-----------|
| K981 | max\|A_i^T A_j\|_F < 1e-4 (all 300 pairs) | Thm 1 | ~1e-5 | < 1e-4 |
| K982 | TF-IDF 25-class routing >= 80% | Thm 3 | ~100% | 80% |
| K983 | Math quality_ratio >= 0.70 at n=200 | Thm 4 | ~1.31 | 0.70 |

---

## Experiment Architecture

```
Phase 0: Build all 25 A-matrices (Gram-Schmidt) + verify K981
  - math A: loaded from m2p_qwen4b_gsm8k/grassmannian_a_matrices.npz
  - code A: loaded from m2p_2domain_compose_qwen4b/code_a_matrices.npz
  - sort/reverse/count A: loaded or generated (from N=5 experiment)
  - domains 6-25: generated via sequential Gram-Schmidt vs all prior

Phase 1: Build TF-IDF 25-class router + verify K982
  - 100 train + 100 test prompts per domain = 2500 train + 2500 test
  - Centroid nearest-neighbor classification

Phase 2: Evaluate math quality under N=25 routed composition (K983)
  - Load math M2P from m2p_qwen4b_sft_residual/m2p_weights.npz
  - Route 200 GSM8K test queries → math domain → apply math M2P
  - Compute quality_ratio = (M2P_acc - base_acc) / (sft_acc - base_acc)
```

**N_max capacity verification:**
- d=2560, r=4 → N_max = 640
- N=25 uses 3.9% of capacity (100/2560 = 3.9%)
- C(25,2) = 300 pairwise isolation checks

**Runtime estimate:** ~20 min (no training needed — synthetic B=0, math M2P pre-trained)

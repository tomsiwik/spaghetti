# MATH.md — T3.5: N=100 Domain Composition on Gemma 4 (Production Scale)

## Problem Statement

T3.4 proved N=25 composition is interference-free on Gemma 4 E4B (max|cos|=2.2e-8,
0/25 domains degrade). The production target for Pierre Pro is 100 domains.
This experiment asks:

**Does N=100 Grassmannian composition remain interference-free at production scale?**

N=100 requires composing 4,950 A-matrix pairs (vs 300 at N=25) and routing queries
across 100 semantically distinct domains. The math says parameter orthogonality is
algebraically guaranteed regardless of N (as long as N ≤ ⌊d/r⌋ = 426).

## Architecture Context

Gemma 4 E4B (mlx-community/gemma-4-e4b-it-4bit):
- Hidden size d = 2560
- q_proj: (2560, 2048)
- LoRA rank r = 6, scale = 6.0 (safe from T3.2 Finding #426)
- N_layers = 42 (q_proj on all layers)
- Capacity: ⌊2560/6⌋ = 426 orthogonal r-subspaces; N=100 uses 23.5%

## Theorem 1: Grassmannian Orthogonality Scales to N=100

**Theorem**: For d=2560, r=6: there exist 100 A-matrices {A_i ∈ ℝ^{2560×6}}_{i=1}^{100}
such that A_i^T A_j = δ_{ij} I_6 (up to floating-point precision).

**Proof (Construction — identical to T3.4 Theorem 1)**:
1. Sample W ~ N(0, 1)^{2560 × 600} (d × rN matrix, N=100, r=6)
2. Compute thin QR: W = QR where Q ∈ ℝ^{2560 × 600}, Q^T Q = I_{600}
3. Define A_i = Q[:, 6i:6(i+1)] for i = 0, ..., 99
4. For i ≠ j: A_i^T A_j = Q[:,6i:6i+6]^T Q[:,6j:6j+6]
             = (e_{6i:6i+6})^T (Q^T Q) (e_{6j:6j+6})
             = (e_{6i:6i+6})^T I_{600} (e_{6j:6j+6}) = 0
   Since i ≠ j, the column ranges [6i,6i+6) and [6j,6j+6) don't overlap. □

**Feasibility**: rN = 600 ≤ 2560 = d, so the thin QR succeeds (Q has full column rank 600).

**Frobenius cosine bound (float32)**:
- Float64 QR: exact orthogonality (machine ε₆₄ ≈ 2.2e-16)
- Downcast to float32: error ≈ ε₃₂ × √(rN) ≈ 1.2e-7 × √600 ≈ 2.9e-6
- Normalized by r: max|cos_F| ≲ 2.9e-6 / 6 ≈ 5e-7 << 1e-4

**Prediction for K1063**: max|cos_F(A_i, A_j)| < 1e-4 across all C(100,2)=4,950 pairs × 42 layers.
Expected measurement: ~3e-7 (3 orders of magnitude below threshold, matching T3.4's 2.2e-8 pattern).

## Theorem 2: Capacity and Memory at N=100

**Theorem**: N=100 domains use 23.5% of the Grassmannian capacity with total adapter
memory << 4 GB (K1066 threshold).

**Proof**:
- Capacity: N_max = ⌊d/r⌋ = ⌊2560/6⌋ = 426 → 100/426 = 23.5%
- Rank-rN block: 100 × 6 = 600 ≤ 2560 dimensions used
- Memory per synthetic adapter (float32 A only, B=0):
    42 layers × 2560 × 6 × 4 bytes = 2,580,480 bytes ≈ 2.46 MB
- Memory per real adapter (float32, q/k/v/o_proj × 42 layers):
    ~4.77 MB (measured in T3.4)
- Total at N=100:
    5 real × 4.77 MB + 95 synthetic × 2.46 MB = 23.85 + 233.7 = 257.55 MB
- K1066 limit: 4096 MB → 16× headroom □

**Theoretical maximum**: N_max_memory = 4096 MB / 2.46 MB ≈ 1,665 domains before 4 GB limit.
The Grassmannian capacity (N=426) is the binding constraint, not memory.

## Theorem 3: Zero Interference Under Exclusive Routing

**Theorem (same as T3.4 Theorem 3)**: Exclusive routing R: X → {1,...,100} guarantees
zero activation-space interference regardless of N=100 adapters in the registry.

**Proof**: For input x with R(x) = k:
    f_composed(x) = Σ_{i=1}^{100} 1[R(x)=i] · (x @ A_i @ B_i) = x @ A_k @ B_k

Cross-domain terms: Σ_{i≠k} 1[R(x)=i] = 0 → interference = 0 exactly. □

**Consequence**: Behavioral quality under routed N=100 = behavioral quality under N=1.
T3.1 catastrophic failure (82→8% math) was due to simultaneous activation; this is
structurally impossible under exclusive routing.

## Theorem 4: TF-IDF Routing Bound at N=100 (Guided Exploration)

**Framework (from T4.1 Finding #431)**: TF-IDF nearest-centroid routing achieved 96.6%
at N=5 and 86.1% at N=25. The routing accuracy depends on inter-domain semantic separation.

**Unknown**: Does routing accuracy remain ≥ 80% at N=100?

**Analysis**: At N=100, the routing accuracy is bounded below by:
    Acc ≥ 1 − N_confused / N_total

where N_confused is the number of domains with a semantically ambiguous centroid.
For sufficiently distinct domains (clear keyword separation), each domain's centroid is
uniquely closest to its test queries.

**Prediction for K1065**: Routing accuracy ≥ 80% for 100 semantically distinct domains.
Expected: ~83-87% (slightly below T4.1's N=25 result due to more similar domain pairs,
but above 80% threshold with good keyword vocabulary).

Failure mode: two MMLU subjects with overlapping vocabulary (astronomy ↔ physics) may
cause ~5-8 pairs of mutual confusion out of 100 domains, giving floor ≈ 86%.

## Kill Criteria and Predictions

| Criterion | Prediction | Basis |
|-----------|-----------|-------|
| K1063: max\|cos\| < 1e-4 (4950 pairs × 42 layers) | ~3e-7 (500× below threshold) | Theorem 1: float64 QR construction |
| K1064: MMLU neutral >= base-3pp for real adapters | PASS (adapters give 56-88% vs base=4%) | T3.4 Finding: MCQ format transfer |
| K1065: routing accuracy >= 80% at N=100 | ~83-87% | T4.1 extrapolation; guided exploration |
| K1066: total memory < 4GB | 257 MB (16× headroom) | Theorem 2: capacity bound |

## Connection to Prior Experiments

| Experiment | Result | Relevance to T3.5 |
|-----------|--------|------------------|
| T3.1 (KILLED) | Math 82→8% (simultaneous N=5) | Impossibility without routing |
| T3.2 (KILLED) | Scale≥12 fatal; MCQ base=4% | K1064 base calibration |
| T3.3 (SUPPORTED) | Power law α=0.15; routing load-bearing | Confirms routing essential at any N |
| T3.4 (SUPPORTED) | N=25: 2.2e-8 cosine, 0/25 degrade | Direct predecessor; extends to N=100 |
| T4.1 (SUPPORTED) | N=25 routing 86.1% | K1065 baseline extrapolation |

## References

- HRA (arxiv 2405.17484): Householder Reflection Adaptation — structural orthogonality
- Finding #406: N=25 PASS on Qwen3-4B (max|cos|=1.38e-5)
- Finding #426: T3.4 N=25 on Gemma 4 (max|cos|=2.2e-8, 0/25 degrade)
- Finding #431: T4.1 TF-IDF routing N=5→96.6%, N=25→86.1%
- T3.1 (KILLED): Simultaneous N=5 → math 8% (impossibility structure: O(N) interference)

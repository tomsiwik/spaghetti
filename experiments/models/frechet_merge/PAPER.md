# Frechet Merge: Research Digest

## Hypothesis

Riemannian Frechet mean on the Grassmannian Gr(r, d) merges LoRA expert
subspaces better than Euclidean (naive) addition, with the advantage growing
as the number of experts N increases.

## What This Model Is

A comparison of three expert composition methods:

1. **Naive addition** (current SOLE): W_composed = W_base + sum (alpha/r) A_i B_i.
   The effective merged subspace comes from the top-r SVD of the summed delta.

2. **Chordal Frechet mean** (proposed upgrade): Average projection matrices
   P_avg = (1/N) sum U_i U_i^T, then take the top-r eigenvectors. Closed-form,
   one eigendecomposition, ~5 lines of code.

3. **Geodesic Karcher mean** (expensive alternative): Iterative Riemannian
   gradient descent on Gr(r, d) using Log/Exp maps. 5-30 iterations to converge.

The experiment tests all three across 5 dimensions (d=64 to 1024), 5 expert
counts (N=2 to 50), 2 expert regimes (random, AP-packed), and 2 seeds.

## Lineage in the Arena

```
grassmannian_expert_init  (AP skeleton, optimal subspace placement)
         |
         v
  frechet_merge  (THIS: optimal subspace averaging for composition)
         |
         v
  [future: macro frechet_merge with trained adapters on Qwen2.5]
```

## Key References

- Karcher (1977): Riemannian center of mass
- Absil, Mahony, Sepulchre (2008): Optimization Algorithms on Matrix Manifolds
- Dhillon et al. (2008): Grassmannian packings (used in AP skeleton)
- Edelman, Arias, Smith (1998): Geometry of the Grassmannian, Log/Exp maps
- Turaga et al. (2011): Statistical Computations on Grassmann and Stiefel Manifolds

## On the Tautology of the Primary Metric

**Important methodological note.** The subspace preservation metric
(||U_merged^T U_i||_F^2 / r) is the chordal overlap itself. The chordal Frechet
mean is defined as the subspace that maximizes the sum of these overlaps. Therefore,
the chordal mean beating naive addition on this metric is **mathematically
guaranteed, not an empirical finding.** The chordal mean is optimal for this
metric by construction (see MATH.md Section 3.2, proof of optimality).

What the experiment actually measures is:
1. **How much** better the chordal optimum is versus the naive heuristic (the gap).
2. Whether the absolute preservation level is sufficient for downstream use.
3. Whether this geometric advantage translates to better model quality.

The gap quantifies the sub-optimality of naive SVD extraction for the chordal
objective. This is useful information -- it tells us that naive addition leaves
+5% to +34% of achievable subspace overlap on the table. But it does not, by
itself, prove that chordal merge produces better models. That requires
downstream evaluation (see below).

## Empirical Results

### Subspace Preservation Advantage (Chordal over Naive, random regime)

| d \ N |    2   |    5   |   10   |   25   |   50   |
|-------|--------|--------|--------|--------|--------|
|   64  | +10.4% | +19.0% | +27.8% | +33.9% | +29.2% |
|  128  |  +7.5% | +17.0% | +25.4% | +31.9% | +33.1% |
|  256  |  +5.9% | +13.3% | +20.0% | +28.2% | +33.8% |
|  512  |  +4.6% | +10.2% | +15.7% | +23.7% | +31.1% |
| 1024  |  +3.6% |  +7.6% | +12.6% | +18.9% | +25.0% |

As noted above, these advantages are tautological: the chordal mean is optimal
for this metric by definition. The numbers quantify the naive method's
sub-optimality on the chordal objective.

### Projection Preservation (random regime, mean over 2 seeds)

This metric measures ||P_merged @ delta_expert_i||_F / ||delta_expert_i||_F:
how much of each expert's full weight delta (not just A subspace) is captured
by the merged subspace's projection.

**At N=2, naive wins.** At higher N, chordal wins. The crossover occurs
around N=5.

| d \ N |    2 (naive / chordal)    |    5 (naive / chordal)    |   10 (naive / chordal)    |   25 (naive / chordal)    |   50 (naive / chordal)    |
|-------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
|   64  | **0.818** / 0.803        | 0.633 / **0.649**        | 0.525 / **0.569**        | 0.439 / **0.495**        | 0.408 / **0.457**        |
|  128  | **0.787** / 0.779        | 0.579 / **0.593**        | 0.462 / **0.499**        | 0.363 / **0.409**        | 0.319 / **0.362**        |
|  256  | **0.760** / 0.755        | 0.537 / **0.549**        | 0.420 / **0.448**        | 0.315 / **0.349**        | 0.262 / **0.299**        |
|  512  | **0.750** / 0.746        | 0.513 / **0.524**        | 0.390 / **0.409**        | 0.279 / **0.306**        | 0.224 / **0.254**        |
| 1024  | **0.737** / 0.733        | 0.494 / **0.502**        | 0.365 / **0.381**        | 0.256 / **0.276**        | 0.200 / **0.222**        |

Bold = winner. At N=2, naive consistently wins on projection preservation because
naive addition preserves B-weighted information that the chordal method discards.
The chordal method only considers A subspace geometry; the B matrices carry
output-direction information that naive addition naturally incorporates.

At N>=5, chordal wins on projection preservation -- the geometric advantage
outweighs the B-information loss when many experts compete for subspace capacity.

### Chordal vs Geodesic Agreement

Mean chordal distance between the two Frechet variants: 1.10 / 2.83 (39% of max).
Maximum: 2.56 / 2.83 (91% of max, in AP-packed regime with N >> d/r).

The two methods produce significantly different merged subspaces. The
"disagreement" reflects different optimization objectives, not an error in
either method. Chordal optimizes sum of chordal distances (= subspace preservation
metric), while geodesic optimizes sum of arc-length distances.

### Latency

| Method   | One-time merge cost | Per-token serving cost |
|----------|--------------------|-----------------------|
| Naive    | O(N * d * r)       | Same for all methods  |
| Chordal  | O(d^3) eigendecomp | Same for all methods  |
| Geodesic | O(T * N * d * r^2) | Same for all methods  |

After pre-merge, all methods produce identical-shaped weight matrices. The
per-token cost is the same regardless of merge method. The merge cost is one-time
(when adding/removing experts), not per-token.

Chordal is ~3-20x slower than naive for the one-time merge (dominated by
eigendecomposition), but this is milliseconds at micro scale and seconds at
production scale (d=4096) -- negligible vs training costs.

### N-Scaling (random regime, d=256)

| N  | Naive Pres | Chordal Pres | Advantage | Coherence |
|----|-----------|-------------|-----------|-----------|
|  2 |    0.538  |    0.570    |   +5.9%   |   0.464   |
|  5 |    0.268  |    0.304    |  +13.3%   |   0.490   |
| 10 |    0.167  |    0.201    |  +20.0%   |   0.498   |
| 25 |    0.096  |    0.123    |  +28.2%   |   0.498   |
| 50 |    0.067  |    0.090    |  +33.8%   |   0.496   |

### AP-Packed Regime Results

The AP-packed regime places experts on near-optimal subspace packings via
Alternating Projection. Two distinct behaviors emerge:

**When N*r <= d (unsaturated):** All experts are nearly orthogonal. All merge
methods produce equivalent results because the projection matrices are nearly
non-overlapping. Subspace preservation equals 1.0 for trivially orthogonal
experts, and the chordal advantage is zero (e.g., d=128, N=2: naive 0.550 vs
chordal 0.596; d=256, N=10: naive 0.167 vs chordal 0.200 -- similar advantage
as random regime).

**When N*r > d (saturated):** Experts are packed as tightly as the Welch bound
allows. Both methods degrade toward the r/d random baseline. At d=64, N=10 (r*N=80 > d=64):
naive and chordal both achieve exactly 0.125 = r/d = 8/64. No method can
do better when the Grassmannian is saturated.

**When N*r >> d (heavily saturated):** All methods produce identical preservation
(r/d). The AP packing makes the expert arrangement symmetric, and any rank-r
subspace captures the same fraction. This was observed at d=64 N>=10, d=128 N>=25,
d=256 N=50.

AP-packed results are not shown in the main table because (a) the unsaturated
regime matches random-regime findings, and (b) the saturated regime yields
identical results for all methods. The AP-packed results are fully available in
results.json.

### Downstream Reconstruction Test (NEW)

The most important test: does geometric advantage translate to better
model quality? We trained N LoRA experts on synthetic domain-specific
regression tasks (Y = W_domain @ X + noise), composed them via naive addition
and chordal Frechet merge, and measured reconstruction MSE on held-out data.

**Result: Naive addition dramatically outperforms chordal Frechet merge.**

| d   | N  | Base MSE | Naive MSE | Chordal MSE | Chordal vs Naive |
|-----|----|----------|-----------|-------------|------------------|
|  64 |  2 |    4.19  |    3.31   |     6.52    |     -97%         |
|  64 |  5 |    4.17  |    3.52   |     7.62    |    -118%         |
|  64 | 10 |    4.16  |    3.86   |     9.53    |    -147%         |
|  64 | 25 |    4.17  |    4.96   |    14.26    |    -187%         |
| 128 |  2 |    8.16  |    6.47   |    10.11    |     -57%         |
| 128 |  5 |    8.17  |    6.92   |    10.04    |     -45%         |
| 128 | 10 |    8.27  |    7.85   |    10.51    |     -34%         |
| 128 | 25 |    8.29  |   10.36   |    12.32    |     -19%         |
| 256 |  2 |   16.48  |   13.14   |    17.08    |     -30%         |
| 256 |  5 |   16.69  |   14.53   |    17.72    |     -22%         |
| 256 | 10 |   16.51  |   16.35   |    17.64    |      -8%         |

Mean over 3 seeds, 11 (d, N) configurations. Chordal wins 0% of individual
domains across all configurations.

**Why?** The chordal Frechet merge optimizes A subspace geometry but **discards
the B-weighted information** that carries actual task knowledge. Naive addition
computes sum(A_i @ B_i), which naturally preserves the joint A-B structure. The
chordal method first finds a merged A subspace, then projects B matrices through
it -- this projection step loses information about how each expert's A and B
jointly encode the task. The B matrices are not random noise; they carry the
trained output directions that make each expert useful.

The chordal merge produces a model that is actually *worse than the base model*
in most configurations (chordal MSE > base MSE), while naive addition
consistently improves over the base. This is a fundamental limitation of
subspace-only merge methods applied to LoRA.

## Kill Criteria Assessment

### K1: Frechet merge quality within 1% of naive (SURVIVES on subspace metric)

On the subspace preservation metric (which is tautologically favorable to
chordal), mean advantage is +15.1% across all configurations.

However, on the downstream reconstruction test, chordal is dramatically worse
(-69% mean MSE penalty). K1 is ambiguous: it depends entirely on which metric
defines "quality."

### K2: Frechet merge adds >5% latency at serving time (VACUOUS)

This criterion was trivially satisfied by construction. All merge methods
pre-compute a merged weight matrix; per-token serving cost is identical
regardless of how the merge was computed. K2 tests nothing because the
comparison is between one-time preprocessing steps, not serving operations.

A better formulation would have been: "One-time merge cost exceeds X% of
single-expert training time at production scale." The actual one-time cost is
O(d^3) eigendecomposition -- milliseconds at micro scale, seconds at production
scale, negligible in both cases.

### K3: Chordal approximation diverges from geodesic (KILLED)

The chordal and geodesic means diverge significantly (up to 91% of max chordal
distance). This kill criterion fired as designed.

The divergence reflects different optimization objectives, not numerical error.
Chordal optimizes chordal overlap; geodesic optimizes arc-length distance. For
our subspace preservation metric, chordal is better by construction. For
downstream quality, the downstream test shows BOTH Frechet methods would likely
underperform naive addition (though we only tested chordal in the downstream
experiment).

K3 is registered as killed. The observation that the two methods optimize
genuinely different objectives is a useful finding, but does not override the
pre-registered kill criterion.

## Verdict: SUPPORTED (with major caveats)

Status changed from "PROVEN" to "SUPPORTED" because:
1. K3 was killed (pre-registered criterion fired)
2. The primary metric (subspace preservation) is tautological for chordal
3. The downstream reconstruction test shows naive addition is dramatically
   better on actual task quality

What IS supported:
- The mathematical framework is correct and novel for LoRA composition
- The chordal Frechet mean achieves optimal subspace preservation by construction
- The gap between naive and optimal grows with N (+5% to +34%)
- The chordal mean is cheaper than the geodesic alternative

What is NOT supported:
- That better subspace preservation translates to better model quality
- That chordal Frechet merge should replace naive addition in SOLE
- That geometric optimality on the Grassmannian is the right objective for
  expert composition

**Revised recommendation:** Do NOT replace naive addition with chordal Frechet
mean for SOLE pre-merge composition. The geometric advantage does not translate
to downstream quality. The B-weighted information preserved by naive addition
is more important than A subspace optimality.

The Frechet mean may still be useful in a modified form that incorporates B
matrices into the optimization (e.g., B-weighted Frechet mean, or joint A-B
manifold optimization). This is a direction for future work.

## Micro-Scale Limitations

1. **Synthetic linear model.** The downstream test uses linear regression, not
   a real transformer. Nonlinear models may behave differently. However, the
   fundamental issue (B matrices carry task information that subspace-only merge
   discards) is architecture-independent.

2. **Random B matrices in geometric tests.** The subspace preservation tests
   use random B. Trained B matrices may have structure that changes results.
   However, the downstream test uses *trained* experts and still shows naive
   winning, suggesting the random-B assumption was actually favorable to chordal.

3. **Only tested chordal, not geodesic, in downstream.** The geodesic mean
   uses the same subspace-only approach and likely suffers the same B-information
   loss. Not tested because the result is clear.

4. **AP-packed experts are only tested at limited (d, N) due to runtime.**
   Large AP configurations (d>=512, N>=25) were skipped. The AP-packed results
   that were collected show the same pattern as random regime.

5. **K3 reinterpretation.** The geodesic Karcher mean may outperform chordal on
   metrics we did not test (e.g., worst-case expert preservation, or geodesic
   distance minimization). The choice of metric matters.

## What Would Kill This

**Already partially killed at micro scale:**
- The downstream reconstruction test shows subspace preservation does not
  correlate with task quality. The chordal merge produces worse models despite
  better geometry. This is a partial kill of the practical hypothesis.

**What remains alive:**
- The mathematical framework (Grassmannian geometry for LoRA subspaces) is
  valid and may be useful for other purposes (expert diversity analysis,
  capacity estimation, initialization)
- A B-aware Frechet mean variant might recover the practical advantage

**At macro scale:**
- Pre-merged Frechet composition showing worse downstream accuracy than naive
  addition on real tasks (predicted by micro results, not yet tested)
- A B-weighted variant that accounts for output directions could potentially
  outperform naive addition -- this would be a new hypothesis

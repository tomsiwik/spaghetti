# Sheaf Cohomology Dimension Estimation: Proof Verification Report

## Theorem

**Theorem (Hansen & Ghrist 2110.03789, applied):** If the Cech nerve of the
adapter specialization cover has first Betti number beta_1 > 0, then there
exist beta_1 independent obstruction directions where adapter representations
are incompatible on shared samples. A bridge adapter of rank >= beta_1 is
necessary to reconcile these.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: 4-6 non-empty pairwise overlaps (k=2) | 6 edges (all 4 active adapters pairwise connected) | YES |
| P2: dim(H^1) >= 1 at intermediate layers | dim(H^1) = 3 at ALL layers (5, 10, 15, 20, 25) | YES (exceeds prediction) |
| P3: H^1 peaks at intermediate layers | H^1 = 3 CONSTANT across layers (topological, not data-dependent) | ILL-FORMED — topological H^1 is determined by the nerve (PPL rankings), which is layer-independent by construction. This prediction should not have been made; the layer-dependence claim conflates the topological invariant with the L2 magnitudes. L2 diffs DO peak at layer 15, but H^1 cannot vary by layer. |
| P4: dim(H^1) in [1, 10] | dim(H^1) = 3 | YES |

## Hypothesis

The Cech nerve of a top-k specialization cover over domain adapters has non-trivial
first cohomology, and dim(H^1) provides a lower bound on bridge adapter rank
needed for lossless composition.

## What This Experiment Is

This experiment computes the sheaf cohomology of a cover built from 5 domain LoRA
adapters (medical, code, math, legal, finance) on BitNet-2B-4T. It uses corrected
inputs from the predecessor experiment (Finding #240): specialization sets instead
of degenerate improvement sets, and L2 relative difference instead of saturated cosine.

## Key References

- Hansen & Ghrist (2110.03789): Knowledge Sheaves — cellular sheaf framework
- Curry (1303.3255): Sheaves, Cosheaves and Applications — theoretical foundation
- Eckmann (1944): Hodge decomposition for simplicial complexes
- Finding #240: Predecessor establishing corrected cover and metric

## Empirical Results

### Cover Structure (k=2)

| Adapter | Cover Size | Own-Domain | Top Cross-Domain |
|---------|-----------|------------|-----------------|
| Medical | 117/250 | 50/50 (100%) | math: 23, legal: 18 |
| Code | 119/250 | 49/50 (98%) | math: 18, finance: 19 |
| Math | 173/250 | 50/50 (100%) | code: 40, finance: 40 |
| Legal | 91/250 | 43/50 (86%) | finance: 26, medical: 13 |
| Finance | 0/250 | 0/50 (0%) | completely dominated |

**Key observation:** Math adapter is the strongest generalizer (173/250 in top-2),
while finance never appears in top-2 (completely dominated by other adapters at
scale=1.0). This creates a disconnected nerve (H^0 = 2).

### Cech Nerve (k=2)

- **Vertices:** 5 (4 active + finance isolated)
- **Edges:** 6 (complete graph on {medical, code, math, legal}; finance has 0 edges)
- **Triangles:** 0 (no three-way overlaps because each sample in exactly 2 cover sets)
- **Connected components:** 2 (K_4 subgraph + isolated finance vertex)
- **β₁ = |E| - |V| + c = 6 - 5 + 2 = 3** (where c=2 is the number of connected components)

### Sheaf Cohomology Results

| Layer | H^0 | H^1 (Betti) | H^1 (Hodge) | Global Diff Rank | Peak L2 Rel Diff |
|-------|-----|-------------|-------------|-----------------|------------------|
| 5 | 2 | **3** | **3** | 200 | 0.113 (med-math) |
| 10 | 2 | **3** | **3** | 250 | 0.131 (med-math) |
| 15 | 2 | **3** | **3** | 250 | 0.191 (med-code) |
| 20 | 2 | **3** | **3** | 247 | 0.175 (med-math) |
| 25 | 2 | **3** | **3** | 248 | 0.144 (med-math) |

**H^1 = 3 at all layers.** This means there are 3 independent topological cycles
in the nerve that cannot be filled by triangles (since no 3-way overlaps exist at k=2).
These represent 3 independent pairwise incompatibility cycles in the adapter composition
structure. **Important:** This is the SCALAR Betti number (topological cycle count),
not a dimension in R^{2560}. The scalar H^1 counts independent conflict cycles; the
actual obstruction space in representation space is captured by the full-rank edge
difference matrices (rank 13-68), which is much larger. See Implications section
and MATH.md Conjecture for the distinction.

### L2 Relative Differences (layer 15, peak)

| Edge | N Samples | L2 Rel Diff | Diff Rank |
|------|-----------|-------------|-----------|
| medical-code | 38 | 0.191 | 38 (full) |
| medical-math | 53 | 0.186 | 53 (full) |
| medical-legal | 26 | 0.135 | 26 (full) |
| code-math | 68 | 0.182 | 68 (full) |
| code-legal | 13 | 0.125 | 13 (full) |
| math-legal | 52 | 0.121 | 52 (full) |

**All edge difference matrices are FULL RANK** — every sample on an overlap contributes
a unique difference direction. The representations are genuinely incompatible, not
just shifted by a constant offset.

### Cover Sensitivity (k=3 comparison)

| Property | k=2 | k=3 |
|----------|-----|-----|
| Edges | 6 | 6 |
| Triangles | 0 | 4 |
| H^1 | **3** | **0** |
| Euler char | -1 | 3 |

At k=3, triangles fill all cycles and H^1 collapses to 0. This shows the obstruction
is real but sensitive to the cover granularity: at coarser covers (larger k), local
representations become "close enough" that the inconsistencies vanish.

**Interpretation:** k=2 captures the TIGHT compatibility requirements (where only
the top-2 adapters compete). k=3 relaxes this to top-3, where enough overlap
exists to transitively reconcile differences. Bridge adapters are needed precisely
in the k=2 regime — when routing selects between closely competing adapters.

### Finance Adapter Degenerate

Finance adapter (scale=1.0) never appears in any top-k set. This is consistent
with Finding #240 and the scale disparity (finance=1.0 vs medical/code/math=20.0).
The finance adapter is completely dominated and contributes nothing to the cover.
It creates a disconnected component (H^0 = 2).

## Kill Criteria Assessment

| Criterion | ID | Result | Evidence |
|-----------|-----|--------|---------|
| K1: Cover not degenerate | #648 | **PASS** | k=2 cover has sizes 0-173 (not all=250) |
| K2: H^1 > 0 at >= 1 layer | #649 | **PASS** | H^1 = 3 at ALL 5 layers |

## Limitations

1. **Topological vs data-informed H^1:** The H^1 = 3 result is purely topological
   (Betti number of the nerve graph), not data-weighted. A proper sheaf H^1 with
   vector-valued stalks might give different dimensions. However, full-rank edge
   differences confirm the differences are real.

2. **Finance degeneracy:** Only 4/5 adapters participate. At equal scales, the
   cover would be different. The finance scale=1.0 vs others at 20.0 creates
   an artifact.

3. **Sample size:** 250 samples (50/domain). Larger datasets might change
   overlap structure.

4. **k sensitivity:** H^1 depends critically on k. k=2 gives H^1=3, k=3 gives
   H^1=0. The "correct" k depends on the routing regime (top-1, top-2, etc.).

5. **Mean-pooled representations:** Token-level variation is averaged out.
   Token-level sheaf analysis might reveal finer structure.

## What Would Kill This

- If all adapters produce identical representations on overlaps (L2 rel diff < 0.01),
  sheaf framework adds no value. **Not the case:** L2 rel diff is 0.04-0.19.
- If H^1 = 0 for the routing-relevant k value, bridge adapters are unnecessary.
  **For k=3, this IS the case.** The question is whether k=2 or k=3 matches
  the routing regime.
- If bridge adapters of rank 3 do not improve composition quality, the theoretical
  prediction does not translate to practice. **Untested at this stage.**

## Implications for Bridge Adapter Design

1. **Topological lower bound: 3 independent conflict cycles.** The scalar H^1 = 3
   counts independent cycles in the nerve graph — i.e., 3 independent pairwise
   conflicts that cannot be transitively reconciled. This is a lower bound on the
   NUMBER of independent conflicts, NOT a proven rank budget in R^{2560}. The
   full-rank edge difference matrices (rank 13-68 per edge) suggest the actual
   obstruction space in representation space is much larger than 3. A bridge
   adapter of rank 3 addresses the topological minimum; the effective rank needed
   is an open empirical question (see Conjecture in MATH.md).

2. **Which pairs need bridges:** All 6 pairwise combinations of {medical, code,
   math, legal}. The strongest incompatibilities are medical-code (L2=0.191 at
   layer 15) and code-math (L2=0.182).

3. **Layer targeting:** L2 differences peak at layer 15 (mid-network), suggesting
   bridge adapters should target mid-network layers for maximum impact.

4. **Finance exclusion:** Finance adapter at scale=1.0 (vs 20.0 for others) is
   completely dominated — this is a scale artifact, not necessarily a property
   of the domain. At equal scales, finance would participate in the cover and
   the nerve topology would differ. Finance is excluded from all structural
   conclusions.

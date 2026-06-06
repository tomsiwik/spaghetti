# Symmetric GS Cost-Benefit Analysis: Research Digest

## Hypothesis

Symmetric Gram-Schmidt (averaging GS over many orderings) provides measurable
quality benefit at high pairwise cosine overlap (cos=0.85, matching attention-layer
production measurements).

**Falsifiable:**
- K1: If symmetric GS quality is within 0.1% of the best single ordering, the 50x cost is not justified.
- K2: If symmetric GS at cos=0.85 shows <1% improvement over a random single ordering, the method is useless.

---

## What This Model Is

This experiment directly tests whether symmetric Gram-Schmidt -- the theoretically
"fairest" orthogonalization (averaging over all orderings to eliminate first-expert
bias) -- provides any quality benefit over a single deterministic ordering.

The parent experiment (merge_order_dependence) showed that order variation scales
as ~80*cos and becomes measurable at cos > 0.06. Since attention layers have
cos = 0.85 between related domains, the question was: does the ~49% variation
at this cosine justify the 50-100x compute cost of symmetric GS?

Three phases:
1. **Cost-benefit sweep** across cosines 0.01-0.90 and P=1-100 orderings
2. **High-overlap deep dive** at cos=0.85 with full statistical characterization
3. **Practical alternatives** comparing deterministic orderings against symmetric GS

---

## Key References

- **merge_order_dependence** (this project, proven): Established variation ~ 80*cos.
  At cos < 0.06, order variation is below 5% CV.
- **layerwise_order_sensitivity** (this project, killed K2): Showed attention and FFN
  layers have identical order sensitivity mechanisms (slope ratio 1.01x).
- **gs_random_permutation_validation** (this project, proven): Random per-layer
  permutation reduces worst-case removal deviation by ~50%.
- **Golub & Van Loan, Matrix Computations:** GS order dependence is classical.

---

## Empirical Results

### Phase 1: Cost-Benefit Sweep

Symmetric GS is WORSE than single orderings at all cosine levels tested.
The quality loss grows with cosine overlap:

| Cosine | Single (canonical) | Symmetric (P=100) | Loss (%) | Cost ratio |
|--------|-------------------|-------------------|----------|------------|
| 0.01   | 0.4454            | 0.4444            | 0.22%    | 100x       |
| 0.10   | 0.4415            | 0.4375            | 0.90%    | 100x       |
| 0.30   | 0.4183            | 0.4017            | 3.98%    | 100x       |
| 0.50   | 0.3801            | 0.3522            | 7.32%    | 100x       |
| 0.70   | 0.3268            | 0.2954            | 9.63%    | 100x       |
| 0.85   | 0.2735            | 0.2490            | 8.95%    | 100x       |
| 0.90   | 0.2520            | 0.2329            | 7.55%    | 100x       |

**Symmetric GS costs 100x more and produces 9% worse quality at the critical
cos=0.85 operating point.** The effect is monotonically negative: more orderings
averaged means more destructive interference.

### Phase 2: High-Overlap Deep Dive (cos=0.85)

At cos=0.85, 3 seeds, 500 single orderings sampled:

| Metric | Value |
|--------|-------|
| Single ordering mean norm | 0.2734 |
| Single ordering CV | 0.12% |
| Best single ordering norm | 0.2741 |
| Worst single ordering norm | 0.2715 |
| Canonical ordering norm | 0.2735 |
| Symmetric GS (P=100) norm | 0.2490 |
| **Symmetric GS loss vs mean single** | **-8.92%** |
| **Symmetric GS loss vs best single** | **-9.15%** |
| Percentile of symmetric in single dist | 0.0% (below all 500 samples) |

The symmetric GS merged vector falls BELOW all 500 sampled single orderings.
It is strictly dominated.

### Phase 3: Practical Alternatives

All deterministic orderings perform identically and all outperform symmetric GS:

| cos | canonical | reverse | norm_desc | norm_asc | random_fixed | sym P=100 |
|-----|-----------|---------|-----------|----------|-------------|-----------|
| 0.30 | 0.4183 | 0.4182 | 0.4183 | 0.4183 | 0.4181 | 0.4017 |
| 0.50 | 0.3801 | 0.3800 | 0.3801 | 0.3801 | 0.3798 | 0.3523 |
| 0.70 | 0.3268 | 0.3267 | 0.3268 | 0.3268 | 0.3265 | 0.2954 |
| 0.85 | 0.2735 | 0.2735 | 0.2735 | 0.2735 | 0.2732 | 0.2490 |

The gap between ANY deterministic ordering and symmetric GS is 4-10%.
The gap between deterministic orderings themselves is < 0.1%.

**Convergence:** Symmetric GS requires P >= 75-100 orderings to stabilize,
but it converges to a quality floor that is 9% below any single ordering.
More compute makes it worse, not better.

---

## Kill Criteria Assessment

### K1: Symmetric GS within 0.1% of best single ordering

**KILLED (inverted).** Symmetric GS is 9.15% WORSE than the best single ordering,
not within 0.1%. The spirit of K1 is "no benefit worth 50x cost." Since symmetric
GS provides negative benefit (active quality degradation), K1 is decisively met.

### K2: Symmetric GS at cos=0.85 shows <1% improvement over random ordering

**KILLED (inverted).** Symmetric GS shows -8.92% change vs random ordering
(a degradation, not improvement). The improvement is less than 1% -- in fact
it is negative. K2 is decisively met.

---

## Verdict: KILLED -- Symmetric GS is Strictly Dominated

Symmetric Gram-Schmidt does not provide a quality benefit at any cosine level
tested. It actively degrades quality by 0.2-9.6%, with the worst degradation
occurring at the high-overlap regime (cos=0.70-0.85) that motivated its
investigation.

The mechanism is clear: different orderings produce merged vectors that share
a common component along the shared direction but differ in their orthogonal
residual components. Averaging these diverse residuals causes destructive
interference, reducing the merged vector's norm (total retained signal).

Meanwhile, the variation among single orderings is tiny (CV = 0.12% at
cos=0.85). Any single deterministic ordering -- canonical sort, random with
fixed seed, or any other -- produces essentially identical quality.

---

## Implications for SOLE

1. **Do not implement symmetric GS.** It costs 100x more and produces 9% worse results.

2. **Any deterministic ordering is fine.** The variation among orderings is 0.12%
   at the worst-case production cosine (0.85). Canonical sort by expert ID is the
   simplest choice.

3. **Order sensitivity is a direction problem, not a quality problem.** Different
   orderings produce merged vectors that point in slightly different directions
   but have nearly identical norms. For downstream model quality (which depends
   on norm, not direction), this variation is irrelevant.

4. **This closes the GS investigation.** Between this experiment, merge_order_dependence
   (proven: order variation negligible), layerwise_order_sensitivity (killed: no layer-type
   effect), and gs_random_permutation_validation (proven: permutation equalizes), the
   entire GS ordering question is resolved. Simple averaging without GS remains the
   SOLE default.

---

## Micro-Scale Limitations

1. **Synthetic vectors, not real LoRA deltas.** The uniform-overlap model creates
   experts that share a single direction. Real LoRA deltas may have more complex
   overlap structure (multiple shared subspaces). However, the destructive interference
   mechanism is geometric and applies to any overlap structure.

2. **Norm as quality proxy.** We use merged vector norm as the quality metric, not
   NTP loss or generation quality. This is justified because: (a) larger norm means
   more expert signal retained, (b) the parent experiment showed norm CV tracks
   quality CV, and (c) at 0.12% variation, no downstream metric could distinguish
   the orderings.

3. **N=5 experts.** With N=5, there are only 120 total orderings. At N=50,
   the permutation space is astronomically larger, but the averaging mechanism
   (destructive interference of residuals) should be even stronger.

4. **d=256 synthetic dimension.** Production d=4096. At higher d, cosines are
   lower (structural orthogonality), making the entire GS ordering question
   even more irrelevant.

---

## What Would Kill This

This experiment is already killed. The finding (symmetric GS is strictly
dominated) could only be overturned if:

- Real LoRA deltas at macro scale (d=4096) showed qualitatively different
  behavior where averaging over orderings somehow increased norm. This would
  require correlated overlap structure that produces aligned (not diverse)
  residuals across orderings -- a very specific and unlikely condition.

- A different aggregation method (e.g., geometric mean, median) instead of
  arithmetic mean produced better results. This was not tested but is unlikely
  to help since the fundamental issue is directional diversity of residuals.

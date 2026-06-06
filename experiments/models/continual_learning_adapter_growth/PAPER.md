# N-Scaling Evaluation Under Uniform Composition: Research Digest

## Hypothesis

Uniform 1/N composition of Grassmannian-skeleton LoRA adapters maintains
composition quality within 1% of base across N=5 to N=15, demonstrating
that the orthogonal adapter pool scales without meaningful degradation.

## What This Experiment Does

Evaluates pre-existing domain adapters (trained in exp_real_data_25_domain_adapters)
at increasing pool sizes N=5 to N=15 under uniform 1/N composition. After each
N-increment, measures:
1. Composition quality (uniform 1/N) across ALL current domains
2. Per-domain composition degradation vs reference cycle
3. Pairwise B-matrix parameter cosine similarity (NOT Grassmannian A-orthogonality,
   which is guaranteed by construction)

**Important framing note:** No training occurs in this experiment. All adapters
are reused from prior work. Individual adapter invariance (0% drift) is a
mathematical tautology when parameters are frozen -- it confirms code correctness,
not a hypothesis. The non-trivial finding is composition robustness to N.

## Key References

- arXiv 2508.11985 (Naive LoRA Summation): orthogonal A-matrices enable additive composition
- Our own N=5 to N=24 scaling (exp_real_data_25_domain_adapters): composition IMPROVES with N
- VISION.md architecture: Grassmannian AP-packed frozen A, STE ternary B

## Empirical Results

### Kill Criteria

| Criterion | Threshold | Actual | Verdict |
|-----------|-----------|--------|---------|
| K1 (#247): Any existing adapter degrades >5% after addition | <5% | 0.53% worst (creative_writing, N=15) | **PASS** |
| K2 (#248): Composition quality degrades monotonically with N | NOT monotonic | NOT monotonic (non-monotonic trajectory) | **PASS** |

### Individual Adapter Invariance (Sanity Check)

Individual adapter PPL is trivially invariant when A_i and B_i are both frozen.
This confirms code correctness, not a scientific finding.

| Domain | Initial PPL | Final PPL (at N=15) | Drift |
|--------|------------|---------------------|-------|
| medical | 6.4077 | 6.4077 | 0.0000% |
| code | 5.1365 | 5.1365 | 0.0000% |
| math | 3.8809 | 3.8809 | 0.0000% |
| legal | 21.7417 | 21.7417 | 0.0000% |
| finance | 19.0582 | 19.0582 | 0.0000% |

### Composition Quality Trajectory (Mean Improvement vs Base)

| N | Mean Composition Improvement | Direction |
|---|-----|-----------|
| 5 | -0.59% | baseline |
| 6 | -0.51% | improving |
| 7 | -0.28% | improving |
| 8 | -0.30% | small regression |
| 9 | -0.22% | improving |
| 10 | -0.22% | flat |
| 11 | -0.12% | improving |
| 12 | +0.04% | **crosses zero** |
| 13 | +0.09% | **best** |
| 14 | -0.04% | small regression |
| 15 | -0.51% | regression |

The trajectory is non-monotonic, ruling out strict dilution dominance (K2 PASS).
Composition quality is approximately flat across N=5-15, fluctuating within ~0.6pp
of zero. The apparent +0.09% at N=13 is within noise (single seed, no error bars)
and should NOT be interpreted as composition beating base. A multi-seed study
would be needed to distinguish any "sweet spot" from measurement variance.

### B-Matrix Parameter Cosine Stability

**Note:** These measurements are pairwise cosine similarity of flattened B-matrix
parameter vectors, NOT Grassmannian A-orthogonality. A-orthogonality is guaranteed
by QR construction and is exactly zero -- measuring it would be uninformative.
B-parameter cosine measures statistical similarity of learned weights across domains.

| N | Mean B-|cos| | Max B-|cos| |
|---|------------|-----------|
| 5 | 0.0205 | 0.0441 |
| 10 | 0.0233 | 0.0704 |
| 15 | 0.0230 | 0.0756 |

B-parameter cosine is stable and low across N=5 to N=15, indicating that
independently trained adapters do not converge to similar weight vectors.
The theoretical orthogonality capacity is N_max = d/r = 160.

### Per-Domain Composition at N=15

| Domain | Base PPL | Composed PPL | Delta |
|--------|----------|-------------|-------|
| medical | 6.50 | 6.55 | +0.8% |
| code | 4.98 | 4.99 | +0.3% |
| math | 3.84 | 3.86 | +0.6% |
| legal | 21.63 | 21.71 | +0.3% |
| finance | 19.43 | 19.46 | +0.2% |
| health_fitness | 11.71 | 11.77 | +0.5% |
| psychology | 17.45 | 17.45 | -0.0% |
| science | 12.46 | 12.53 | +0.6% |
| history | 16.70 | 16.83 | +0.8% |
| philosophy | 16.39 | 16.51 | +0.7% |
| creative_writing | 20.73 | 20.87 | +0.7% |
| cooking | 3.21 | 3.22 | +0.5% |
| education | 3.58 | 3.59 | +0.5% |
| engineering | 4.21 | 4.22 | +0.4% |
| agriculture | 14.56 | 14.67 | +0.8% |

At N=15 with uniform 1/N, most domains are within 1% of base. No domain exceeds
+0.8% degradation -- far below the 5% kill threshold.

## Key Finding

Uniform 1/N composition of orthogonal adapters maintains PPL within ~1% of base
across N=5-15. The composition trajectory is non-monotonic (ruling out strict
dilution dominance), but all fluctuations are small enough to be consistent with
noise at single-seed resolution.

**What this shows:** The Grassmannian skeleton provides composition robustness
to N under uniform weighting. Adding adapters does not meaningfully degrade
composed output quality in the N=5-15 regime.

**What this does NOT show:** "Continual learning" or "forgetting immunity" --
individual adapter invariance is tautological with frozen parameters, and no
training occurs during this experiment.

## Limitations

1. **No training occurs** -- all adapters reused from exp_real_data_25_domain_adapters.
   This is an N-scaling evaluation, not a continual learning experiment. A true
   continual learning test would train each adapter sequentially, potentially with
   different data distributions per step.

2. **Uniform 1/N composition** -- this is the WORST case for composition quality.
   Routing (top-k selection) would improve results significantly. The N=15 regression
   is expected under uniform weighting and would be mitigated by routing.

3. **Single seed (42), no confidence intervals** -- composition deltas are <1%,
   comparable to the ~0.5% CV observed in prior multi-seed work. The trajectory
   shape (non-monotonic) may not be reproducible. Multi-seed validation needed
   before drawing conclusions about optimal N.

4. **Several "domains" are arbitrary dataset slices** -- science, history, philosophy,
   creative_writing, agriculture are consecutive 500-sample slices from dolly-15k,
   not genuinely distinct domains. This was inherited from prior experiments.

5. **15 domains only** -- prior experiment showed composition improvement at N=24.
   This experiment covers the intermediate N regime (N=5 to 15).

## What Would Kill This

- K1 would fail if composition degradation exceeded 5% for any domain after adding
  a new adapter. This would indicate that the Grassmannian guarantee is insufficient
  and that B-matrix cross-terms cause meaningful interference.

- K2 would fail if composition quality decreased monotonically with N, indicating
  that dilution always dominates coverage. This would mean the "more contributors
  = better" thesis is false under uniform composition.

K1 passed with margin (0.53% vs 5% threshold). K2 passed (trajectory is non-monotonic),
though the trajectory is approximately flat rather than showing a clear improvement region.

## Verdict: SUPPORTED

Total runtime: 962 seconds (~16 minutes) on Apple M5 Pro 48GB.

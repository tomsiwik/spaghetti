# NP-LoRA Null Space Projection: Research Digest

## Hypothesis

Null space projection (NP-LoRA) can mathematically eliminate cross-adapter
interference during composition, potentially improving upon our Grassmannian
skeleton approach for ternary LoRA composition.

**Falsifiable prediction:** NP-LoRA projection of adapter deltas into the null
space of all other adapters will reduce composition ratio below Grassmannian
baseline and compute the projection in under 1 second for N=50.

## What This Experiment Tests

NP-LoRA (arxiv 2511.11051) projects each adapter's effective delta
(Delta_i = B_i @ A_i) into the orthogonal complement of the subspace spanned
by all other adapters' deltas, using SVD to identify the interference subspace.

We test three conditions:
1. NP-LoRA applied to Grassmannian (QR-orthogonal A) adapters
2. NP-LoRA applied to random (non-orthogonal A) adapters
3. Computational scaling to N=50

## Key References

- NP-LoRA (arxiv 2511.11051) -- Null space projection for LoRA fusion
- Our Grassmannian skeleton (VISION.md) -- QR-orthogonal frozen A matrices
- FlyLoRA (arxiv 2510.08396) -- JL-lemma analysis of random A orthogonality

## Empirical Results

### Setup
- d=256, 6-layer ternary GPT, rank-8 LoRA, 5 domain adapters
- Ternary (STE) adapters on character-level names dataset

### Grassmannian Baseline vs. NP-LoRA

| Method | Mean Composed PPL | Composition Ratio | Mean |cos| |
|--------|-------------------|-------------------|------------|
| Grassmannian (no proj) | 1.5580 | 1.0224 | 2.54e-7 |
| NP-LoRA + Grassmannian | 1.5580 | 1.0224 | 2.54e-7 |

**NP-LoRA is a complete no-op on Grassmannian adapters.** The relative change
from projection is 3.37e-6 (essentially zero). This is expected: with
A_i @ A_j^T = 0 by construction, the adapter deltas already lie in mutually
orthogonal subspaces, leaving nothing for null space projection to fix.

### Random A Matrices: Does NP-LoRA Help When Needed?

| Method | Mean Composed PPL | Composition Ratio | Mean |cos| |
|--------|-------------------|-------------------|------------|
| Random A (no proj) | 1.5595 | 1.0227 | 9.34e-4 |
| Random A + NP-LoRA | 1.5595 | 1.0227 | 9.29e-4 |

**NP-LoRA provides 0.00% improvement even on random A matrices.** Despite
random A having 3700x higher cross-talk than Grassmannian (|cos| 9.34e-4 vs
2.54e-7), this level of interference is too small to meaningfully impact PPL
at d=256/r=8. The Johnson-Lindenstrauss lemma predicts this: at d/r=32,
random subspaces are already near-orthogonal.

### Computation Time Scaling

| N | Projection Time (sec) | K2 Verdict |
|---|----------------------|------------|
| 5 | 0.79 | PASS |
| 15 | 12.3 | FAIL |
| 50 | 318.3 | FAIL (318x over) |

The O(N^3 * L * d^2) scaling is catastrophic. At N=50, the null space
projection takes over 5 minutes -- 318x over the 1-second threshold.
This alone disqualifies NP-LoRA for our use case.

### Kill Criteria

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: NP-LoRA not worse than Grassmannian | PPL ratio <= 1.01 | 1.0000 | PASS |
| K2: N=50 projection < 1 sec | < 1.0s | 318.3s | **FAIL** |

**Status: KILLED (K2)**

## Why NP-LoRA Fails Here (Three Independent Reasons)

1. **Grassmannian already solves the problem.** With A_i @ A_j^T = 0, there
   is zero cross-term interference by construction. NP-LoRA has nothing to
   project away.

2. **Even without Grassmannian, interference is negligible.** At d/r=32,
   random subspaces are nearly orthogonal (|cos| < 0.001). The interference
   is too small to affect PPL. This validates the JL-lemma argument from
   FlyLoRA.

3. **The computation is prohibitively expensive.** SVD of (N-1, d_out*d_in)
   matrices across all layers scales as O(N^3*L*d^2). For practical N,
   this is orders of magnitude slower than the composition itself.

## Implications for the Project

1. **Grassmannian skeleton is validated as optimal.** No post-hoc projection
   method can improve on zero interference. The QR-orthogonal A matrix
   approach is both cheaper and more effective.

2. **Random A works surprisingly well at our scale.** Composition ratio
   1.0227 (random) vs 1.0224 (Grassmannian) -- practically identical.
   This confirms FlyLoRA's finding that high d/r ratios give natural
   orthogonality. Grassmannian is insurance, not necessity, at d=256/r=8.

3. **Post-hoc interference reduction is the wrong approach.** The right
   place to ensure orthogonality is at initialization (Grassmannian A),
   not at composition time (NP-LoRA). Pre-hoc beats post-hoc.

## Limitations

- Tested at d=256 only. At smaller d/r ratios (e.g., d=64/r=16=4),
  random A would show more interference and NP-LoRA might help.
- Only tested with 5 domains. Cross-term interference grows with N.
- Toy task (character-level names) -- real-world tasks may have higher
  B-matrix correlation.
- NP-LoRA was designed for diffusion model LoRA fusion (subject+style),
  not language model adapter composition.

## What Would Kill This Kill

If at larger scale (d/r < 10), NP-LoRA showed meaningful PPL improvement
AND could be computed efficiently (e.g., via iterative projection or
approximate null space), it would deserve re-evaluation. But the
Grassmannian approach eliminates the need entirely.

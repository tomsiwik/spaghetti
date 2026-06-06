# Finding Critique: Frechet Merge

## Target
`micro/models/frechet_merge`

## Critique

1. **Tautological Metric Hacking:** The primary metric, "subspace preservation" (`||U_merged^T U_i||_F^2 / r`), is exactly the mathematical definition of the chordal Frechet mean's optimization objective. By definition, the chordal mean maximizes this overlap. Thus, showing that chordal Frechet mean beats naive addition on this metric is a mathematical tautology, not an empirical finding of general validity.

2. **Catastrophic Downstream Failure (Ignored B-matrices):** The chordal Frechet merge exclusively optimizes the A-matrix subspace geometry while entirely discarding the B-weighted information. This B-weighted information carries the actual trained task knowledge. As the downstream reconstruction test empirically proves, this omission leads to a catastrophic degradation in model quality, where naive addition dramatically outperforms chordal Frechet merge (e.g., -187% worse MSE at d=64, N=25). The method essentially produces a model worse than the base model.

3. **Vacuous Latency Evaluation (K2):** Kill criterion K2 assessed whether the Frechet merge adds >5% latency at serving time. This is vacuously satisfied because all merge methods pre-compute the same shaped weight matrix before inference. Therefore, the inference serving cost is perfectly identical by definition.

## Verdict
**Invalid.** The experiment relies on a tautological metric, and the geometric "optimality" of the A-subspace directly destroys the B-weighted representations critical for downstream performance.
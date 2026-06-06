# KSG Mutual Information Estimator

## Paper
Kraskov, A., Stogbauer, H., & Grassberger, P. (2004).
"Estimating mutual information."
Physical Review E, 69(6), 066138.
https://arxiv.org/abs/cond-mat/0305641

## What It Is
K-nearest-neighbor based estimator for mutual information between
continuous random variables. Algorithm 1 computes:

I(X;Y) = psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N)

where psi is the digamma function, n_x/n_y count neighbors in marginal
spaces within the k-th neighbor distance in joint space.

## Implementation
Our implementation in `micro/models/mi_expert_independence/mi_expert_independence.py`
uses scipy.special.digamma and scipy.spatial.KDTree. Two variants:
- `ksg_mi()`: General d-dimensional KSG
- `ksg_mi_1d_fast()`: Optimized for 1D inputs using sorted arrays

## Key Properties
- Non-parametric (no distribution assumptions)
- Reliable for 1D with N >= ~100 samples
- Degrades in high dimensions (curse of dimensionality)
- At d=64, N=640: unreliable. At d=1, N=640: reliable.

## Alternatives Considered
- MINE (neural network estimator): Overkill for pairwise measurement
- sklearn.feature_selection.mutual_info_regression: Uses KSG internally but requires sklearn
- Binning: Wastes data badly at d=64

## Used In
- exp_mi_expert_independence: MI vs cosine for expert independence measurement

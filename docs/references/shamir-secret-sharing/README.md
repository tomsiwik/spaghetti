# Shamir's Secret Sharing (1979)

## Source
- Paper: Shamir, A. "How to share a secret." Communications of the ACM, 22(11), 1979.
- Wikipedia: https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing
- Python library: https://github.com/blockstack/secret-sharing (GF(p) implementation)

## Key Insight
k-of-n threshold scheme using polynomial interpolation. Any k shares reconstruct
the secret; fewer than k shares reveal nothing (information-theoretic security
over finite fields).

## Relevance to Our Work
Applied to expert MLP weight sharing for fault-tolerant composition:
- Over real numbers (not GF(p)) -- trades security for simplicity
- Exact reconstruction via Lagrange interpolation (float64 precision)
- Provides distributed serving redundancy for expert libraries
- Does NOT provide meaningful expert blending (polynomial diverges away from x=0)

## What We Learned
- Quality: 0.000% degradation across 3 seeds, all share subsets -- numerically exact
- Overhead: 14-27% per-reconstruction vs forward pass (KILLED at per-call level)
- Amortized overhead: 0.018% for B=1000 generation (trivial)
- Blending: polynomial evaluation at non-share points is random perturbation, not interpolation
- No prior work found applying Shamir to individual neural network weights

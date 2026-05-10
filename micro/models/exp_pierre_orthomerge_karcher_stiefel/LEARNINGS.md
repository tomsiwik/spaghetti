# LEARNINGS — OrthoMerge Karcher Stiefel

## Finding #ORTHO-1: Riemannian methods needing W_0 incompatible with adapter-only composition

OrthoMerge (arxiv 2602.05943) requires the base weight W_0 for Procrustes alignment. The no-base fallback (W_0=I) is both mathematically degenerate and numerically unstable (SVD crash at scale=6.0). Combined with Finding #ACE-1, this closes the class of "structure-aware" merge methods for Pierre.

## Finding #ORTHO-2: Rectangular weight matrices break naive orthogonal-group algorithms

Pierre's Gemma 4 has non-square projections (q_proj: 2560→2048, up_proj: 2560→4096). Any method operating in O(n) must handle d_in ≠ d_out gracefully. Two shape bugs surfaced before the numerical crash.

## Finding #ORTHO-3: Pierre composition methods must be Euclidean

Viable: Fisher-Rao, DARE, TIES, simple averaging — all operate in Euclidean space on B-matrices or materialized ΔW without structural assumptions about W_0.

Not viable: ACE (covariance degenerate under shared-A), OrthoMerge (needs W_0), any Riemannian method requiring base weight access.

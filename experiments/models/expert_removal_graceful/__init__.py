"""Expert Removal Graceful: tests whether removing an expert from a
Gram-Schmidt-composed merged model breaks remaining experts.

Hypothesis: Naive subtraction (O(1)) is sufficient for expert removal
thanks to near-orthogonality; GS cascade recomputation is unnecessary.

Kill criteria:
  - Removing expert causes >3% PPL regression on remaining experts
  - Gram-Schmidt cascade recomputation takes >10 min at N=50
"""

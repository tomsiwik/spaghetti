# LEARNINGS: exp_p5_universal_subspace_analysis

## Core Finding
Universal Subspace Hypothesis (arXiv:2512.05117) does not apply to mixed-init adapter collections:
B-matrices converge to a rank-5 domain subspace (5 unique directions for 5 domains), but
A-matrices cluster by initialization method, not by domain.

## Why
The dataset was confounded — Grassmannian-ortho adapters share identical B-matrices with
their standard counterparts, and the two A-matrix populations (standard cos≈0.82 vs ortho cos≈0)
are structurally incompatible. Within each initialization method, behavior matches theory exactly.
Finding #65 confirmed on Gemma 4: Grassmannian A-matrices have no shared subspace.

## Key Structural Result
Universal subspace compression and Grassmannian composition are mutually exclusive:
compression into a shared basis (K=4: 30% A-info lost) destroys the orthogonality guarantee
(max_cos 0.60 → 0.96 after projection). Pierre correctly chose composition over compression.

## Kill Criteria Correction (per adversarial review)
- K1282: FAIL (not degenerate PASS) — at K=4, variance=70.7% < 80% threshold
- K1283: FAIL — universal vs naive delta <0.02, noise-level
- K1284: PASS for fidelity but kills composition (orthogonality destroyed)

## Implications for Next Experiment
B-matrices are not the composition bottleneck — they naturally domain-cluster regardless
of A-init. The interference problem lives entirely in A-space. Next step: exploit the
B-matrix domain structure for faster adapter specialization (B-matrix warm-starting for
new domains, since B converges to a predictable rank-5 subspace after domain training).

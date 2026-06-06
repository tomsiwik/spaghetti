# LEARNINGS: exp_slerp_b_composition

## Core Finding

SLERP norm preservation is mathematically proven (Theorem 2, K931 PASS: 2.06× norm ratio) but does NOT improve multi-domain composition quality — LERP wins on all 5 domains (K932 FAIL, mixed loss 0.402 vs 0.463). The candy-wrapper norm collapse is **implicit regularization**, not a defect.

## Why

For N=5 diverse adapters with near-orthogonal B matrices (mean cosine ≈ 0.06), both LERP and SLERP produce equally arbitrary directions for any specific domain. SLERP's larger magnitude amplifies noise without improving direction alignment. LERP's 1/√N magnitude reduction keeps the composed adapter closer to the base model, which already generalizes across domains — this is beneficial regularization, not a bug.

## Implications for Next Experiment

SLERP is only useful for merging **similar** adapters (same domain, different seeds). For diverse multi-domain composition, routing (Finding #354: 95% TF-IDF accuracy) is the correct solution — it selects the right adapter per input rather than composing all adapters with any blending method. No further SLERP/blending experiments needed; pursue routing and null-space isolation instead.

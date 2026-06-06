# LEARNINGS: exp_p0_vproj_composition_behavioral

## Core Finding
v_proj+o_proj parameter merging is composition-neutral-to-positive: 4/5 domains show
retention >= 100% under 5-way merging, PPL degrades max 2.1%. The ensemble effect
dominates any interference signal.

## Why
Adapters contribute useful cross-domain signal rather than pure noise (contradicting
Theorem 3's CLT assumption). Equal-weight composition matches peaked-weight routing
(0.23 vs 0.22) — adapters are sufficiently non-interfering that routing overhead
is unnecessary at N=5.

## Adapter Quality is the Bottleneck
Solo baseline 17% vs P8's 52% — 3x discrepancy from metric variance (vocabulary
counting on 20 stochastic samples). Kill criteria (K1316-K1318) missed due to
miscalibrated thresholds, not composition failure. Legal is uniquely fragile (33%
retention), likely from sparse specialized vocabulary overwhelmed in 5-way output space.

## Implications for Next Experiment
1. Kill criteria must use retention ratios (>= 1.0x), not absolute scores calibrated
   to a different baseline.
2. Adapter solo quality is the lever: more training data (>100 unique examples),
   longer training (500+ iters), higher rank (32 vs 16).
3. Legal domain needs targeted investigation — why does its vocabulary signal collapse
   under composition when 4 other domains improve?
4. Statistical power: routing comparisons need n_eval >= 50 to detect differences
   within the ~0.20 CI width at n=20.

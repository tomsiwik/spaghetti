# LEARNINGS — exp_spark_logit_decile_clash

## Core finding
The "logit-prior clash" signal — adapter pushing probability mass onto base-pruned (bottom-decile)
tokens — is effectively uncorrelated with per-token KL damage: rho = 0.0241 (z≈1.55, p>0.05).
The mechanism is also rarely active (mean clash 0.081; only 2.2% of tokens with clash>0.5).

## Why
The composition damage does not concentrate at the output-distribution boundary between the adapter's
top logit shifts and the base's suppressed tokens. The proposed clash geometry is not where damage
lives; the signal is ~9× weaker than even the weak surviving geometric baseline (delta-magnitude,
rho = 0.2076), making the output-prior-clash class of predictors a dead end.

## Implication for the next experiment
Delta-magnitude (hidden-state delta norm) holds the only statistically real signal against a
per-token KL label (rho = 0.2076, ~4% rank variance, z≈13.3, p≪0.001). This nuances F#864/868/869
("magnitude is dead"): those nulls were label-dependent (sign/correctness/timing labels); against
KL-divergence, magnitude has a small but genuine correlation. The next experiment should probe
*why* delta-magnitude predicts KL damage — e.g., whether it is driven by a subset of layers,
token types, or adapter overlap — rather than discarding magnitude as a predictor class entirely.

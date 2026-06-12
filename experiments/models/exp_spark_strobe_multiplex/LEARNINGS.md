# LEARNINGS — exp_spark_strobe_multiplex

## Core finding
Blind round-robin strobing of adapter weights across decode steps loses −27.45pp aggregate accuracy
against a magnitude-matched (1/N)Σ STATIC_NORM baseline (STROBE 56.86 vs STATIC_NORM 84.31, n=51),
failing on all three domains (math −35pp, python −18pp, medical −29pp).

## Why
The earlier apparent +19.6pp "win" was a magnitude confound: strobing was compared to an un-scaled
raw sum (STATIC_RAW 37.25pp) that over-drives activations. Once the baseline is correctly normalized
to the same per-token magnitude, strobing introduces token-step incoherence — each forward pass sees
a different adapter's weight surface — and the model cannot build consistent hidden state trajectories
across layers, collapsing accuracy well below static composition.

## Implication for the next experiment
Time-multiplexing / strobing is not a viable multiplexing mechanism; the interference problem must be
solved in weight-space or output-space at each token, not by cycling adapters across tokens. Any
future multi-adapter scheme must beat STATIC_NORM (magnitude-matched average) as its gating baseline,
not a raw unscaled sum.

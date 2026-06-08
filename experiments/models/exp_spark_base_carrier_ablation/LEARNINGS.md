# LEARNINGS — exp_spark_base_carrier_ablation

## Core Finding
The frozen 4-bit base q_proj is load-bearing, not a carrier wave: zeroing it collapses
on-domain accuracy to chance (R_mean(0)=0.094; math R=0.0, code R=0.0, medical R=0.281).

## Why
A rank-6 LoRA delta is defined relative to W_q — it is a correction, not a standalone
query engine. Removing W_q entirely breaks attention routing universally across all 42
q_proj layers, leaving no usable query signal. The monotone-steep decay curve (not flat)
lands squarely on the load-bearing arm of the Theorem 2.1 dichotomy.

## Implication for the Next Experiment
The "frozen base = sacred" invariant is confirmed: do NOT attenuate the full base
projection. If anyone revisits carrier decomposition, attenuate only a domain-specific
subspace of W_q (e.g., projection onto the LoRA column space) — never the whole W_q.
The confound here is that α=0 cannot separate "base carries domain knowledge" from "base
carries any query signal at all"; medical's 0.281 retention is consistent with the 0.25
MCQ chance floor, not genuine partial retention.

## What NOT to Retry
- Do not re-ablate the whole base q_proj (this experiment closes that direction).
- Do not interpret medical's slower decay as evidence of partial domain retention.
- The carrier-wave hypothesis (flat R curve) is refuted; future adapter work must treat
  W_q as a substrate the adapter perturbs, not a gain it rides on.

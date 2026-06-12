# LEARNINGS — exp_spark_velocity_mask_interference

## Core finding
Destructive interference from composing a thinking adapter with a math adapter is localized in the thinking
adapter's late-moving weights; the early-velocity core (sign-stable, high-magnitude across training) composes
cleanly and recovers +30pp over the full-thinking composition, even exceeding the math-solo ceiling (0.74 vs 0.70).

## Why
The late weights of a thinking adapter contain direction-unstable deltas that destructively interfere with the
math adapter's q_proj directions; the early-stabilized core carries only sign-consistent signal and adds
independently without cross-projection conflict.

## Implication for the next experiment
Early-velocity masking is a viable interference-prevention primitive for adapter composition — the next
experiment should test whether the mask generalizes across adapter pairs (e.g., different domain adapters or
different base thinking runs) or whether the core fraction (0.43 here) is data-dependent and must be calibrated
per-pair.

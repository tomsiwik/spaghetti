# LEARNINGS — exp_spark_temporal_interference

## Core Finding
Off-domain LoRA interference on frozen Gemma-4 is NOT temporally localized to high-base-entropy decode steps.
Entropy-gating recovered exactly as much accuracy as random dropout (recov(D) = recov(E) = 0.063, ΔD−E = 0.0pp).

## Why
The perturbation (spark) hypothesized that interference is a sparse temporal event concentrated on ~5% of decode steps with the highest base-model entropy ("choice points"), and that zeroing the off-domain adapter only there would recover most of the 20pp accuracy drop from composition. The data refuted this: the entropy gate and the random gate produced identical aggregate outcomes on n=80 GSM8K (D≠C on 11/80, E≠C on 7/80, D≠E on 12/80 — all small, no signal in the entropy-indexed subset). The 20pp composition drop (B 66.2% → C 46.2%) is real and behavioral, but the variance is diffuse across all decode positions; no low-cardinality token subset carries it.

## Measured Table
| Arm | Accuracy | vs. B (math-only) |
|-----|----------|-------------------|
| A base | 11.2% | — |
| B math-only | 66.2% | — |
| C compose (math+code) | 46.2% | −20.0pp |
| D entropy-gate | 47.5% | recov = 0.063 |
| E random-gate | 47.5% | recov = 0.063 |

Kill criteria: K1 FAIL (recov < 0.50), K2 FAIL (ΔD−E = 0.0 < 2pp), K3 pass-but-moot. Verdict: KILLED.

## Why the Random-Gate Control Is Load-Bearing
D == E in aggregate is not a numerical coincidence — they differ on 12/80 items in opposite directions. This rules out the gate being a no-op; it acted but its entropy signal carried zero information above chance. The temporal axis is therefore not merely unpredictive but random-equivalent.

## Implication for the Next Experiment
Do not re-attempt temporal or entropy-indexed interference gating in any form. The decode-step axis joins weight-space merge weights, routing, and Grassmannian orthogonality as axes that fail to localize or fix LoRA composition interference. Interference of this magnitude (20pp) requires a static-composition solution (NRE / Fisher-Rao). Future work should focus on K=7 Fisher-Rao upper-bound experiments, not dynamic per-step gating.

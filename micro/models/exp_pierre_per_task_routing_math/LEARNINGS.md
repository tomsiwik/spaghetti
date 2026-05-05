# LEARNINGS — Per-Task Routing for Math

## Core Finding

The GSM8K -6.7pp regression (Finding #831) is NOT caused by DARE composition degrading math capability. Single math adapter and DARE composition both yield identical 63.3% on GSM8K (N=30). The 70% reference was a measurement artifact from a different sample set or generation config.

## Why It Matters

Routing cannot fix a regression that doesn't exist. The "disease" was misdiagnosed — there is no DARE-specific math interference to route around. Future composition work should not treat GSM8K 63.3% as a regression from 70%; it may simply be the adapter's ceiling on this eval set.

## Reusable Assets

- **TF-IDF + RidgeClassifier router**: 100% accuracy, 0.072ms overhead, near-zero cost. Validated primitive for any future per-domain routing needs.
- **Finding**: N=30 code-generation benchmarks (HumanEval) have ~23pp stochastic variance between identical runs. Any result within that band is noise.

## Implication

Improving GSM8K requires a better math adapter or better composition math — not routing. The composition pipeline is not the bottleneck for math accuracy.

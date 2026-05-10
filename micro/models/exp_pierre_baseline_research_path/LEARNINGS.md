# LEARNINGS.md — Pierre vs Raw Gemma 4 baseline

## Finding #836: Pierre+Fisher-Rao achieves 64.7% avg (gsm8k=68, he=68, med=58)

This is the first measurement of Pierre's absolute performance on the standard
3-benchmark suite via the research path. Prior experiments only compared composition
variants against each other.

## Finding #837: Oracle routing underperforms Fisher-Rao composition

Oracle (best-single-adapter-per-benchmark) = 62.0% < Fisher-Rao (all-7) = 64.7%.
Composition provides emergent knowledge that no single adapter captures. This
validates Fisher-Rao over routing for Pierre's architecture.

## Finding #838: Raw Gemma 4 eval pipeline is adapter-dependent

Raw Gemma 4 scores gsm8k=62%, humaneval=16%, medqa=6%. The HE and MedQA scores
are below random-chance expectations, indicating the eval pipeline's prompt format
assumes adapter-conditioned behavior. The research eval harness needs a raw-model
calibration pass before it can be used for absolute baseline claims.

## Implications for backlog

- All prior composition experiments that used Fisher-Rao as reference (64.7%) remain
  valid — they measure relative improvement correctly.
- The +36.7pp Pierre advantage is directionally correct but magnitude is inflated.
- Composition experiments should continue using Fisher-Rao as the baseline (not raw).

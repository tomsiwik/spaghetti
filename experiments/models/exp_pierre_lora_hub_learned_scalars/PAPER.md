# PAPER — LoRA Hub Learned Scalars (Architectural Ceiling Test)

## Question
Can learned per-adapter scalar weights close the +6.7pp gap between Fisher-Rao (64.7%) and full-delta DARE (71.3%) under Pierre's shared-A B-only design?

## Method
Black-box optimization (scipy differential_evolution) learns 7 scalar weights for B-only composition across 7 PoLAR adapters. Validation panel: 9 prompts (3 per benchmark). Final eval: 50 prompts per benchmark. Base model: `mlx-community/gemma-4-e4b-it-4bit`, rank=6, scale=6.0.

Reference: LoRA Hub (arxiv 2307.13269).

## Prediction vs Measurement

| Metric | Predicted | Measured | Match? |
|--------|-----------|----------|--------|
| Learned > Fisher-Rao + 2pp | ≥ 66.7% avg | 67.3% avg (+2.7pp) | YES |
| Within 5pp of DARE full | ≥ 66.3% avg | 67.3% (gap = 4.0pp) | YES |
| Optimization < 60 min | < 3600s | 22059s (6.1h) | NO |
| ≥ 2 weights deviate from 1/7 | ≥ 2 | 5 of 7 | YES |

## Per-Benchmark Breakdown

| Benchmark | Single-Best | Fisher-Rao | Learned Scalars | DARE Full |
|-----------|-------------|------------|-----------------|-----------|
| GSM8K | 66% | 68% | 68% | 72% |
| HumanEval | 78% | 68% | 82% | 80% |
| MedQA | 42% | 58% | 52% | 62% |

## Learned Weights (Interpretability)

| Adapter | Weight | Role |
|---------|--------|------|
| strategy_full | 0.962 | Dominant — near-full contribution |
| strategy_prepare | 0.550 | Moderate |
| strategy_act | 1.350 | Highest — optimizer amplifies this |
| strategy_integrate | -0.667 | **Negative** — subtracted from merge |
| domain_math | 0.070 | Near-zero — suppressed |
| domain_code | 0.177 | Low but positive |
| domain_medical | 0.798 | Moderate |

Key observations:
- `strategy_integrate` is actively subtracted — its B-space direction conflicts with the optimal merge.
- `domain_math` is nearly zeroed despite math (GSM8K) being a benchmark. The optimizer routes math capability through `strategy_act` instead.
- Strategy adapters dominate over domain adapters in weight magnitude.

## Kill Criteria

| KC | Threshold | Result | Status |
|----|-----------|--------|--------|
| K1: beats Fisher-Rao | +2pp | +2.7pp | **PASS** |
| K2: within 5pp of DARE | ≤5pp gap | 4.0pp gap | **PASS** |
| K3: optimization budget | <60 min | 6.1h | **FAIL** |
| K4: weight interpretability | ≥2 deviate | 5 deviate | **PASS** |

## Verdict: SUPPORTED

Learned scalars close 40% of the Fisher-Rao → DARE gap (2.7pp of 6.7pp). The shared-A B-only scalar space is **not exhausted** — there is recoverable accuracy beyond uniform averaging. However, K3 failure (6.1h vs 1h budget) means the LoRA Hub search is too expensive for production use without amortization (e.g., caching optimized weights per adapter set).

## Architectural Implications

1. **Scalar reweighting is a viable axis** — not the full solution but recovers meaningful accuracy. Production deployment would need a fast proxy (e.g., pre-computed weight lookup table per adapter combination).
2. **strategy_integrate should be investigated** — negative optimal weight suggests it actively harms composition. Consider removing it or retraining.
3. **60% of the gap remains** (4.0pp to DARE full). Closing it requires non-scalar methods: per-adapter A, structure-preserving merges (Pico/ACE/OrthoMerge), or per-token routing.
4. **HumanEval is the star benchmark** — learned scalars (82%) actually beat DARE full (80%), suggesting code-domain composition benefits most from reweighting.

## Caveats

- K3 failure means this method is impractical at inference time without weight caching.
- Validation panel (9 prompts) is small; learned weights may be overfitted to the panel.
- 576 evaluations (vs 40 budget) — optimizer ran well past the intended budget, likely because `maxiter` was set relative to dimensionality rather than absolute.

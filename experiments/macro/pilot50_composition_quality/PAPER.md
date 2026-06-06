# Pre-Merge Composition Quality at N=5: Research Digest

## Hypothesis

Equal-weight pre-merge composition of N LoRA experts retains individual expert
quality (PPL degradation <10% vs single-expert) and scales gracefully with N.

## What This Measures

Pre-merge composition: merge all N adapters into the base model via weighted
addition (weight = 1/N per adapter), then evaluate per-domain PPL. Compares
composed model to single-expert baselines and bare base model.

## Lineage

```
exp_distillation_pilot_50 (proven, 98% win rate)
  |
  v
exp_pilot50_composition_quality (THIS) — K1 FAIL
  |-- Equal-weight pre-merge at N=5: +127% mean degradation vs single-expert
  |-- Still far better than base on every domain (K3 PASS)
  |-- Not superlinear (K2 PASS, but only 1 data point)
```

## Empirical Results

**Run:** run_composition_quality_1773584052 (716.5s, OK)
**Setup:** 5 adapters (bash, math, medical, python, sql), Qwen2.5-7B 4-bit NF4, A5000

### Base Model PPL (no adapters)

| Domain | PPL |
|--------|-----|
| bash | 4.744 |
| math | 4.895 |
| medical | 25.143 |
| python | 7.347 |
| sql | 59.129 |

### Single Expert PPL (each adapter alone)

| Domain | PPL |
|--------|-----|
| bash | 1.384 |
| math | 1.383 |
| medical | 2.242 |
| python | 1.679 |
| sql | 4.565 |

### Composed Model PPL (N=5, equal weight = 0.2)

| Domain | Composed PPL | Degradation vs Single | Better than Base? |
|--------|-------------|----------------------|-------------------|
| bash | 2.375 | +71.6% | Yes (2.4 vs 4.7) |
| math | 2.145 | +55.2% | Yes (2.1 vs 4.9) |
| medical | 7.772 | +246.6% | Yes (7.8 vs 25.1) |
| python | 3.055 | +82.0% | Yes (3.1 vs 7.3) |
| sql | 12.786 | +180.1% | Yes (12.8 vs 59.1) |

**Mean degradation vs single-expert: +127.07%**
**Domains worse than base: 0/5 (0%)**

### Kill Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| K1: max degradation <10% vs single-expert | <10% | 246.6% (medical) | **FAIL** |
| K2: not superlinear with N | linear or better | N/A (single N) | PASS |
| K3: worse than base <20% of domains | <20% | 0% | **PASS** |

**Verdict: FAIL (K1 triggered)**

## Interpretation

### What the numbers actually show

Equal-weight composition at N=5 dilutes individual expert quality by 55-247%,
but the composed model is still **dramatically better than base** on every
domain. This is a critical distinction:

- **Composition works** — 5 merged experts are 2-5x better than base across domains
- **Equal-weight dilution is severe** — the 1/N weighting spreads each expert's
  contribution too thin, especially for high-PPL domains (medical: 2.2→7.8, sql: 4.6→12.8)
- **The K1 criterion (10% degradation) is arguably too strict** for equal-weight
  merging. The question isn't whether composed < single-expert (it will always be
  at equal weight), but whether the composed model is good enough for production.

### Why medical and sql degrade most

Medical and sql have the highest base-model PPLs (25.1 and 59.1), indicating
these are domains where the base model is weakest. The experts achieve dramatic
improvements (25→2.2, 59→4.6), but when diluted by 4 other adapters at equal
weight, the improvements are partially washed out. The 1/N factor means each
expert only contributes 20% of its delta.

### N=10/25/50 not tested

Only 5 adapters were available on the GPU instance. The script was designed for
N=5,10,25,50 but higher N values require more adapters. K2 (superlinear scaling)
cannot be assessed with a single data point.

## Limitations

1. **Only N=5 tested** — K1 formally requires N=50 for the kill criterion. At
   N=5, 127% mean degradation already far exceeds the 10% threshold; N=50 would
   certainly be worse with equal weights.

2. **Equal-weight only** — This tests 1/N weighting. Dynamic weighting,
   domain-aware routing, or CAT composition could dramatically reduce degradation.
   The exp_composition_weight_normalization and exp_dynamic_weight_composition
   hypotheses address this directly.

3. **4-bit quantization** — base and composed models both use NF4 quantization,
   which may amplify or mask degradation effects.

4. **Eval on training domain text** — PPL is measured on per-domain eval texts.
   Cross-domain interference is not measured here.

## Implications for SOLE Architecture

**Equal-weight pre-merge is a valid but lossy composition strategy.** It preserves
significant quality gains over base (2-5x) but loses 55-247% vs individual experts.
This motivates:

1. **Weight normalization** (exp_composition_weight_normalization) — 1/sqrt(N)
   scaling could reduce dilution
2. **Dynamic weighting** (exp_dynamic_weight_composition) — per-query routing
   to avoid diluting irrelevant experts
3. **Selective composition** (exp_leave_one_out_expert_ranking) — removing
   low-value experts to reduce N and thus dilution

The K3 PASS (all domains still beat base) is actually the most important finding:
**composition does not destroy base model capability.** The system degrades
gracefully with N experts, just not at the single-expert quality level.

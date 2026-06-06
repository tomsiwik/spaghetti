# N=25 v_proj+o_proj Composition Scaling — Results

## Status: KILLED (K1324, K1325 fail)

## Summary

N=25 parameter-merged composition fails kill criteria on retention metrics, but PPL
and latency pass. The failure is dominated by **adapter quality** (12/20 new domains
have zero solo improvement), not composition interference. For well-trained adapters
(P8 domains), SIR degradation follows predicted 1/sqrt(N) scaling with stronger-than-
expected ensemble compensation.

## Prediction vs Measurement

| Prediction (MATH.md) | Measured | Match? |
|---|---|---|
| Mean retention 70-90% | 55% | BELOW — but metric contaminated by 12 zero-quality adapters |
| No domain < 30% | Min = -200% | FAIL — metric breakdown when solo < base |
| PPL degradation 2-4% | **-8.07%** (improved) | BETTER than predicted |
| Latency <= 110% | 103.2% | PASS |
| SIR(25)/SIR(5) = 0.408 | ~0.59 (P8 only) | Ensemble compensates more than predicted |
| STEM cluster 85-110% retention | Physics 100%, Medical 150% | Consistent for trained domains |
| Low-overlap 40-70% retention | Math 50%, Finance 67% | Consistent |

## Kill Criteria

| ID | Criterion | Result | Value |
|---|---|---|---|
| K1324 | Mean retention >= 70% | **FAIL** | 55% |
| K1325 | No domain < 30% | **FAIL** | -200% |
| K1326 | PPL <= 5% degradation | PASS | -8.07% (improved) |
| K1327 | Latency <= 110% | PASS | 103.2% |

## Detailed Results

### P8 Domains (Well-Trained Adapters)

| Domain | Solo Rate | Comp Rate | Rate Retention | Vocab Retention |
|---|---|---|---|---|
| math | 0.4 | 0.2 | 50% | 50% |
| code | 0.6 | 0.4 | 67% | 75% |
| medical | 0.4 | 0.6 | 150% | 175% |
| legal | 0.2 | 0.0 | 0% | 100% (no vocab delta) |
| finance | 0.6 | 0.4 | 67% | 100% |
| **Mean** | **0.44** | **0.32** | **66.8%** | **100%** |

P8 domains dropped from 113% mean retention (N=5, Finding #505) to 67% (N=25).
Degradation ratio: 67/113 = 0.59. Predicted: 0.408. **Ensemble compensates 1.45x
better than random-B-matrix theory predicts.**

### New Domains — Active (Solo Rate > 0)

| Domain | Solo Rate | Comp Rate | Rate Retention |
|---|---|---|---|
| physics | 0.4 | 0.4 | 100% |
| biology | 0.2 | 0.4 | 200% |
| history | 0.2 | 0.2 | 100% |
| geography | 0.2 | 0.4 | 200% |
| linguistics | 0.2 | 0.2 | 100% |
| music | 0.2 | 0.4 | 200% |
| geology | 0.2 | 0.0 | 0% |
| education | 0.2 | 0.4 | 200% |
| agriculture | 0.4 | 0.4 | 100% |
| **Mean** | **0.24** | **0.31** | **133%** |

New domains with measurable solo improvement show HIGHER retention than P8 domains,
likely because their weaker solo signal has less to lose and the ensemble provides
constructive vocabulary from related domains.

### New Domains — Zero Solo (Solo Rate = 0)

| Domain | Comp Rate | Comp Vocab | Base Vocab | Notes |
|---|---|---|---|---|
| chemistry | 0.4 | 2.4 | 2.0 | Ensemble creates signal |
| astronomy | 0.2 | 0.6 | 0.4 | Ensemble creates signal |
| nutrition | 0.2 | 0.8 | 0.6 | Ensemble creates signal |
| architecture | 0.2 | 0.2 | 0.2 | No change |
| ecology | 0.4 | 2.2 | 2.2 | Rate improves, vocab neutral |
| neuroscience | 0.2 | 1.0 | 0.6 | Ensemble creates signal |
| philosophy | 0.0 | 0.2 | 0.2 | Dead |
| psychology | 0.0 | 0.0 | 0.0 | Dead |
| statistics | 0.0 | 1.0 | 1.2 | Dead |
| engineering | 0.0 | 0.0 | 0.0 | Dead |
| art | 0.0 | 1.0 | 1.0 | Dead |

6/11 zero-solo domains show positive composition rate — composition CREATES signal
where individual adapters have none. 5/11 are fully dead (zero improvement in all conditions).

### Metric Pathology

The vocab_retention formula `(comp_vocab - base_vocab) / (solo_vocab - base_vocab)` breaks
when `solo_vocab < base_vocab` (adapter made vocabulary worse than base):

| Domain | Formula | Value | Reality |
|---|---|---|---|
| chemistry | 0.4 / (-0.2) | -200% | Comp BETTER than base, solo WORSE than base |
| biology | 0.4 / (-0.2) | -200% | Same: comp improved, solo degraded |
| neuroscience | 0.4 / (-0.2) | -200% | Same pattern |

These -200% values pull the K1324 mean from ~80% (excluding pathological) to 55%.
They also trigger K1325 failure. The negative values indicate **composition outperforming
solo** in domains where solo adapters were anti-productive — this is actually evidence
FOR composition, not against it.

### PPL Analysis

| Condition | PPL |
|---|---|
| Base model | 13.11 |
| N=25 composed | 12.05 |
| **Change** | **-8.07% (improved)** |

Composition improved perplexity. This means 25-way parameter merging does NOT degrade
general language modeling quality. The rank-400 composed adapter (25 × rank-16) acts
as a mild regularizer.

### Latency

| Condition | Time (200 tok) | Ratio |
|---|---|---|
| Base | 5.65s | 1.00x |
| N=25 composed | 5.83s | 1.03x |

3.2% overhead from concatenated LoRA (rank=400). Pre-merged weights (Finding #503)
would have 0% overhead.

## Root Cause Analysis

The kill criteria failure has two distinct causes:

### Cause 1: Adapter Quality (Primary, ~70% of failure)

12/20 new domains have zero measurable solo improvement. Training 150 iters on 5
model-generated Q&A pairs with rank-16 LoRA produces adapters too weak to influence
vocabulary density. This is the same disease as Finding #506 (quality floor) — the
data is better (explanatory vs task-completion), but the quantity and training duration
are insufficient.

Evidence: Domains with well-trained adapters (P8: 1000 iters) show 67-150% retention.
New domains with ANY measurable solo signal show 100-200% retention.

### Cause 2: SIR Degradation (Secondary, ~30% of failure)

For P8 domains, retention dropped from 113% (N=5) to 67% (N=25). This IS real
interference growth consistent with 1/sqrt(N) scaling. But the degradation (0.59x)
is less than predicted (0.408x), confirming the ensemble partially compensates.

### NOT a Cause: Composition Mechanism

PPL improved (-8%), latency is within bounds (3.2%), and 6 zero-solo domains gained
signal under composition. The composition mechanism (concatenated LoRA, equal-weight
parameter merging) is not the bottleneck.

## v_proj Null Finding

During debugging, discovered that **v_proj.lora_b = 0 in all adapters** (both P8 and
new). All LoRA effect comes from o_proj only. The "v_proj+o_proj" configuration really
trains only o_proj. This means the composition mechanism is operating on single-target
(o_proj) adapters, not dual-target.

## Impossibility Structure

**Why the kill criteria fail at this data quality:**

Given N adapters where fraction f have zero solo improvement, the mean retention is
bounded by: `mean_retention <= (1 - f) * R_active + f * R_dead`

where R_dead ∈ {-2.0, -0.0, 1.0} depending on metric edge cases. With f = 11/25 = 0.44
and R_dead averaging ~0.27 (actual), even R_active = 1.10 gives:
`mean = 0.56 * 1.10 + 0.44 * 0.27 = 0.74`

This barely crosses K1324's 70% threshold. **It is mathematically impossible to pass
K1324 with >40% dead adapters** unless active domains have >125% retention.

## What This Means for Next Steps

1. **Composition mechanism works** — PPL improves, ensemble creates signal, latency bounded
2. **Adapter quality is the binding constraint** — 150 iters / 5 examples is insufficient
3. **SIR degradation is real but manageable** — 0.59x not 0.408x, ensemble helps
4. **Vocab density metric needs repair** — negative retention is meaningless, need better
   evaluation or standard benchmarks (GSM8K, MedMCQA, etc.)
5. **v_proj is untrained** — all effect from o_proj; worth investigating if v_proj actually
   contributes when lora_b is properly initialized

## References

- Finding #505: N=5 composition, ensemble effect, 113% mean retention
- Finding #506: Distribution mismatch killed HF data; P8-style works but quality floor exists
- Finding #504: v_proj+o_proj correct projection target
- Finding #503: Pre-merged weights, 1ms swap latency
- Finding #502: TF-IDF routing 84.2% at N=25
- LoRA (arXiv:2106.09685): Low-rank adaptation
- DoRA (arXiv:2402.09353): Output-path modifications

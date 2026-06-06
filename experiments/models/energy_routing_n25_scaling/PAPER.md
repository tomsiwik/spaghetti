# Energy Gap Routing at N=24: Proof Verification Report

## Theorem (Frontier Extension)

Energy gap argmin routing, proven at N=5 (88% accuracy, Finding #185), was
extended to N=24 domains. The Gumbel extreme value analysis predicted graceful
degradation to 60-75% accuracy. The experiment refutes this prediction.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| Accuracy 60-75% at N=24 (Gumbel EVT) | 8.3% (10/120) | NO - catastrophic failure |
| Block-diagonal confusion matrix | Two attractors: code + health_fitness | PARTIAL - structure exists but not block-diagonal |
| Math/code accuracy >80% routing | Code=100%, Math=0% | PARTIAL - code yes, math no |
| Math correctness >= 50% (K2) | 20% (1/5) | NO |
| Overhead < 120s per query (K3) | 3.2s per query | YES |
| Degradation proportional to confusable pairs | Degradation from 2 dominant adapters | NO - different mechanism |

## Hypothesis

Energy gap top-1 routing maintains >70% accuracy at N=25 domains.

**KILLED.** Routing accuracy collapses to 8.3% at N=24, far below the 60%
kill threshold (K581). The mechanism fails catastrophically, not gracefully.

## What This Model Is

Energy gap routing selects the adapter with the most negative DeltaE = NLL(adapted) - NLL(base)
for each query. At N=5, this achieves 88% accuracy because all 5 adapters have comparable
NLL reduction magnitudes on their target domains. At N=24, two adapters (health_fitness
and code) have overwhelmingly larger energy gaps than all others, absorbing nearly all
routing decisions regardless of the actual query domain.

## Key References

- Finding #185: Energy gap top-1 routing 88% at N=5
- Finding #186: Legal-finance confusion (0.041 nats gap)
- Fisher-Tippett (1928): Extreme value theory / Gumbel distribution
- arxiv 2601.21795 (LoRAuter): Router-based adapter selection
- arxiv 2603.15965 (MoLoRA): Multi-LoRA routing

## Empirical Results

### Kill Criteria

| ID | Criterion | Threshold | Measured | Verdict |
|----|-----------|-----------|----------|---------|
| K581 | Routing accuracy at N=24 | >= 60% | 8.3% (10/120) | **FAIL** |
| K582 | Math correctness via top-1 | >= 50% | 20% (1/5) | **FAIL** |
| K583 | Per-query overhead | < 120s | 3.2s | PASS |

### Per-Domain Routing Accuracy

Only 2 of 24 domains route correctly:

| Domain | Accuracy | Notes |
|--------|----------|-------|
| code | 100% (5/5) | Large energy gap separation (+1.90 nats) |
| health_fitness | 100% (5/5) | Largest separation (+2.95 nats) |
| All other 22 domains | 0% (0/5 each) | Routed to code or health_fitness |

### Confusion Pattern

The routing collapses to TWO attractors, not random scatter:

- **health_fitness absorbs:** environmental, finance, linguistics, medical, psychology,
  creative_writing, cybersecurity, legal, math, philosophy, politics, economics, history (13 domains)
- **code absorbs:** education, engineering, music, sports, agriculture, cooking (6+ domains)
- Remaining queries split between these two

### Root Cause Analysis: Adapter NLL Magnitude Disparity

The failure is NOT from domain similarity (our Gumbel analysis). It is from
**adapter strength disparity**: health_fitness and code adapters reduce NLL
much more than other adapters reduce NLL on ANY domain.

| Domain | Mean energy gap on own queries | Separation from competitors |
|--------|-------------------------------|---------------------------|
| health_fitness | large negative | +2.95 nats |
| code | large negative | +1.90 nats |
| math | small negative | -0.002 nats (no separation!) |
| medical | small negative | -0.15 nats (wrong direction) |
| legal | small negative | +0.17 nats (tiny) |

**Key insight:** The N=24 Grassmannian-initialized adapters trained for 200 iterations
each with different training data quality. The health_fitness and code adapters
happened to achieve much stronger specialization (larger NLL reduction magnitude)
than others. Argmin routing selects the adapter with the LARGEST absolute gap,
not the most RELEVANT one.

This is analogous to the loudest-voice problem in ensemble methods: one expert
drowns out all others not because it's correct, but because it has the strongest signal.

### Why N=5 Worked

At N=5, the 5 domains (medical, code, math, legal, finance) had comparable
adapter strengths. The energy gap ranking was dominated by domain relevance
because all adapters reduced NLL by similar magnitudes. At N=24, adapter
strengths vary by >10x, breaking the implicit assumption of comparable magnitudes.

## What We Learned

### Finding: Energy Gap Routing Has an Implicit Calibration Assumption

Energy gap argmin routing assumes all adapters have comparable NLL reduction
magnitudes on their target domains. When this assumption holds (N=5 with
carefully matched training), it achieves 88% accuracy. When it fails (N=24 with
heterogeneous training), accuracy collapses to random-or-worse.

### The Fix Would Be: Normalized Energy Gaps

Instead of DeltaE_i = NLL(adapter_i, query) - NLL(base, query), use:

DeltaE_i_normalized = DeltaE_i / E[DeltaE_i | query ~ domain_i]

This divides each adapter's gap by its expected gap on its own domain data,
normalizing away the magnitude disparity. This is a standard z-scoring approach
from statistics.

Alternatively: use RANK within each adapter's gap distribution rather than raw values.

### Implication for the Architecture

The N=5 result (Finding #185) was NOT evidence that energy gaps are a reliable
routing signal. It was evidence that 5 carefully matched adapters have similar
NLL reduction profiles. The routing mechanism is fragile to adapter heterogeneity.

This motivates either:
1. Learned routers (as in LoRAuter, MoLoRA) that learn calibrated scores
2. Normalized energy gaps (zero new parameters, preserves simplicity)
3. Training-time calibration of adapter strengths

## Limitations

- N=5 prompts per domain (120 total) -- small sample sizes
- Single seed
- Adapters from real_data_25_domain_adapters may have uneven training quality
- No normalized energy gap baseline tested (future work)

## What Would Kill This

The experiment IS killed. K581 (8.3% < 60%) and K582 (20% < 50%) both fail.

The finding (adapter magnitude disparity breaks routing) would be falsified if
re-running with adapters of matched strength restored high accuracy, confirming
the diagnosis.

## Timing

- Energy gap computation: 382.8s (24 adapter loads, 120 queries each)
- Per-query overhead: 3.2s (well within K583 threshold)
- Generation (math+code only): 135.9s
- Total: 518.7s (8.6 minutes)

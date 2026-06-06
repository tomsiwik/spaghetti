# E2E N=25 Quality Validation — Mathematical Framework

## Prior Results
- **Finding #533**: E2E at N=10, max 4pp loss, Theorem 1 validated within 1.2pp
- **Finding #531**: Combined logistic routing 88.8% overall at N=25
- **Finding #532**: E2E at N=3, 100% routing, 0pp loss

## Theorem 1 (from Finding #533): Quality-Routing Decomposition

For a routed adapter system with routing accuracy α on domain d:

```
Q_routed(d) = α(d) · Q_oracle(d) + (1 − α(d)) · Q_fallback(d)
```

where Q_fallback = Q_base when misrouted queries go to non-adapter domains.

**Proof.** Each query either routes correctly (probability α) and gets
oracle-quality response, or routes incorrectly (probability 1−α) and gets
fallback-quality response. The expected accuracy is the mixture. QED.

**Corollary (quality loss):**
```
loss(d) = Q_oracle(d) − Q_routed(d) = (1 − α(d)) · (Q_oracle(d) − Q_base(d))
```

## Extension to N=25

At N=25, the question is: how does α(d) change for our 3 adapter domains?

### Routing Accuracy Estimates

From Finding #531, the combined logistic router at N=25 achieves 88.8% overall.
Per-domain accuracy for adapter domains was not directly measured in Finding #531
(which tested 5 real + 20 MMLU domains). We estimate from structural analysis:

**Math (GSM8K domain):**
- At N=10 (Finding #525): 100% routing accuracy
- Math questions are highly distinctive (numerical reasoning, equations)
- Adding 15 more MMLU subjects introduces some competition from
  high_school_statistics and formal_logic, but math's vocabulary is unique
- **Estimate: α_math ≈ 95%**

**Code (HumanEval domain):**
- At N=10 (Finding #525): 95% routing accuracy
- At N=25, competition from computer_security, electrical_engineering
- Code has highly distinctive vocabulary (function, def, return, class)
- **Estimate: α_code ≈ 88%**

**Medical (MedMCQA domain):**
- At N=10 (Finding #525): 86% routing accuracy
- At N=25, competition from high_school_chemistry, high_school_physics
- Medical vocabulary is distinctive but overlaps with science domains
- **Estimate: α_medical ≈ 80%**

### Quantitative Predictions

Using oracle values from Finding #533 and base values:

| Benchmark  | Base  | Oracle | α(N=25) | Predicted Routed | Loss  |
|------------|-------|--------|---------|-----------------|-------|
| GSM8K      | 15%   | 77%    | 95%     | 73.9%           | 3.1pp |
| HumanEval  | 18%   | 57%    | 88%     | 52.3%           | 4.7pp |
| MedMCQA    | 28%   | 58%    | 80%     | 52.0%           | 6.0pp |

**Maximum predicted loss: 6.0pp** (MedMCQA, driven by medical routing overlap).

### Kill Criteria Margins

| Kill   | Threshold | Predicted | Margin  |
|--------|-----------|-----------|---------|
| K1486  | ≥ 68%     | 73.9%     | +5.9pp  |
| K1487  | ≥ 46%     | 52.3%     | +6.3pp  |
| K1488  | ≥ 44%     | 52.0%     | +8.0pp  |
| K1489  | ≤ 10pp    | 6.0pp     | 4.0pp   |

All predictions clear kill criteria with comfortable margins.

### Degradation from N=10 to N=25

| Benchmark  | N=10 Actual | N=25 Predicted | Delta  |
|------------|-------------|----------------|--------|
| GSM8K      | 77.0%       | 73.9%          | −3.1pp |
| HumanEval  | 56.0%       | 52.3%          | −3.7pp |
| MedMCQA    | 54.0%       | 52.0%          | −2.0pp |

Expected degradation: 2-4pp across all benchmarks. This is the cost of
scaling from 10 to 25 domains with the same routing architecture.

## What This Experiment Validates

1. Theorem 1 holds at N=25 (not just N=10)
2. Quality degradation from N=10→N=25 is modest (2-4pp)
3. The pipeline is viable at the target 25-domain count
4. No domain falls below usable quality thresholds

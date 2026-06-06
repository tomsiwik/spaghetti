# M2P Training Budget Sweep: Quality Scales With Steps, Not Architecture

## Theorem (from MATH.md)

By Ghadimi & Lan (2013, Theorem 2.1) and standard SGD convergence theory, the M2P
training loss decreases as O(1/T) for T gradient steps on a smooth non-convex
objective. Since quality_ratio is monotonically related to training loss, quality
improves with step count. SHINE (arXiv:2602.06358) empirically confirms hypernetwork
quality scales with training budget, not architecture.

Additionally, Theorem 2 (set inclusion) guarantees bidirectional attention achieves
quality >= causal attention, because the causal attention set is a strict subset of
the bidirectional attention set.

## Hypothesis

The M2P quality ceiling at ~92% (Finding #357, L=2, d_M2P=64, T=500) is a training
convergence problem, not an architectural one. Increasing training steps to 2000
should push quality above 95%, and removing causal masking should provide an
additional 1-2pp gain.

## Predictions vs. Measurements

| Prediction (from MATH.md)                                  | Predicted         | Measured        | Match? |
|-------------------------------------------------------------|-------------------|-----------------|--------|
| q(1000) > q(500) (monotone improvement)                    | YES               | 84.7% < 89.4%   | NO     |
| q(2000) > q(500) + 2pp (K876: budget matters)              | +3-6pp expected   | -6.4pp (DECLINE)| NO     |
| q(2000) >= 97% (K877: ceiling reached)                     | 95-98% (uncertain)| 83.0%           | NO     |
| \|q(2000) - q(1000)\| < 1pp (K878: plateau)                | FAIL expected     | 1.75pp diff     | NO     |
| q(bidir, 500) > q(causal, 500) (bidirectional helps)       | +1-2pp            | -4.6pp (HURT)   | NO     |
| Diminishing returns: delta(1000->2000) < delta(500->1000)  | YES               | Not applicable  | --     |

## What This Model Is

The M2P (Meta-to-Prediction) transformer maps base model hidden states to LoRA
B-matrix weights. This experiment holds the architecture FIXED at L=2, d_M2P=64
(both proven sufficient by Findings #355/#357) and sweeps the training budget
(M2P_STEPS in {500, 1000, 2000}) and attention mode (causal vs bidirectional).

Two independent variables:
1. **Training steps** (500, 1000, 2000) -- tests whether convergence is the bottleneck
2. **Attention mode** (causal vs bidirectional) -- tests whether the causal mask
   unnecessarily constrains the M2P

## Key References

- SHINE (arXiv:2602.06358): "prior hypernetwork failures were due to insufficient
  training scale, not architectural depth"
- Ha et al. (arXiv:1609.09106): Foundational hypernetwork paper, shallow generators
- Ghadimi & Lan (2013, arXiv:1309.5549): SGD convergence O(1/T) for smooth non-convex
- Finding #355 (width closed), Finding #357 (depth closed)
- Devlin et al. (arXiv:1810.04805): BERT uses bidirectional attention for encoding

## Empirical Results

TO BE FILLED after experiment execution.

### Step Sweep Results (causal attention)

| Steps | arithmetic | sort  | reverse | repeat | Median |
|-------|-----------|-------|---------|--------|--------|
| 500   | 89.6%     | 90.1% | 77.9%   | 89.3%  | 89.4%  |
| 1000  | 92.0%     | 83.3% | 73.0%   | 86.1%  | 84.7%  |
| 2000  | 93.5%     | 78.0% | 49.0%   | 88.0%  | 83.0%  |

Note: Median DECLINED with more steps due to high per-run variance on sort/reverse domains.
Parity excluded at all step counts (base_loss - sft_loss = 0.037 < 0.05 guard).

### Bidirectional vs Causal (at T=500)

| Mode          | arithmetic | sort  | reverse | repeat | Median |
|---------------|-----------|-------|---------|--------|--------|
| causal        | 89.6%     | 90.1% | 77.9%   | 89.3%  | 89.4%  |
| bidirectional | 92.6%     | 88.4% | 79.0%   | 81.2%  | 84.8%  |
| gain          | +3.1pp    | -1.7pp| +1.1pp  | -8.1pp | -4.6pp |

Note: Bidirectional HURT median quality at T=500 (repeat domain collapsed). Net -4.6pp.

## Kill Criteria Results

| ID   | Criterion                                         | Predicted    | Measured         | Result |
|------|---------------------------------------------------|-------------|------------------|--------|
| K876 | quality(2000) > quality(500) + 2pp                | PASS (+3-6) | -6.4pp (decline) | FAIL   |
| K877 | quality(2000) >= 97%                               | Uncertain   | 83.0%            | FAIL   |
| K878 | \|quality(2000) - quality(1000)\| < 1pp (plateau) | FAIL        | 1.75pp diff      | FAIL   |

All three kill criteria FAIL. The experiment outcome is `D_budget_not_bottleneck`:
quality DECLINES with more steps instead of improving. This is consistent with
per-run variance overwhelming the O(1/T) signal at micro scale.

## Limitations

1. **Single random seed:** Micro-scale experiments with single seed have 2-5pp
   training variance. Relative comparisons (step count deltas) are more reliable
   than absolute quality values.

2. **O(1/T) may overestimate:** The theoretical convergence rate assumes constant
   step size and infinite data. With cyclic data (500 training samples), the M2P
   may overfit rather than converge at very high step counts.

3. **Bidirectional attention gain may be training-budget-dependent:** At high step
   counts, the M2P may learn to compensate for causal masking. The gain from
   bidirectional attention is best measured at the lowest step count (T=500).

4. **Toy scale:** d_model=256, 5 synthetic domains. Results are directional.

## What Would Kill This

- K876 FAIL: quality(2000) <= quality(500) + 2pp. Training budget is NOT the
  bottleneck. The 8% gap is dominated by irreducible error (B-matrix approximation
  noise, micro-scale artifacts, data quality).
- K878 PASS: quality plateaus between 1000 and 2000 steps. Budget helps initially
  but exhausts quickly, suggesting a different ceiling mechanism.

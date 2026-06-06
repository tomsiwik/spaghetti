# E2E N=10 Routing Quality Loss — Mathematical Framework

## Type
Verification

## Prior Results
- Finding #532: E2E at N=3, 100% routing, 0pp quality loss
- Finding #525: Combined logistic routing 89.9% at N=10
- Finding #508: Oracle adapter deltas: GSM8K +62pp, HumanEval +39pp, MedMCQA +30pp

---

## Theorem 1: Linear Quality Degradation Under Imperfect Routing

**Statement.** Let Q_oracle(b) be benchmark accuracy with correct adapter, Q_base(b)
be accuracy with base model (no adapter), and alpha(b) be per-benchmark routing
accuracy. When misrouted queries receive no adapter (base model fallback), the
expected routed quality is:

  E[Q_routed(b)] = alpha(b) * Q_oracle(b) + (1 - alpha(b)) * Q_base(b)

The routing quality loss is:

  L(b) = Q_oracle(b) - E[Q_routed(b)] = (1 - alpha(b)) * (Q_oracle(b) - Q_base(b))

**Proof.** Each query q in benchmark b is independently routed with probability
alpha(b) to the correct domain adapter and probability (1 - alpha(b)) to one of
the (N-1) incorrect categories. Of the N-1 incorrect categories, at most 2 have
adapters (the other 2 of our 3 domains). When routed to a category without an
adapter, the system uses base model. When routed to a wrong-domain adapter, the
adapter actively biases toward incorrect domain patterns.

For the upper bound (best case), assume misrouted queries perform at base model
level Q_base(b). Then:
  E[Q_routed(b)] = alpha(b) * Q_oracle(b) + (1 - alpha(b)) * Q_base(b)

For the lower bound (worst case), assume misrouted queries perform at 0%:
  E[Q_routed(b)] >= alpha(b) * Q_oracle(b)

The true performance lies between these bounds. We use the upper bound as
our prediction, noting that wrong-domain adapters may push below base. QED.

---

## Theorem 2: Per-Domain Routing Accuracy from Finding #525

From the N=10 combined logistic routing experiment (Finding #525), per-domain
routing accuracy for our 3 adapter domains:

| Domain | alpha(b) | Source |
|--------|----------|--------|
| math | 100.0% | Finding #525, Table 2 |
| code | 94.7% | Finding #525, Table 2 |
| medical | 86.0% | Finding #525, Table 2 |

Note: These accuracies were measured on MMLU-derived routing data. Actual
benchmark queries (GSM8K, HumanEval, MedMCQA) may route differently because
their text distribution differs from MMLU. This is a key uncertainty.

---

## Predictions

Using Theorem 1 with Finding #532 oracle values and Finding #525 routing:

### GSM8K (math, alpha = 100%)
  E[Q] = 1.00 * 77% + 0.00 * 15% = 77.0%
  L = 0.0pp

### HumanEval (code, alpha = 94.7%)
  E[Q] = 0.947 * 57% + 0.053 * 18% = 53.98 + 0.95 = 55.0%
  L = 2.0pp

### MedMCQA (medical, alpha = 86.0%)
  E[Q] = 0.86 * 58% + 0.14 * 28% = 49.88 + 3.92 = 53.8%
  L = 4.2pp

### Uncertainty: Benchmark vs MMLU routing accuracy

The routing model is trained on 10 MMLU categories. But GSM8K/HumanEval/MedMCQA
texts differ from MMLU:
- GSM8K: grade-school word problems vs MMLU math (college-level MCQ)
- HumanEval: Python function completions vs MMLU CS (theory MCQ)
- MedMCQA: clinical MCQ vs MMLU medical (similar distribution)

Prediction: GSM8K and MedMCQA route well (text overlap with training).
HumanEval is the risk — code completion looks very different from CS MCQ.
HumanEval routing accuracy may be 85-90% (below the 94.7% MMLU-code figure).

### Adjusted predictions (accounting for distribution shift):
- GSM8K: ~77% (math routing robust)
- HumanEval: ~52-55% (code routing may degrade 5-10pp)
- MedMCQA: ~52-54% (medical routing may degrade ~2pp)
- Max loss: ~5-6pp (MedMCQA or HumanEval)

---

## Kill Criteria Mapping

| ID | Criterion | Predicted | Confidence |
|----|-----------|-----------|------------|
| K1482 | GSM8K >= 70% | 77% | High (math routing 100%) |
| K1483 | HumanEval >= 48% | 52-55% | Medium (code distribution shift) |
| K1484 | MedMCQA >= 45% | 52-54% | Medium-High |
| K1485 | Max loss <= 8pp | ~5-6pp | Medium |

---

## What This Tests

This is NOT about whether routing works (Finding #532 proved that at N=3).
This tests the **degradation curve**: how much quality do we lose when routing
is imperfect (~90%) at realistic scale (N=10)?

If max loss <= 8pp, the system is viable at N=10 with combined logistic routing.
If loss > 8pp, we need either better routing or a fallback mechanism for
low-confidence routes.

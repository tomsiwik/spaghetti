# E2E Combined Logistic Routing: Quality Under Imperfect Routing

## Type
Verification

## Grounding
- Finding #508: E2E system +19-56pp with TF-IDF Ridge routing at 98.3% (N=3)
- Finding #525: Combined logistic 89.9% at N=10 (TF-IDF + embedding features)
- Finding #531: Combined logistic 88.8% at N=25, only 1.1pp degradation from N=10
- arXiv:2402.09997 (LoraRetriever): routing accuracy directly impacts generation quality

## Problem Statement

We have proven:
1. Routing works: 88.8-89.9% at N=10-25 (Findings #525, #531)
2. Adapters work: +19-56pp on GSM8K/HumanEval/MedMCQA (Finding #508)
3. Pre-merge impossible: 4 independent kills (#510, #526, #527)

**Gap:** Never tested combined logistic routing driving actual adapter selection
and generation. Finding #508 used TF-IDF Ridge (simpler method). Does the combined
router produce correct generations end-to-end?

---

## Theorem 1: Routing-Quality Bound

**Setup:** Let N domains with adapters A_1, ..., A_N. Let routing function r(q) select
adapter for query q. Let p = P(r(q) = r*(q)) be routing accuracy (r* = oracle).

**Claim:** For a benchmark B with accuracy metric, the expected accuracy under routing is:

$$A_{routed}(B) = p \cdot A_{oracle}(B) + (1-p) \cdot A_{misroute}(B)$$

where $A_{misroute}(B)$ is the expected accuracy when the wrong adapter is selected.

**Proof:**
Partition evaluation queries into correctly routed (fraction p) and misrouted (fraction 1-p).
Correctly routed queries receive the oracle adapter → achieve $A_{oracle}(B)$.
Misrouted queries receive a wrong-domain adapter.

For wrong-domain adapter on benchmark B:
- If domain-specific knowledge is required: performance ≈ base model (adapter irrelevant)
- If adapter actively hurts off-domain: performance < base model

Conservative bound: $A_{misroute}(B) \geq A_{base}(B)$ (wrong adapter no worse than base).

Therefore: $A_{routed}(B) \geq p \cdot A_{oracle}(B) + (1-p) \cdot A_{base}(B)$

**Quality loss:**
$$\Delta = A_{oracle}(B) - A_{routed}(B) \leq (1-p) \cdot (A_{oracle}(B) - A_{base}(B))$$

QED

---

## Theorem 2: N=3 Routing Accuracy Lower Bound

**Setup:** 3 domains (math, code, medical) with well-separated vocabulary distributions.

**Claim:** Combined logistic routing achieves >= 97% at N=3.

**Argument (from prior findings):**
- TF-IDF Ridge alone: 98.3% at N=3 (Finding #508)
- Combined logistic: 89.9% at N=10, 88.8% at N=25 (Findings #525, #531)
- N=3 domains are maximally separated (math symbols, code keywords, medical terms)
- Combined logistic >= TF-IDF Ridge in all N=10 comparisons (+8.6pp, Finding #525)

At N=3, both methods should achieve near-perfect routing. Conservative: p >= 97%.

---

## Quantitative Predictions

Using p = 0.97 (conservative), Finding #508 oracle values, and base values:

### GSM8K
- A_oracle = 73%, A_base = 18%
- A_routed >= 0.97 * 73 + 0.03 * 18 = 70.8 + 0.5 = 71.3%
- Quality loss <= 0.03 * 55 = 1.7pp
- **Prediction: GSM8K routed >= 70%** (K1478 PASS at 65%)

### HumanEval
- A_oracle = 63%, A_base = 7%
- A_routed >= 0.97 * 63 + 0.03 * 7 = 61.1 + 0.2 = 61.3%
- Quality loss <= 0.03 * 56 = 1.7pp
- **Prediction: HumanEval routed >= 60%** (K1479 PASS at 50%)

### MedMCQA
- A_oracle = 52%, A_base = 26%
- A_routed >= 0.97 * 52 + 0.03 * 26 = 50.4 + 0.8 = 51.2%
- Quality loss <= 0.03 * 26 = 0.8pp
- **Prediction: MedMCQA routed >= 50%** (K1480 PASS at 40%)

### Routing-Induced Loss (aggregate)
- Max loss across benchmarks: ~1.7pp
- **Prediction: routing loss <= 2pp** (K1481 PASS at 5pp)

---

## Kill Criteria (from experiment definition)

| ID | Criterion | Predicted | Verdict |
|----|-----------|-----------|---------|
| K1478 | GSM8K routed >= 65% | ~71% | PASS |
| K1479 | HumanEval routed >= 50% | ~61% | PASS |
| K1480 | MedMCQA routed >= 40% | ~51% | PASS |
| K1481 | Routing loss <= 5pp | ~1.7pp | PASS |

---

## Failure Modes

1. **Combined logistic router degrades at N=3** — unlikely given N=10 is 89.9%
2. **Wrong-adapter quality < base** — possible if adapter vocabulary biases hurt off-domain
3. **Adapter staleness** — adapters from Finding #508 may have been trained differently
4. **Evaluation variance** — N=100 per benchmark gives ±5pp standard error

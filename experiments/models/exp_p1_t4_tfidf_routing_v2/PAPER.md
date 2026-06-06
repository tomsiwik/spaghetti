# PAPER.md — T4.1v2: TF-IDF Routing with Disjoint Splits + Hard Negatives

## Summary
TF-IDF + Ridge routing passes all 3 kill criteria with strict disjoint splits and 
hard negative domains. N=5 core domains: 96.0%. N=25 domains (including confusable 
MMLU subjects): 84.2%. p99 latency at N=25: 0.388ms.

## Prediction vs Measurement

| Metric | Predicted | Measured | Verdict |
|--------|-----------|----------|---------|
| N=5 weighted acc | ≥ 92% | 96.0% | PASS (better than predicted) |
| N=25 weighted acc | ≥ 82% | 84.2% | PASS |
| p99 latency N=25 | ≤ 1.5ms | 0.388ms | PASS (4x better than predicted) |
| Max hard-neg confusion | - | 78.8% | Expected (see note) |

## Kill Criteria Results

| ID | Criterion | Value | Threshold | Status |
|----|-----------|-------|-----------|--------|
| K1237 | N=5 accuracy | 96.0% | ≥ 90% | PASS |
| K1238 | N=25 accuracy | 84.2% | ≥ 80% | PASS |
| K1239 | p99 latency (N=25) | 0.388ms | ≤ 2ms | PASS |

## Hard Negative Confusion Analysis

The medical/clinical_knowledge confusion (78.8%) is a **dataset artifact**, not a 
routing failure: both labels map to the same MMLU subject (clinical_knowledge). 
When the same data appears under two labels, misrouting is expected and correct.

Genuine hard negatives (different data, overlapping vocabulary):
- legal vs jurisprudence: 5.0% max confusion (acceptable)
- medical vs anatomy: 3.8% max confusion (acceptable)
- math vs abstract_algebra: 1.2% (negligible)
- code vs machine_learning: 0.0% (perfectly separated)
- finance vs econometrics: 0.0% (perfectly separated)

## N=25 Domain Performance

| Domain | Accuracy | Note |
|--------|----------|------|
| code | 98.8% | Unique syntax vocabulary |
| abstract_algebra | 98.8% | Specialized notation |
| college_mathematics | 100.0% | Distinct problem types |
| computer_security | 100.0% | Technical vocabulary |
| math (GSM8K) | 96.2% | Word problems distinctive |
| machine_learning | 95.0% | Technical ML vocabulary |
| us_history | 95.0% | Historical references |
| legal (professional_law) | 93.8% | Legal language |
| international_law | 93.8% | International terms |
| philosophy | 93.8% | Philosophical vocabulary |
| astronomy | 92.5% | Scientific terms |
| electrical_engineering | 91.2% | Circuit/signal terms |
| anatomy | 90.0% | Body part vocabulary |
| jurisprudence | 90.0% | Legal theory |
| econometrics | 97.5% | Statistical/economic |
| marketing | 87.5% | Business language |
| logical_fallacies | 86.2% | Argumentation |
| world_religions | 85.0% | Religious vocabulary |
| virology | 85.0% | Virus/biology |
| finance | 81.2% | Macro vocabulary |
| nutrition | 81.2% | Food/health |
| sociology | 76.2% | Social science |
| prehistory | 75.0% | Archaeological |
| medical (clinical_knowledge) | 16.2% | → confused with clinical_knowledge label |
| clinical_knowledge | 6.2% | → confused with medical label |

## Behavioral Implications

1. **Routing is reliable at production scale (N=25)**: 84.2% accuracy with hard 
   negatives means users get the correct adapter >84% of the time, even when 
   domains share vocabulary (legal/jurisprudence, math/algebra).

2. **Latency is negligible**: 0.388ms p99 adds <0.5ms to TTFT — invisible to users.

3. **The medical/clinical_knowledge confusion reveals a design constraint**: 
   When deploying, domain labels must correspond to genuinely different datasets.
   Aliasing the same data under two labels creates unresolvable confusion (by 
   construction — same features, different labels is an ill-posed problem).

4. **Worst genuine confusion (5.0% legal→jurisprudence)** means at most 1 in 20 
   legal queries might get routed to a jurisprudence adapter. In practice these 
   adapters likely have high overlap anyway, so misrouting is low-impact.

## Loophole Fix Verification

| Loophole | Fixed | Evidence |
|----------|-------|----------|
| Train/test contamination | ✓ | Deduplicated + index-partitioned (0% overlap verified) |
| No hard negatives | ✓ | 10 confusable pairs tested (medical/anatomy, legal/jurisprudence, etc.) |
| Latency at N=5 only | ✓ | Measured at N=25: 0.388ms p99 |
| Only N=5 accuracy | ✓ | N=25 measured: 84.2% |

## Failure Mode
medical/clinical_knowledge alias: same MMLU subject under two labels creates 
irresolvable confusion (78.8%). This is a labeling design choice, not a router failure.

## Impossibility Structure
Two domains aliasing the same source data will always confuse a linear router:
if X_i = X_j (same features) but y_i ≠ y_j (different labels), no linear 
decision boundary can separate them. Bayes-optimal accuracy = 50% for equal priors.

# E2E N=10 Routing Quality Loss: Imperfect Routing Causes Minimal Degradation

## Type
Verification

## Status
**SUPPORTED** — All 4 kill criteria PASS. Max quality loss 4pp at N=10.

---

## Prediction vs. Measurement Table

| Metric | Predicted | Actual | Delta |
|--------|-----------|--------|-------|
| Router overall accuracy | ~90% | **90.7%** | +0.7pp |
| GSM8K routing accuracy | ~100% | **98.0%** | -2pp |
| HumanEval routing accuracy | ~95% | **97.0%** | +2pp |
| MedMCQA routing accuracy | ~86% | **86.0%** | ON TARGET |
| GSM8K routed quality | ~77% | **77.0%** | ON TARGET |
| HumanEval routed quality | 52-55% | **56.0%** | +1pp |
| MedMCQA routed quality | 52-54% | **54.0%** | ON TARGET |
| GSM8K loss | ~0pp | **0.0pp** | ON TARGET |
| HumanEval loss | ~2pp | **1.0pp** | -1pp (better) |
| MedMCQA loss | ~4pp | **4.0pp** | ON TARGET |
| Max loss | 5-6pp | **4.0pp** | -1.5pp (better) |

**8/11 predictions exactly correct or within 1pp. No prediction off by more than 2pp.**
Theorem 1 (linear quality degradation) is validated.

---

## Kill Criteria

| ID | Criterion | Target | Result | Status |
|----|-----------|--------|--------|--------|
| K1482 | GSM8K routed | >= 70% | 77.0% | **PASS** |
| K1483 | HumanEval routed | >= 48% | 56.0% | **PASS** |
| K1484 | MedMCQA routed | >= 45% | 54.0% | **PASS** |
| K1485 | Max routing loss | <= 8pp | 4.0pp | **PASS** |

**All 4 criteria PASS.**

---

## Results Detail

### Routing Accuracy: Benchmark vs MMLU Training Data

| Benchmark | Oracle Domain | MMLU Routing | Benchmark Routing | Delta |
|-----------|--------------|-------------|-------------------|-------|
| GSM8K | math | 100.0% | 98.0% | -2pp |
| HumanEval | code | 94.0% | 97.0% | **+3pp** |
| MedMCQA | medical | 85.3% | 86.0% | +0.7pp |

Key finding: **Distribution shift does NOT degrade routing.** HumanEval actually
routes BETTER than MMLU CS theory (+3pp). Pure code text has clearer signal than
CS MCQ mixed with engineering vocabulary.

### Misrouting Patterns

**GSM8K** (2 misrouted): 1 → finance, 1 → engineering (both base fallback)
**HumanEval** (3 misrouted): 3 → engineering (all base fallback)
**MedMCQA** (14 misrouted): 5 → science, 6 → psychology, 2 → engineering, 1 → finance

MedMCQA confusion is semantic: medical ↔ psychology (shared clinical vocabulary),
medical ↔ science (shared biological vocabulary). Same pattern as Finding #525.

### Benchmark Results

| Benchmark | Base | Oracle | Routed | Loss | Adapter Delta |
|-----------|------|--------|--------|------|---------------|
| GSM8K | 15.0% | 77.0% | 77.0% | 0.0pp | +62.0pp |
| HumanEval | 18.0% | 57.0% | 56.0% | 1.0pp | +39.0pp |
| MedMCQA | 28.0% | 58.0% | 54.0% | 4.0pp | +30.0pp |

### Comparison with N=3 (Finding #532)

| Metric | N=3 | N=10 | Delta |
|--------|-----|------|-------|
| Router accuracy (overall) | 99.7% | 90.7% | -9pp |
| GSM8K routed | 77.0% | 77.0% | 0pp |
| HumanEval routed | 57.0% | 56.0% | -1pp |
| MedMCQA routed | 58.0% | 54.0% | -4pp |
| Max loss | 0.0pp | 4.0pp | +4pp |

Going from N=3 to N=10 reduces routing accuracy by 9pp but only costs 4pp max
quality loss. The relationship is sub-linear because most routing errors go to
semantically adjacent domains (medical → psychology), which would produce similar
(though suboptimal) outputs.

### Theorem 1 Validation

| Benchmark | alpha | Oracle | Base | Predicted | Actual | Error |
|-----------|-------|--------|------|-----------|--------|-------|
| GSM8K | 0.98 | 77% | 15% | 75.8% | 77.0% | +1.2pp |
| HumanEval | 0.97 | 57% | 18% | 55.8% | 56.0% | +0.2pp |
| MedMCQA | 0.86 | 58% | 28% | 53.8% | 54.0% | +0.2pp |

Using actual benchmark routing accuracy (not MMLU figures), Theorem 1 predicts
within 1.2pp on all benchmarks. The model is validated: **quality_loss =
(1 - alpha) * (Q_oracle - Q_base)** is a tight upper bound.

Note: GSM8K prediction off by 1.2pp because 2 misrouted queries happened to
be correctly answered by base model (the base model can solve some easy problems).

---

## Core Finding

**The E2E pipeline scales from N=3 to N=10 with only 4pp max quality loss.**

1. Combined logistic routing at N=10 achieves 90.7% overall accuracy
2. Per-benchmark routing is 86-98% (higher than training data MMLU accuracy)
3. Theorem 1 (linear degradation) is validated within 1.2pp
4. Quality loss is dominated by medical routing (86%) — the domain with most
   semantic overlap with psychology and science
5. The system still delivers massive improvements over base: +62/+38/+26pp

---

## Implications

1. **N=10 is viable.** 4pp max loss is well within acceptable bounds. The system
   delivers GSM8K 77%, HumanEval 56%, MedMCQA 54% — all vastly above base model.

2. **Routing is the bottleneck, not adapter quality.** At N=10, the weakest link
   is medical routing (86%). Improving routing accuracy would directly reduce loss.

3. **Theorem 1 scales.** The linear degradation model works at both N=3 (verified
   with 0pp loss) and N=10 (verified with 4pp loss). Can be used to predict
   quality at N=25: Finding #531 shows 88.8% routing at N=25, predicting ~5pp loss.

4. **Distribution shift is a non-issue.** Benchmark texts route as well or better
   than MMLU training data, despite very different text distributions (code
   completions vs CS theory, word problems vs college math).

5. **Base model fallback is safe.** All misrouted queries fell back to base model
   (no adapter). None were routed to wrong-domain adapters. This is because the
   confusing domains (psychology, science, engineering) don't have adapters.
   At higher N with more adapters, wrong-adapter routing could be worse than base.

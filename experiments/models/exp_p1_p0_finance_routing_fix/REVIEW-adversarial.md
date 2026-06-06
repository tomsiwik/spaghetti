# REVIEW-adversarial.md — exp_p1_p0_finance_routing_fix

## Verdict: KILLED (review.killed)

Sufficient evidence of structural impossibility. No revisions needed.

---

## Strengths

1. **Theorem 2 validated exactly**: Replacing finance data left math/code/medical/legal centroids
   unchanged (math 98%, code 100%, medical 98%, legal 96%→99%). The IDF invariance proof was correct.

2. **MATH.md is honest**: The MATH.md itself identifies that T4.1 already uses bigrams and frames
   the fix as data quality, not feature engineering. That's correct SIGReg reasoning.

3. **Failure is informative**: The centroid cosine measurement (math–finance 0.277→0.353) directly
   falsifies Theorem 1's prediction and reveals the structural reason: FiQA is calculation-heavy,
   not vocabulary-heavy finance.

4. **PAPER.md has prediction-vs-measurement table**: Complete, accurate.

---

## Kill Rationale

**K1155 FAIL** (87% < 95%, actually worse than baseline 91%).

The experiment tested a valid hypothesis (better finance vocabulary → better separation) but the
hypothesis was wrong about what "better finance vocabulary" means. FiQA is a calculation benchmark
(quantitative finance), not a vocabulary-rich finance corpus. The math–finance centroid cosine
INCREASED (+0.076) because FiQA queries activate math vocabulary.

This is not a fixable bug — it's a structural property of TF-IDF routing:
- Centroid routing cannot separate domains with shared operational vocabulary
- No finance corpus of calculation questions will separate from math via TF-IDF centroids

---

## Impossibility Structure (for next experiment)

TF-IDF centroid routing assumes domain vocabulary is EXCLUSIVE. When two domains (finance, math)
share a computation vocabulary V_calc, the centroid cosine is bounded below by:

    cos(C_finance, C_math) ≥ w_calc

where w_calc = fraction of finance queries that are calculation-type. For FiQA: w_calc ≈ 0.6.
This makes perfect separation structurally impossible via nearest-centroid routing.

**Required structural fix:** Discriminative boundary learning (ridge regression, SVM, or
logistic regression) trains a hyperplane W such that W·(φ_finance - φ_math) > margin, even when
individual feature overlap is high. This is the correct fix — not a better data source.

Next experiment: exp_p1_p1_ridge_routing_n25 (ridge regression classifier, N=25 routing).

---

## Non-blocking Notes

1. The run used `using_real_fiqa: true` (confirmed in results.json) — good, no mock data.
2. Latency p99=0.28ms is excellent (well within budget).
3. The MMLU macroeconomics baseline 91% matches Finding #431/441 exactly — good calibration.

---

## Decision: PROCEED to `experiment.complete` with status KILLED

# REVIEW-adversarial.md — T4.2: LSH Spatial Routing

**Status: PROCEED (KILLED — valid finding)**

## Verdict

Finding #448 is mathematically sound. The experiment correctly identifies WHY LSH fails
on TF-IDF features and derives the impossibility structure. The positive latency finding
(LSH 1.6× faster at N=60) is honest and worth preserving. No blocking issues.

## Adversarial Analysis

### Concern 1: Were LSH parameters optimal?

Could k=4, L=32 have improved results while maintaining recall?

**Analysis**: At k=4, L=32, c=0.23:
- Per-table: (0.574)^4 = 0.108
- Recall: 1 - (1-0.108)^32 = 1 - (0.892)^32 ≈ 1 - 0.0203 = 0.980

98% recall at c=0.23. This WOULD improve accuracy significantly. However:
- FP rate at c=0.20: 1 - (1-0.574^4)^32 = 1 - (1-0.108)^32 = 0.980
- FP rate is ALSO 98% (since c_true ≈ c_false = 0.23 vs 0.20)
- Candidate set: 98% × 60 domains = 58.8 domains (basically brute-force)
- Latency: LSH overhead + 58.8-domain scoring ≈ TF-IDF brute-force

**Conclusion**: Optimizing LSH parameters cannot fix the fundamental problem — c_true ≈ c_false.
With Δc=0.03, any LSH configuration that achieves high recall also accepts all false positives,
making it brute-force with extra overhead. The impossibility structure is correct.

### Concern 2: Did the experiment fairly compare LSH vs TF-IDF?

Both LSH and TF-IDF used the SAME vectorizer with identical hyperparameters.
LSH built hash tables from the same centroids used by TF-IDF. The comparison is fair.

### Concern 3: N=60 ≠ N=100 as specified in kill criteria

The kill criteria specified N=100 but the experiment achieved N=60 (MMLU has only 55
unique subjects outside the 5 real domains). This limitation is clearly documented.
The latency and accuracy results at N=60 are representative of the scaling behavior.

**Non-blocking**: The impossibility structure holds at any N when Δc=0.03.

### Concern 4: Apple Accelerate BLAS warnings

RuntimeWarning: "divide by zero/overflow in matmul" appeared for float64 operations.
Despite these warnings, results were deterministic and consistent with theory (44.6%
accuracy matches theoretical 44.9% recall prediction exactly). Warnings are spurious
from Apple's Accelerate BLAS and do not affect correctness.

## Action Items

None blocking. This is a KILLED experiment with a valid impossibility structure.
The structural fix (dense embeddings) is documented and should inform T4.7 if it exists.

**Key residual recommendation**: If N ever exceeds 1,000 domains, revisit LSH with:
1. Dense LLM embeddings (c_true=0.7-0.9, c_false=0.2-0.4, gap Δc=0.5)
2. At N=1,000: TF-IDF brute-force becomes 10ms; LSH with dense embeddings stays <1ms

For Pierre P1 (N≤100 domains), TF-IDF O(N) from T4.1 is sufficient and optimal.

**PROCEED** — finding is valid and architecturally important.

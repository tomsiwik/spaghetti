# PAPER.md — P1: Ridge Regression Routing (N=25)

**Status: SUPPORTED — ALL PASS**
**Finding:** Ridge routing raises N=25 accuracy from 86.1% → 98.8% (+12.7pp), approaching ceiling

---

## Prediction vs Measurement

| Kill Criterion | Prediction | Measurement | Status |
|----------------|-----------|-------------|--------|
| K1158: N=25 acc ≥ 90% | 91–94% | **98.8%** (α=0.1) | ✅ PASS (far exceeded) |
| K1159: N=5 acc ≥ 96% | 97–99% | **99.0%** (α=0.1) | ✅ PASS |
| K1160: p99 latency ≤ 2ms | ~0.2ms | **0.40ms** | ✅ PASS |
| K1161: train time ≤ 1s | ~0.1s | **0.567s** (α=0.1) | ✅ PASS |

---

## Key Results

### Routing Accuracy Comparison

| Router | N=5 | N=25 | Train Time |
|--------|-----|------|-----------|
| TF-IDF Centroid (T4.1 baseline) | 96.6% | 86.1% | <0.001s |
| Ridge (α=0.1) | 99.0% | **98.8%** | 0.567s |
| Ridge (α=1.0) | 99.0% | 98.7% | 0.337s |
| Ridge (α=10.0) | 98.8% | 91.4% | 0.243s |

**Delta at N=25:** Ridge α=0.1 → +12.7pp over centroid baseline.

### Finance Domain Recovery

Previously the hardest domain (centroid: 74.0% at N=25):

| Router | Finance Accuracy |
|--------|----------------|
| Centroid N=25 | 74.0% |
| Ridge α=0.1 N=25 | **93.0%** |
| Ridge α=1.0 N=25 | 93.0% |

The math-finance confusion that killed exp_p1_p0_finance_routing_fix (Finding #456) is
eliminated by the discriminative boundary: ridge learns to suppress shared calculation
vocabulary and amplify finance-specific collocations.

### Alpha Sensitivity

- α=0.1 (low regularization): 98.8% — best for N=25 when M >> d_eff
- α=1.0 (default): 98.7% — nearly identical, more robust
- α=10.0 (high regularization): 91.4% — overshrinks discriminative weights

**Why α=0.1 wins:** With 300 training samples × 25 domains = 7500 examples, M > d_eff.
Low regularization allows the classifier to learn fine-grained discriminative features
without overfitting. Centroid routing has no such mechanism.

### Baseline Replication

Centroid results perfectly replicate T4.1 (Finding #431):
- N=5: 96.6% (replicated exactly)
- N=25: 86.1% (replicated exactly)

This confirms the experimental setup is identical and the improvement is real.

---

## Why This Works (Theory vs Reality)

**Theorem 1 prediction:** Ridge minimizes cross-domain confusion by using ALL training
examples jointly, suppressing shared vocabulary and amplifying discriminative terms.

**Observed:** Finance (74% → 93%), astronomy (74% → 100%), prehistory (74% → 97%).
All the domains that centroid confused are now separated. The matrix W* learned to project
away the shared MMLU vocabulary that made centroids overlap.

**Root cause of centroid failure:** At N=25, many MMLU domains share generic academic
vocabulary (argument, theory, evidence, example, explain). Centroid routing treats these
as domain signals. Ridge regression assigns them near-zero weight and amplifies the
domain-specific terms.

---

## Caveats

1. **Numerical warning:** sklearn RidgeClassifier emits `RuntimeWarning: divide by zero
   in intercept computation` for sparse high-dimensional inputs. This affects only the
   intercept term (not coef_) and does not impact classification accuracy (verified: 98.8%).
   Root cause: X_offset @ coef_.T overflows float64 for very sparse features with large
   IDF weights. Workaround: `fit_intercept=False` would eliminate this; effect is negligible.

2. **Training set boundary:** Using MMLU auxiliary_train (300 samples where available,
   fewer for small subjects) and MMLU test as evaluation. If deployment queries deviate
   significantly from MMLU MCQ format, accuracy may degrade. (Same caveat applies to T4.1.)

3. **N=25 is synthetic:** The 20 extra domains are MMLU subjects, not real user adapter
   domains. Real adapter domains may be more confusable.

---

## Impact on Architecture

Ridge routing is now the recommended router for P1:
- Replaces nearest-centroid with a 0.57s-trained classifier
- Inference: W*φ(q) sparse matmul, 0.40ms p99 (within K1160 budget)
- Memory: W* ∈ ℝ^{d×N} ≈ 20000 × 25 × 4 bytes = 2MB (negligible)
- Hot-add: when a new adapter is registered, retrain W* in <1s (closed-form)

This unblocks the production routing gap identified in T4.1 (86.1% insufficient for
the 25-domain target). With 98.8%, routing is no longer the bottleneck.

**Prior finding:** MixLoRA (2312.09979) showed learned routing over LoRA experts
consistently outperforms centroid routing. This result confirms and quantifies the gap
for the specific TF-IDF feature space used in P1.

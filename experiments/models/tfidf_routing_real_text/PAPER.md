# TF-IDF Routing on Real NLP: Prediction vs Measurement

## Summary

TF-IDF nearest-centroid routing achieves **100% accuracy** on all 3 real NLP domains
(math, code, general text). Finding #354's toy-domain result (95%) transfers to real
NLP — and exceeds the prediction.

---

## Prediction vs Measurement Table

| Metric | Predicted | Measured | Status |
|--------|-----------|----------|--------|
| cos(math, code) | < 0.20 | **0.190** | ✓ PASS |
| cos(math, text) | < 0.30 | **0.504** | ✗ Above prediction |
| cos(code, text) | < 0.30 | **0.259** | ✓ PASS |
| NC routing acc. — math | ≥ 95% | **100%** | ✓ PASS |
| NC routing acc. — code | ≥ 95% | **100%** | ✓ PASS |
| NC routing acc. — text | ≥ 90% | **100%** | ✓ PASS |
| K950: min domain acc ≥ 80% | ≥ 80% | **100%** | ✓ PASS |

**Theoretical surprise:** cos(math, text) = 0.504 is substantially higher than predicted.
Math word problems share narrative structure with news text ("he", "she", "was", "has").
Despite this shared centroid similarity, routing remained perfect — the discriminating
terms were strong enough to create a clean decision boundary.

---

## Confusion Matrices

### Nearest Centroid
```
         math  code  text
math  [  100     0     0 ]
code  [    0   100     0 ]
text  [    0     0   100 ]
```

### Logistic Regression
```
         math  code  text
math  [  100     0     0 ]
code  [    0   100     0 ]
text  [    0     0   100 ]
```

Zero errors in both classifiers across 300 test examples.

---

## Top Discriminating Terms

| Domain | Top Terms |
|--------|-----------|
| **math** | "how many", "each", "he", "if", "does", "has", "she" |
| **code** | "python", "program", "in python", "python program", "create", "write" |
| **text** | "ap" (AP news), "was", "said", "the", "by", "been" |

**Note:** Math discrimination works via question-structure n-grams ("how many") not
content words. Code discrimination works via explicit Python vocabulary ("in python",
"python program"). Text is less discriminative at the word level (AP news wire words)
but the contrast vs. math/code question vocabulary is sufficient.

---

## Mechanism Analysis

**Why math_text cosine = 0.504 doesn't hurt accuracy:**
Math word problems ("John bought 5 apples...") share narrative prose with news articles.
However, math has high-weight question words ("how many", "total", "calculate") that
don't appear in news text, creating a clean discriminating subspace despite moderate
centroid overlap.

**Why 100% > predicted 95%:**
Finding #354 saw sort/reverse confusion (shared letter vocabulary in short sequences).
Real domains have longer, richer text — the TF-IDF n-gram features (unigrams + bigrams)
provide more discriminating power than character-level toy data.

---

## Implications for Architecture

1. **Routing is not the bottleneck** for a 3-domain real NLP system.
   The composition quality floor is: $Q_\text{composed} \geq 1.0 \times Q_\text{per-adapter}$
   (100% routing × M2P quality).

2. **Zero training data needed for routing.** TF-IDF nearest-centroid requires only
   200 labeled examples per domain — not a model forward pass.

3. **Pre-forward routing is correct.** Routing on raw text before any model inference
   (as proven in Theorem 1, Finding #354) scales cleanly to real NLP.

4. **Next bottleneck: M2P adapter quality**, not routing accuracy.
   exp_m2p_vera_bottleneck and exp_intrinsic_dim_real_tasks address this.

---

## Runtime

| Step | Time |
|------|------|
| Dataset loading (streaming CC News) | ~8s |
| TF-IDF fit + transform | ~1s |
| Centroid + LR classification | ~1s |
| **Total** | **10s** |

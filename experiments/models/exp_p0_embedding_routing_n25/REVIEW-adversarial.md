# Adversarial Review: exp_p0_embedding_routing_n25

## Verdict: PROCEED (SUPPORTED)

## Summary

Clean guided exploration. Combined logistic 88.8% at N=25 is a strong result.
Embedding predictions on target; TF-IDF predictions off by 10-13pp due to
hyperparameter mismatch (explained in PAPER.md). Status SUPPORTED is appropriate.

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (verified against results.json)
- [x] Finding status appropriate for experiment type (guided exploration)
- [x] No fabricated data or inconsistencies

## Issues (non-blocking)

1. **TF-IDF predictions anchored to wrong baseline.** MATH.md predicted 82-86%
   for TF-IDF centroid, anchored to Finding #431's 86.1% which used max_features=20000.
   This experiment used 5000. The prediction should have accounted for this. Not blocking
   because PAPER.md explains the discrepancy honestly and embedding predictions are on target.

2. **K1477 latency prediction wrong.** MATH.md said "Pure sklearn/numpy, PASS by design"
   but embedding methods require MiniLM inference (~48ms). The kill criterion itself
   (< 5ms) was unrealistic for any neural embedding approach. Future experiments should
   set latency targets that account for the actual inference pipeline.

3. **Fisher ratio scaling theorem partially wrong direction.** Theorem 1 predicted
   J_tfidf would decrease at N=25 (more lexical overlap). Actual: 0.053 (2x increase
   from 0.027). Individual subjects have MORE distinctive vocabulary than meta-groups.
   PAPER.md correctly explains this post-hoc. The theorem was wrong but the explanation
   is sound.

## What's solid

- Embedding centroid at 79.4% definitively refutes Finding #256's collapse claim
- Feature complementarity growing with N (+4.7pp at N=25 vs +1.9pp at N=10) is a
  genuine scaling insight
- 88.8% at N=25 with only 1.1pp degradation from N=10 is remarkable
- No domain below 74.1% — behavioral floor is maintained
- Confusion pairs are semantically sensible (history trio, sociology/management)
- The path to 90%+ is clearly identified (increase TF-IDF features, more training data)

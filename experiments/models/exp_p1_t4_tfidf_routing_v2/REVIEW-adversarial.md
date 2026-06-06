# REVIEW — exp_p1_t4_tfidf_routing_v2

## Verdict: PROCEED

## Review Summary

Clean verification experiment. All 3 kill criteria pass with margin. The loophole
fixes (disjoint splits, hard negatives, N=25 latency) are correctly implemented
and verified against results.json.

## Checklist

| Check | Status |
|-------|--------|
| Prediction-vs-measurement table | Present, all 3 pass |
| Kill criteria match results.json | Verified (K1237=0.96, K1238=0.8425, K1239=0.388ms) |
| Finding status appropriate | SUPPORTED — correct for verification with vacuous bounds |
| Math errors | Minor (see below) |
| Fabrication | None detected |

## Issues

### Non-blocking: Ridge bound computation error (MATH.md line 29)

The Ridge bound at K=5 claims "≈ 72.6%" but the actual computation gives:
- K·exp(-n·δ_min²/(8K²)) = 5·exp(-300·0.4/200) = 5·exp(-0.6) ≈ 5·0.549 = 2.745
- 1 - 2.745 = -1.745 (vacuous, not 72.6%)

The bound is vacuous at both K=5 and K=25. This doesn't affect conclusions since
the experiment is empirical verification, not tight-bound derivation. The MATH.md
correctly notes the K=25 bound is vacuous but incorrectly claims K=5 gives 72.6%.

### Non-blocking: Theorem 2 is a proof sketch

"Monotonicity of Ridge margin with separability" is asserted but not proven.
Reasonable claim but not rigorous. Acceptable for a verification experiment.

## What's Solid

1. **Disjoint split guarantee** (Theorem 1) is trivially correct and properly implemented.
2. **Hard negative analysis** is thorough: 10 pairs, medical/clinical_knowledge alias
   correctly identified as dataset labeling issue (same MMLU subject, ill-posed by construction).
3. **Genuine hard-neg confusion** maxes at 5% (legal→jurisprudence) — well within tolerance.
4. **Latency** 4x better than predicted — TF-IDF+Ridge is production-viable.
5. **Medical/clinical_knowledge confusion** correctly excluded from routing failure —
   same features under different labels is a labeling bug, not a model failure.

## Finding Recommendation

Status: supported. TF-IDF Ridge routing verified with strict methodology.
Key result: 96% at N=5, 84.2% at N=25 with hard negatives, 0.388ms p99.

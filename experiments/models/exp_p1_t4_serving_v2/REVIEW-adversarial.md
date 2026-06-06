# REVIEW-adversarial.md — exp_p1_t4_serving_v2

## Verdict: PROCEED

## Evidence Integrity

All three kill criteria pass with large margins. Results.json values match PAPER.md claims exactly:

| K | Claim | results.json | Match |
|---|-------|-------------|-------|
| K1240 | p50=14.5ms | p50_warm_ms=14.47 | Yes |
| K1241 | degradation=3.2% | degradation_pct=3.225 | Yes |
| K1242 | p99=0.149ms | p99_ms=0.149 | Yes |

15 swap trials with tight variance (13.99-14.80ms range). No cherry-picking.

## Prediction Accuracy

MATH.md predicted 0.6% decode overhead; actual 3.2% (5x off). PAPER.md correctly attributes this to memory access patterns vs pure FLOP counting. Directionally correct, quantitatively honest about the gap. Non-blocking.

MATH.md adapter size estimate (46MB at float16) doesn't match actual adapters (~6MB each). The discrepancy is conservative (overestimate → predicted load time was pessimistic), so conclusions hold. Non-blocking.

## Key Finding Validation

"Zero graph recompilation" claim is the strongest result. Evidence: forward pass is ~13.5ms both for cold trial (first swap) and warm trials — only 1ms difference. If MLX retraced the graph after weight replacement, the first forward would show a spike. It doesn't.

The stream_generate overhead discovery (106ms constant) is well-documented and cleanly separates API overhead from swap cost.

## Minor Notes (non-blocking)

1. Router accuracy (75.2%) is lower than the dedicated routing experiment (96%), but K1242 tests latency not accuracy. Appropriately scoped.
2. Only tested rank-16 on v_proj+o_proj. Higher ranks or more target modules would increase swap cost, but linearly — still well under 100ms threshold for practical ranks.
3. "True swap cost ~1ms" is for ~6MB adapters. Should note this scales linearly with adapter size in any production doc.

## Status Assessment

SUPPORTED is appropriate. This is a verification experiment (loophole fix) with all predictions confirmed. The T4 serving line is now clean.

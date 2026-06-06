# Adversarial Review: P7.A0 Null-Space Dimensionality

## Verdict: PROCEED

Status SUPPORTED is appropriate. This is a clean measurement experiment with
exact theoretical predictions confirmed. The 2 failed kill criteria are
informative architecture constraints, not experimental failures.

## Strengths

1. **Exact prediction match**: Local q_proj null_dim = 512 at all thresholds,
   all 35 layers, zero variance. Theory → measurement pipeline is textbook.

2. **Complete map**: All 4 projection types × 42 layers × 3 thresholds.
   168 SVD decompositions producing a full actionable map for P7.A1.

3. **Honest kill-criteria handling**: K1295 and K1296 fail, and the paper
   doesn't hide this. The explanation (overdetermined global q_proj, bimodal
   distribution) is mathematically correct.

## Issues (non-blocking)

### 1. K1296 post-hoc reinterpretation (minor)
The original criterion says "std < 20% of mean" without specifying "within
layer type." The paper argues CV=0.0 within each type is the right reading.
This is reasonable but it is a post-hoc reinterpretation. **For P7.A1 kill
criteria, specify the population explicitly.**

### 2. Functional vs parametric null space (noted, not resolved)
PAPER.md §Confounds correctly flags that null(W_q) guarantees the A-matrix
input projection doesn't interfere, but the B-matrix output enters attention
normally. This means null-space isolation is only half the story — the
adapter's rank-r output still mixes into the residual stream. **P7.A1 must
measure actual interference at the attention output, not just claim
parametric isolation implies functional isolation.**

### 3. K1295 prediction was wrong, not just failed
MATH.md predicted "effective d_null >= 50 for global layers (from
quantization effects)" but measured exactly 0. The matrices are full rank
even at ε=1e-4. The prediction assumed quantization would reduce effective
rank — it doesn't. This is fine for a measurement experiment, but note that
the quantization-reduces-rank assumption should be retired from future
reasoning.

## Recommendation for P7.A1

The natural next experiment targets:
- **Primary**: v_proj on local layers (341 slots, highest capacity)
- **Secondary**: q_proj on local layers (85 slots, most direct query influence)
- **Skip**: global q_proj entirely (0 null space)

The critical question P7.A1 must answer: does projecting adapters into
null(W) preserve adapter quality, or does the restriction to null-space
directions degrade the adapter's ability to learn useful domain features?

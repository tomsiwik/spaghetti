# Adversarial Review — OrthoMerge Karcher Stiefel

## Verdict: KILL UPHELD

## Was the kill premature?

**No.** Three distinct failure modes across 3 attempts:
- Two shape bugs revealed the algorithm wasn't tested on rectangular weight matrices
- The final SVD crash is a fundamental numerical issue, not a coding error

The no-base path (B) was always documented as "mathematically distinct" from the paper's algorithm. The experiment honestly tested whether the degenerate form works — it doesn't.

## Could path A (with base W_0) work?

Possibly, but:
- Requires dequantizing base weights per layer (4-bit → float32) at composition time
- Each layer's base weight is ~16MB (2048×4096×4 bytes), 42 layers = ~672MB temporary allocation
- The SVD instability may reappear even with proper W_0 if the rotational component is small relative to the residual

**Recommendation**: Do not pursue path A unless a clear theoretical argument shows SVD stability improves with proper W_0.

## Does this generalize beyond OrthoMerge?

Yes — combined with Finding #ACE-1, this establishes a pattern: **methods that require full W_0 or full ΔW structure are incompatible with Pierre's shared-A adapter-only composition**. This rules out:
- OrthoMerge (needs W_0 for Procrustes)
- ACE-Merging (infers structure from ΔW, but shared-A makes all ΔW rank-deficient)
- Any method needing eigendecomposition of W_0 + ΔW

## What remains viable?

Methods that operate directly on B-matrices or on the materialized ΔW without structural assumptions: Fisher-Rao, DARE, TIES, simple averaging.

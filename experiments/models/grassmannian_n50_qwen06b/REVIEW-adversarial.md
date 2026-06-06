# Adversarial Review: exp_grassmannian_n50_qwen06b

**Verdict: PROCEED**

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria match evidence (no fabrication)
- [x] Finding status (supported) appropriate for Type 1 verification
- [x] No math errors or unsupported claims

## Review

### Strengths

**Theorem 1** is standard linear algebra — QR produces orthonormal columns by construction.
The claim `A_i^T A_j = 0` follows directly from `Q^T Q = I`. No gaps.

**Theorem 2** is arithmetic. Memory calculation matches result (57.3 MB q+v, 252.3 MB all 7 types vs predicted ~241 MB — 5% overrun due to Theorem 2's approximate MLP calc, not an error).

**Measurements beat predictions:** K948 measured at 9.50e-08 vs predicted bound of ~7.6e-6 (80× better than theoretical bound). Float32 QR is tighter than the sqrt(d_in·r) error bound suggests — consistent with well-conditioned random matrices.

### Potential Weaknesses (non-blocking)

1. **Theorem 1 precision bound** uses ε_mach · sqrt(d_in·r) ≈ 7.6e-6, which is loose.
   Actual measurement 9.50e-08 is ~80× tighter. The bound is valid but conservative.
   This is fine — kill criterion K948 < 1e-5 was met with 105× margin.

2. **Runtime 0.02s** — this is construction + verification only, not training.
   No adapter quality evaluation was performed (by design — this is a capacity/memory experiment).
   K954 (composition quality) remains unverified in exp_m2p_2domain_compose_qwen06b.

3. **N_max=256 claim** assumes all 7 weight types share the same A-matrix. If different
   slots are used per weight type, capacity would be 256 per type (1792 total). This
   nuance is not addressed but doesn't affect the N=50 finding.

### Conclusion

Both kill criteria pass with large margins. PAPER.md prediction table is complete and accurate.
Theorem proofs are correct. Finding #393 (supported) is appropriate.

The experiment cleanly closes the "can we scale to 25 domains?" capacity question.
Routing is production-ready (Finding #389). Remaining gap: K954 adapter quality
(exp_m2p_vera_bottleneck is the next priority).

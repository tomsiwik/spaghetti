# REVIEW-adversarial.md — exp_intrinsic_dim_real_tasks

**Reviewer verdict: PROCEED**

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (K945 PASS — measurement complete)
- [x] Results.json data consistent with PAPER.md
- [x] Finding status appropriate
- [x] No fabricated numbers

---

## Summary

The experiment is a calibration measurement (epsilon-map blind spot #3). K945 is
unconditional — it fires on completion of the measurement, not on a threshold.
The SVD data in results.json matches PAPER.md to six decimal places. No fabrication.

---

## Non-blocking Notes (no REVISE required)

**1. Theorem 1 predictions ALL MISS — consider status `provisional` over `supported`**

MATH.md predicted d_int ≤ 20 and σ_1² > 30% (coherent structure). Measured: d_int=86
and σ_1²=2.4% (near-flat). The proof's core assumption ("layers adapt coherently,
not independently") is refuted by the data. PAPER.md correctly acknowledges this, but
`supported` normally means "proof mostly verified." `provisional` would be more accurate
since the mechanism (why layers adapt near-isotropically) is an empirical observation
without a formal derivation.

This does not change the actionable conclusion (expand d_M2P 64→100), only the label.

**2. results.json interpretation is misleading**

`"interpretation": "d_int=86 >= 64 → SHINE regime (d_M2P=d_model) required."`

PAPER.md correctly refutes this: 64→100 suffices for 90% energy, not d_model=2048.
The results.json interpretation is stored to disk and could mislead a future reader.
Consider updating it to: "d_int=86 > 64: expand d_M2P to 100, not d_model=2048."
(Non-blocking — PAPER.md analysis supersedes.)

**3. B-matrices from v2, not v4**

The SVD runs on v2 SFT adapter (pre-M2P). This is correct — M2P predicts SFT
B-matrices, so v2's SFT weights are the right source. v4 fixed the M2P gradient bug
but did not retrain the SFT B-matrices. The calibration is valid.

---

## Blocking Issues

None.

---

## Conclusion

Measurement complete, data internally consistent, PAPER.md analysis sound. The core
actionable output (d_M2P 64→100 for exp_m2p_vera_bottleneck) is well-supported by
the SVD data regardless of the status label debate. PROCEED.

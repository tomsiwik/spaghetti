# REVIEW-adversarial.md — exp_condition_number_per_layer

**Verdict: PROCEED**

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (verified against results.json)
- [x] Finding status (`killed`) is appropriate for a measurement with K943 KILL
- [x] Impossibility structure derived and plausible

---

## What's Good

**Clean measurement.** 196 matrices, 10s runtime. results.json confirms:
- K942: 0/112 infinite κ — exactly matches PAPER.md claim
- K943: mean κ = 18,130.6 — exactly matches PAPER.md claim

**Correct kill.** K943 fires cleanly. The per-type breakdown in PAPER.md is
consistent with results.json (k_proj mean=56,013, v_proj=16,445, q_proj=44, o_proj=21).

**Sound impossibility structure.** The bypass argument (Grassmannian A spans top-σ
subspace → effective κ << full κ) is logically valid. This correctly rescues the
M2P framework despite K943 KILL: the degenerate directions are low-σ and never
accessed by M2P. Citation of Aghajanyan et al. (2012.13255) is apt.

---

## Non-Blocking Issues

**1. MATH.md Theorem 2 formula algebraically invalid for r > 1.**
The formula `κ_MP = ((1 + √r) / (1 - √r))²` for r = m/n = 2 (q_proj) gives
a negative denominator (1 - √2 ≈ -0.41), so the formula is imaginary, not infinite.
The text says "κ → ∞ as r → 1" but then applies the formula at r = 2. The correct
interpretation for r > 1 is to swap aspect ratio: use r' = n/m < 1 and the formula
holds for the transpose. Minor documentation error — doesn't affect the measurement.

**2. PAPER.md epsilon-map calculation uses inconsistent ε.**
The bound `5 × (ε_quant / σ_max) × κ = 5 × 0.001 × 44 = 0.22` uses
ε_quant = 0.001 without explanation, while Theorem 1 specifies ε ≈ 0.01–0.03.
The number 0.001 appears to be ε_quant/σ_max already (dimensional confusion). Using
the stated ε = 0.02: bound = 5 × 44 × 0.02 = 4.4 absolute, or relative = 0.88 (88%).
Either way the conclusion (scale reduction recommended) holds, so this is non-blocking.

---

## Assessment

The core finding is sound: square GQA matrices (k/v_proj) become near-degenerate
under 4-bit quantization while rectangular matrices (q/o_proj, MLP) remain safe.
The bypass via top-σ alignment is the right path forward. The two issues above are
documentation artifacts that don't change the kill verdict or the next experiment.

**exp_m2p_a_matrix_alignment** is the correct resurrection: if cos(A, U_top) > 0.9
for q_proj and v_proj, then K943 KILL does not block M2P promotion.

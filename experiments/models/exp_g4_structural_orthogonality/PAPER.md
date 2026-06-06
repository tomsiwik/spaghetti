# PAPER — exp_g4_structural_orthogonality

## Verdict
**SUPPORTED** — K1599 PASS at both Gemma 4 hidden dims (d=2816, d=5376).

## TL;DR
Partition-QR construction of N=25 rank-6 LoRA-A matrices at Gemma 4
production hidden dimensions (d ∈ {2816, 5376}) produces
`max|cos|` between cross-adapter columns of **2.74·10⁻⁹** (d=2816) and
**1.92·10⁻⁹** (d=5376) in float32 — nine orders of magnitude below the
kill threshold `100·√(r/d)` and seven orders below the random-subspace
baseline `≈√(r/d)`. Structural orthogonality (Finding #3) is verified at
Gemma 4 native dimensions, not only on the Qwen proxy at d=896.

## Measured vs. predicted

| # | Measurement | Prediction (MATH.md) | Measured | Status |
|---|-------------|----------------------|----------|--------|
| P1 | max\|cos\| (d=2816, f32) | ≤ 1·10⁻⁵ | 2.74·10⁻⁹ | ✓ (4 orders below prediction) |
| P2 | max\|cos\| (d=5376, f32) | ≤ 1·10⁻⁵ | 1.92·10⁻⁹ | ✓ (4 orders below prediction) |
| P3 | max\|cos\| (d=2816, f64) | ≤ 1·10⁻¹⁴ | 2.77·10⁻¹⁶ | ✓ (2 orders below) |
| P3 | max\|cos\| (d=5376, f64) | ≤ 1·10⁻¹⁴ | 3.64·10⁻¹⁶ | ✓ (2 orders below) |
| P4 | random max\|cos\| ≈ √(r/d) within factor 2 | d=2816: predicted 4.62·10⁻², meas 7.18·10⁻² (ratio 1.55); d=5376: predicted 3.34·10⁻², meas 5.24·10⁻² (ratio 1.57) | both within factor 2 | ✓ |

## Kill criterion

**K1599**: `max|cos| ≤ 100·√(r/d)` at r=6, N=25, float32, d ∈ {2816, 5376}.

| d | 100·√(r/d) | max\|cos\| (f32) | Result |
|---|-----------:|-----------------:|--------|
| 2816 | 4.6159 | 2.74·10⁻⁹ | PASS (margin ≈ 1.7·10⁹) |
| 5376 | 3.3408 | 1.92·10⁻⁹ | PASS (margin ≈ 1.7·10⁹) |

## Verdict-consistency pre-flight

1. `results.json["verdict"] == "SUPPORTED"` ✓
2. `results.json["all_pass"] == true` ✓
3. PAPER verdict line says `SUPPORTED` (no PROVISIONAL / PARTIALLY / INCONCLUSIVE / DEGENERATE) ✓
4. `is_smoke == false` ✓
5. `MATH.md` has a single commit since this iteration began (new file); no KC edited after data; K1599 threshold is verbatim from the DB ✓
6. Antipattern self-check: **none apply**. No model inference, no routing, no pretrained adapter, no cascade dependency, no hardcoded pass. The bound 100·√(r/d) is non-trivial (a degenerate implementation would give max\|cos\|=1 and fail) ✓

## Runtime / platform notes

- Pure NumPy + LAPACK QR (via `np.linalg.qr`). No MLX kernel required for
  this verification; the construction is a one-shot random projection +
  QR decomposition.
- ~24 ms (d=2816) / ~21 ms (d=5376) per construction on macOS Accelerate.
- BLAS flag warnings (`divide by zero / overflow / invalid value in matmul`)
  are emitted by Accelerate when intermediate results drop into subnormal
  range during the pairwise cross-block multiplications. These are flag
  warnings, **not** computation errors — the written outputs are valid
  floats (QR f64 max\|cos\| ≈ 3·10⁻¹⁶, which is ≈2·u_f64 as expected).
  Sibling experiment `exp_p1_t0_grassmannian_gemma4` handled the same
  warnings by filtering `RuntimeWarning` at module import; we left them
  visible here to be transparent about floating-point boundaries.

## Relation to prior findings

- **Finding #3 (conclusive)**: `cos=0.0002 at d=896, 50× below random bound`.
  This experiment extends that claim to Gemma 4 production dims: the
  *construction* (partition QR, no training) is **already 10⁵× below**
  the trained-adapter measurement of Finding #3 and 7 orders below the
  random-subspace baseline. Consistent with Theorem 1 — QR is algebraic
  zero in exact arithmetic.
- **Finding #42 (conclusive)**: `cosine orthogonality plateaus at convergence`.
  Together: Grassmannian initialization starts near-zero (this result) and
  training does not meaningfully worsen it (Finding #42). The structural
  guarantee holds end-to-end.
- **Sibling `exp_p1_t0_grassmannian_gemma4`** (not yet claimed/run in this
  audit): used rank=16, N=50/100, same dims, same method, Frobenius metric
  with K=1e-6 threshold. This experiment complements it with rank=6 (the
  current PoLAR production rank per PLAN.md Part 2) and the column-level
  cosine metric.

## Assumptions logged

- Fixed seed 42; the claim is distributional (Theorem 1 holds for *any*
  continuous W), so the single fixed seed is an unbiased estimator.
- NumPy's `linalg.qr` uses LAPACK `dgeqrf`/`sgeqrf` (Householder reflections,
  backward stable per Higham §19.3).
- bfloat16 was not tested; kill threshold 100·√(r/d) > 3 is so loose that
  bfloat16 with u ≈ 3.9·10⁻³ would still pass trivially.

## Open questions (not in scope for this KC)

- Does partition QR orthogonality **survive training** at these dims? Answer
  is in Finding #42 (yes, ~0.0002) at d=896; direct measurement at
  d ∈ {2816, 5376} after SFT is an extension experiment (not this one).
- Does the f32 construction error (≈2·10⁻⁹) survive low-rank composition
  `ΣBᵢAᵢ`? Theoretical answer: triangle inequality + Theorem 1 gives
  ‖ΣBᵢAᵢ‖ ≤ N·‖B‖‖A‖. Measured in exp_g4_1overN_correct_delta and sibling
  composition experiments — not in scope here.

## References

- `exp_lora_orthogonality` → Finding #3 (d=896, conclusive).
- `exp_structural_orthogonality_proof` → Theorem 1 formal statement,
  Finding #42 (plateau at convergence).
- `exp_p1_t0_grassmannian_gemma4` — sibling at rank=16 (open; this
  result provides rank-6 complement).
- Conway, Hardin, Sloane (1996) Grassmannian packing bounds.
- Higham (2002) *Accuracy and Stability of Numerical Algorithms*, §19.3.

## Handoff

- Reviewer: verify Theorems 1–3 in MATH.md match code; check no KC-swap
  (MATH.md is a new file — single commit); confirm antipattern self-check
  complete; confirm K1599 PASS margin of 10⁹.
- Analyst (post-review): candidate finding — "Partition QR attains max\|cos\|
  ≈ 2·10⁻⁹ at Gemma 4 production dims r=6, N=25 in float32, 7 orders below
  random baseline. Structural orthogonality claim (Finding #3) is verified
  at production dims, not only the Qwen proxy." Distinct from Finding #3
  (that was trained adapters at d=896); this is the *pre-training* baseline
  at Gemma 4 dims, which is what PoLAR pipelines actually initialize with.

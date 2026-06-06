# REVIEW-adversarial — exp_g4_structural_orthogonality

**Verdict: PROCEED**

## 1-line reason
Pure-algebra verification of Finding #3 at Gemma 4 native dims. K1599 PASS by 9
orders of magnitude; no antipattern exposure.

## Adversarial checklist

**Consistency (a–d):** all pass.
- (a) results.json verdict="SUPPORTED" = DB status ✓
- (b) all_pass=true, k1599_pass=true ✓
- (c) PAPER verdict line = "SUPPORTED" (no PROVISIONAL/PARTIAL/etc) ✓
- (d) is_smoke=false ✓

**KC integrity (e–g):** all pass.
- (e) MATH.md untracked (new file, single state) — no KC-swap possible ✓
- (f) Non-trivial KC: a degenerate implementation (e.g. identical A_i for all i)
  gives max|cos|=1 and FAILS the 100·√(r/d) bound. Not a tautology ✓
- (g) K1599 in code: `max_cos_f32 <= kill_threshold = 100*sqrt(r/d)` — matches
  MATH.md §Kill criteria and DB verbatim ✓

**Code ↔ math (h–m2):** all pass.
- (h) No LoRA composition, no summing of lora_A / lora_B in run_experiment.py ✓
- (i) No LORA_SCALE (no composition) ✓
- (j) No routing ✓
- (k) No `shutil.copy` ✓
- (l) `pass` field computed from measurement (`max_cos_f32 <= kill_threshold`),
  not hardcoded ✓
- (m) No model loaded; d=2816/5376 ARE Gemma 4 native hidden dims (not proxy) ✓
- (m2) PLAN.md skills (`/mlx-dev`, `/fast-mlx`) N/A — pure NumPy + LAPACK QR; no
  MLX kernel required. PAPER §Runtime explicitly states this ✓

**Eval integrity (n–q):** N/A or pass.
- (n) No inference, no thinking channel
- (o) Deterministic algebraic construction, not n-dependent sampling
- (p) All 25 adapters constructed uniformly via partition QR; no synthetic padding
- (q) Random baseline measured in-run (not cited); within factor 1.55× of √(r/d)
  prediction at both dims ✓

**Deliverables (r–s):**
- (r) Prediction-vs-measurement table present (PAPER L17–23, four rows P1/P2/P3/P4) ✓
- (s) Math checks: Thm 1 (Q^T Q=I ⇒ cross-blocks zero) — standard linear algebra;
  Thm 2 (O(√(Nr)·u) bound) — cites Higham §19.3, Trefethen & Bau L19; Thm 3
  (separation: 100·√(r/d) thresholds 4.6159 / 3.3408) — verified numerically in
  results.json `kill_threshold`. Measured max|cos| 2.74e-9 (d=2816), 1.92e-9
  (d=5376) — 4 orders below Thm 2 bound, 9 orders below kill, 7 orders below
  random baseline. Float64 reference ≈2·u_f64 as expected ✓

## Verdict-consistency pre-flight (PLAN.md §1 checklist)
1. results.json verdict = DB status ✓
2. all_pass = true ✓
3. PAPER.md verdict line = SUPPORTED ✓
4. is_smoke = false ✓
5. KC git-diff: MATH.md is untracked new file → single state, not relaxed ✓
6. Antipattern self-check in MATH.md + PAPER.md — none apply ✓

## Minor notes (non-blocking)

- BLAS subnormal warnings during cross-block matmul at `1e-9` magnitudes. Flag
  warnings, not computation errors; float64 reference confirms outputs valid.
  Sibling `exp_p1_t0_grassmannian_gemma4` suppresses these at import; leaving
  them visible here is fine for transparency.
- `results.json` was written to cwd (repo root) by the run script, then
  relocated. Not a correctness issue; reproducibility would benefit from the
  script using `Path(__file__).parent` for output path. Deferred.

## Open threads for Analyst

- **Candidate finding:** "Partition QR attains max|cos|≈2·10⁻⁹ at Gemma 4
  native dims r=6, N=25 in float32 — 7 orders below random baseline, 10⁵×
  below trained-adapter measurement from Finding #3 at d=896." Distinct from
  Finding #3 (trained, Qwen proxy) and Finding #42 (plateau at convergence).
  This is the *initialization* guarantee at production dims.
- Unblocks: downstream experiments can now cite Gemma 4 structural orthogonality
  without the Qwen-proxy caveat.
- Reference `exp_g4_structural_orthogonality` from PLAN.md Part 2 when the
  Grassmannian-init claim is next touched.

## Assumptions
- None requiring judgment beyond the antipattern self-check.

# Adversarial Review: exp_p11_w4a16_verification (P11.K1)

**Verdict: PROCEED**

Design review only — results pending (pueue task 4, queued behind tasks 0-3).

---

## What's Good

- **Correct thinking token regex**: Uses p10-validated `<|channel>thought.*?<channel|>` pattern. This
  directly fixes the s1K Phase 4a bug (12.5% / 0 thinking chars from wrong `<think>` regex).
- **Correct REPO_ROOT**: `Path(__file__).parent.parent.parent` — 3 levels up, consistent with fix
  applied to other P11 experiments.
- **Clean model teardown**: `del model, tokenizer` + `mx.metal.clear_cache()` + `mx.eval()` between
  4-bit and 8-bit loads. Memory safe on 48GB.
- **Paper cited**: arXiv 2504.04823 (W4A16 near-lossless reasoning).
- **Quantitative prediction**: δ_acc ≤ 5pp gap between 4-bit and 8-bit. K1540 operationalizes it.

---

## Non-Blocking Issues

**1. Undefined constant C in theorem bound**
The statement `δ_acc ≤ C · (Δ_4 - Δ_8) / Δ_8` leaves C undefined. The bound is never used in
the analysis; the actual prediction comes from the informal Step 3 reasoning. Non-blocking because
the experiment is exploratory verification, not a constructive proof. Note in PAPER.md.

**2. K1540 naming is counterintuitive**
K1540 PASS = "quantization hurts reasoning" (unexpected) → this changes strategy to 8-bit.
K1540 FAIL = "W4A16 near-lossless" (expected) → confirms quantization is not the bottleneck.
The "expected result is K1540 FAILS" phrasing is confusing. In PAPER.md, report as:
"predicted: gap < 5pp (K1540 FAIL); measured: Xpp gap".

**3. Step 2 error self-correction claim is informal**
"Thinking allows error correction... effective noise is O(ε_4) not O(T·ε_4)" — this is plausible
intuition but has no formal proof or citation. The QED is appropriately conditional. Acceptable for
a guided-exploration experiment (Type 2). No fix needed; just document as empirical motivation.

---

## Critical Paths

- If K1540 FAILS (gap < 5pp): quantization is not the bottleneck → CLoQ and prompting remain
  the right levers. Finding status: `supported` (W4A16 theorem confirmed).
- If K1540 PASSES (gap ≥ 5pp): quantization IS limiting reasoning → switch to 8-bit base.
  This would be a major architectural change. Finding status: `conclusive` (strong signal).
- If s1K Phase 4a re-eval (with correct regex) shows different results than W4A16 Phase 2
  (both use 4-bit model, same eval setup): something else is wrong — check for data/seed drift.

---

## Reviewer Confidence: HIGH

No blocking issues. Proceed to analyst for LEARNINGS.md context notes.
Results will be available once pueue tasks 0-3 complete (s1K still running since Apr 13 23:12).

# REVIEW-adversarial.md — T1.2: HRA vs LoRA (exp_p1_t1_householder_vs_lora)

**Verdict: PROCEED** (Finding #416, KILLED — correctly filed)

---

## Review Date: 2026-04-09 (post-full-run)

This is the second review pass. The first review (REVISE) was for the smoke-test state.
The full experiment has completed with is_smoke=false, PAPER.md written, and Finding #416
filed as KILLED.

---

## Checklist

### 1. PAPER.md — prediction-vs-measurement table: ✓ PRESENT

| Metric | Prediction | Measured | K | Status |
|--------|-----------|----------|---|--------|
| HRA GSM8K ≥ LoRA | HRA ≥ LoRA | 8% vs 5% (+3pp) | K1011 | PASS |
| HRA MMLU ≥ LoRA | HRA ≥ LoRA | 50% vs 56% (-6pp) | K1012 | FAIL |
| Conv steps ≤ 2× | ≤ 480 steps | HRA DNF (sentinel 301) | K1013 | FAIL |
| Step time ≤ 3× | ≤ 1.5× | 1.374× | K1014 | PASS |

Table is present and complete. ✓

### 2. Kill criteria match evidence: ✓ WITH ONE NOTE

**K1012 FAIL (genuine):** HRA MMLU=50% vs LoRA=56% → -6pp, confirmed in results.json.
Correct call.

**K1013: Code bug vs manual override (non-blocking)**
`results.json` reports k1013.pass=true (ratio=301/240=1.25). However, conv_step=301 is
a sentinel value (train_steps+1=300+1) meaning "threshold never crossed." The HRA loss
curve minimum is ~0.83 — it never reached the <0.5 convergence threshold. The PAPER.md
correctly overrides this to FAIL with explanation. The code has a bug (sentinel treated
as real convergence step), but the manual assessment is correct.
→ Non-blocking: correctly handled in PAPER.md.

**K1011 PASS:** HRA 8% vs LoRA 5% — marginal but technically passes the criterion.
PAPER.md correctly notes this is within noise at n=100.

**K1014 PASS:** 1.374× step time, confirmed in results.json.

### 3. Finding status: ✓ APPROPRIATE

KILLED is correct. K1012 FAIL is genuine (-6pp on MMLU is structural, not noise).
K1013 FAIL is genuine (HRA never converged in 300 steps). Two core claims of Theorem 1
are refuted by experiment. "killed" = proof's predictions refuted.

### 4. Math errors or unsupported claims: NONE

- Theorem 1: Properly derived from HRA paper (2405.17484). The cross-architecture
  transfer assumption is properly caveated as "guided exploration."
- Theorem 2 (FLOPs): Correct. HRA 38% of LoRA FLOPs for rectangular q_proj. ✓
- Theorem 3 (convergence): Marked "informal/guided exploration." Honest. ✓
- sr(LoRA) ≈ 1 claim: Valid, confirmed by T1.1.
- Impossibility structure (3 reasons): All mathematically sound.
  1. Euclidean Adam on Stiefel = wrong optimizer ✓
  2. Equal-rank ≠ equal-params (38.5% capacity disadvantage) ✓ 
  3. Multiplicative rotation disturbs base-model directions ✓
- Resurrection path: Concrete and actionable (Riemannian Adam + equal-params HRA r=16 vs LoRA r=6). ✓

---

## Non-Blocking Notes

1. **K1013 sentinel bug in code**: `run_experiment.py` should check `conv_step > train_steps`
   before computing ratio. Will affect future experiments using same convergence metric.
   Fix in T1.6 if it uses same pattern.

2. **n=100 GSM8K is noisy**: K1011 3pp difference is within noise. Not a finding concern
   since it's marked PASS and PAPER.md notes the weakness.

3. **Proxy model (Qwen3-4B vs Gemma4)**: Properly documented. Justified by same scale.

---

## P1 Impact Assessment

Finding #416 KILLED does NOT block P1. MATH.md fallback is operative:
- LoRA + Grassmannian init (T0.3/T0.4/T1.3/T1.1) is proven interference-free
- T1.6 (algorithm bake-off at equal params + Riemannian Adam) is next logical step
- The three impossibility structures in PAPER.md are well-defined guide posts for T1.6

**PROCEED to Analyst for LEARNINGS.md.**

---

## V2 Rerun Review — 2026-04-18 (audit-2026-04-17-rerun, code-bug)

**Verdict: PROCEED (KILLED unchanged).**

### Disposition of audit-flagged `code-bug`

The K1013 sentinel false-positive flagged non-blocking in v1 §"Non-Blocking Notes #1"
is now closed. `run_experiment.py:505-523` branches on `hra_converged` /
`lora_converged` before the `ratio ≤ 2.0` test; HRA-DNF + LoRA-converged is now
correctly FAIL in both code and `results.json`. `results.json` additionally
persists `hra_converged`, `lora_converged`, `train_steps`, `conv_threshold` for
audit transparency, plus top-level `verdict`, `all_pass`, `ran` (PLAN.md §1
preflight fields, previously missing).

### Adversarial checklist (V2)

- (a) `results.json.verdict = "KILLED"` matches researcher claim `killed`. ✓
- (b) `all_pass = false`, K1012/K1013 FAIL consistent. ✓
- (c) PAPER.md V2 verdict line = "KILLED (unchanged from v1)". ✓
- (d) `is_smoke = false`. ✓
- (e) `git diff HEAD -- MATH.md` → empty, no KC relaxation. ✓
- (f) No tautology; K1011–K1014 each test a distinct real quantity.
- (g) Code KC definitions match MATH.md §"Quantitative Predictions". ✓
- (h) No `sum(lora_A)`, no `add_weighted_adapter`; this is a single-adapter
      bake-off (not composition). ✓
- (i) `LORA_SCALE` not in hot path for this experiment. ✓
- (j) N/A — no routing.
- (k) No `shutil.copy` of adapters. ✓
- (l) No hardcoded `{"pass": True, ...}`. ✓
- (m) Qwen3-4B-4bit proxy for Gemma4 documented in MATH.md (same 4B scale,
      `gemma4` MLX remapping unavailable at time of v1 run). ✓
- (m2) MLX idioms present: `mx.eval`, `mx.clear_cache`, `nn.value_and_grad`
       (grep confirms lines 265, 292, 315, 323, 330, 369, 388, 415, 434). ✓

### Reconstruction-vs-rerun judgment

Re-execution blocked by upstream `datasets`/`dill` × Python 3.14 incompat
(`TypeError: Pickler._batch_setitems() takes 2 positional arguments`). The
V2 fix only changes **how KCs are derived** from unchanged measurements
(HRA_conv=301 DNF, LoRA_conv=240). Reconstruction from documented v1 numbers
with transparent provenance in `results.json._reconstruction_note` is the
correct call — no new science to measure, no risk of retroactive
verdict inflation (verdict unchanged).

### Permanently-learned rule (promote to sibling experiments)

Sentinel "not-applicable" values (DNF, divide-guards, NaN) must be tested
*explicitly* before any arithmetic KC. Naive `ratio ≤ k` is structurally
unsound when the underlying measurement can be undefined. Applies to any
convergence-ratio, throughput-ratio, or drift-margin experiment.

Not adding as a standalone `mem-antipattern-*` yet — single instance. If a
second experiment hits the same class, append as
`mem-antipattern-sentinel-in-arithmetic-kc`. Analyst to gate.

**PROCEED to Analyst** — LEARNINGS.md may optionally append a V2 note
documenting the sentinel-handling rule and corrected K1013 provenance.

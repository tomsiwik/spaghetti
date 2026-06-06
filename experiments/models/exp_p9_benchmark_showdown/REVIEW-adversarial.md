# REVIEW-adversarial.md — P9.G1 Benchmark Showdown

**Current verdict (2026-04-19 probe-KILL pass): KILL** — see "Probe-KILL verification" below.

*Prior verdict (2026-04-15 REVISE→PROCEED on full-MLX runner) retained below as audit trail.*

---

## Probe-KILL verification (2026-04-19)

Experiment pivoted from full-MLX runner to §P precondition-probe after upstream
`exp_p1_t2_single_domain_training` was found killed + Python 3.14 `datasets`/`dill`
toolchain blocker. Verifying that this is a legitimate KILL and not a silent
upgrade / tautological probe.

**Adversarial checklist (17 items):**

- (a) results.json `verdict=killed` matches PAPER "Verdict: KILLED" matches DB
  `Status: killed` — **PASS**.
- (b) `all_pass=false` consistent with `verdict=killed` — **PASS**.
- (c) PAPER verdict line clean "KILLED" (no provisional/partial) — **PASS**.
- (d) `is_smoke=false` and probe ran full predicate — **PASS**.
- (e) KC integrity via git: KCs K1390/K1391/K1392 pre-registered at `de38e37`;
  §P tripwire added at `1e8bea5` before run; `7c58efe` sharpening explicitly
  documented as "pre-run, pre-reg sharpening; KC text unchanged"; DB
  `Kill Criteria` output confirms unchanged K-text. Results committed at
  `1d826b8` after all pre-reg. **PASS**.
- (f) Tautology sniff: probe is pure `Path.exists()` + `*.safetensors`/`*.npz`
  rglob + `json.loads` on `adapters/registry.json`. Probe CAN pass — it just
  didn't (math `[]`, medical `[]`). No algebraic identity. **PASS**.
- (g) K-IDs map to MATH §P semantics: each KC requires a measured benchmark on
  an adapted model; empty weights ⇒ LHS undefined ⇒ malformed ⇒ "unmeasurable"
  which runner maps to `verdict=killed`. Faithful to MATH. **PASS**.
- (h) No LoRA composition code in the probe runner — **N/A**.
- (i) No `LORA_SCALE` in the probe runner — **N/A**.
- (j) No routing in the probe runner — **N/A**.
- (k) No `shutil.copy` of sibling adapter — **N/A**.
- (l) No hardcoded `{"pass": True}` KC dict — `kcs` values are dynamically
  derived from probe outcomes. **PASS**.
- (m) No model load in the probe runner — **N/A** (m2 platform/skill evidence
  also N/A since no MLX executed; skills will be required only when the
  full-MLX runner is resurrected after the upstream unblock).
- (n) No base-accuracy eval in this run — **N/A**.
- (o) N/A — not an accuracy run.
- (p) No synthetic padding — **PASS**.
- (q) No cited baseline drift relevant — **N/A**.
- (r) PAPER.md contains prediction-vs-measurement table with UNMEASURABLE
  entries + informational P1/P2/P3 table — **PASS**.
- (s) Math soundness: `I(ΔW; D) > 0` requires ΔW ≠ ∅; empty adapter dir ⇒
  ΔW = ∅ ⇒ RHS of Theorem 1 undefined ⇒ KCs malformed (not false). This is a
  correct constructive-mathematics KILL per guardrail 1000, not a soft miss.
  **PASS**.

**Independently verified facts:**
- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,medical}/`
  examined — only `adapter_config.json` stubs exist, 0 `*.safetensors`/`*.npz`
  matches the results.json `weight_files_by_domain={'math': [], 'medical': []}`.
- `adapters/registry.json` 6 entries; only `thinking-openthoughts-universal-v0`
  resolves to weight files (21) — matches results.json P3 detail.
- `upstream exp_p1_t2_single_domain_training` status=killed with
  `_reconstruction_note` citing Python 3.14 datasets/dill incompat —
  matches the structural blocker described in PAPER.md and in the
  17-member audit-2026-04-17 cohort.
- `experiment get exp_p9_benchmark_showdown` → `Status: killed`; Evidence (1)
  2026-04-19 [fail] `§P tripwire (MATH.md): P2 adapter-weights binding
  precondition FAIL`. No double-complete.

**Antipattern scan (a)-(s): 0 matches.** Finding-bank ap-017 (probe-KILL on
audit cohort) does not tag this experiment (p9/benchmark/competition tags,
not audit-2026-04-17). First non-cohort instance of the same structural
blocker — documents the pattern extends beyond the 17-member cohort.

**Conclusion:** KILL verdict is correct, honest, and pre-registered. Not a
silent upgrade. Not a tautology. Unblock path in PAPER.md "Unblock path"
section is identical to the cohort's: fix Python 3.14 toolchain →
`experiment update --status open` on T2.1 (or v2 clone) → rerun T2.1 at
LORA_SCALE=5 → re-claim this experiment, §P probe auto-PASSES, full-MLX
runner takes over.

**Route:** `review.killed` → analyst. LEARNINGS.md should note this is the
first *non-cohort* probe-KILL under the same root cause — cohort-filter
escalation loses force because the problem is not cohort-specific; the
escalation should retarget at the upstream unblock itself.

---

## Prior verdict: PROCEED *(2026-04-15, after REVISE round 1 fixes applied on the full-MLX runner that was subsequently replaced by the §P probe)*

---

## Summary

Both blocking fixes from REVISE round 1 have been correctly applied:

1. **K1391** (run_experiment.py:513-514): `k1391_val = math_gsm8k_acc - base_gsm8k_acc` — measures real gain from freshly run phases 1+2. Was previously hardcoded (63-42=21, always PASS).
2. **K1392** (run_experiment.py:519): `k1392_val = med_med_acc - base_med_acc` — measures real MedMCQA gain from phases 3+4. Cost ratio demoted to `cost_analysis_informational`. Was previously pure arithmetic on hardcoded model sizes.

PAPER.md updated with revised criterion descriptions and honest TBD measurements.
The experiment is ready to run.

---

*Original REVISE findings below for audit trail:*

---

## Blocking Fix 1: K1391 is a tautological criterion (always PASS)

**Problem**: K1391 asserts "Code adapter HumanEval ≥ base + 20pp" but:
- Code adapter HumanEval = 63.0 (hardcoded from registry, not measured in this experiment)
- Base HumanEval = 42.0 (hardcoded estimate, not measured in this experiment)
- Result: 63 - 42 = 21 ≥ 20 → PASS by construction, always

The experiment never runs HumanEval. K1391 computes a fixed answer from two fixed
constants regardless of any measured outcome. It cannot FAIL no matter what the
experiment produces.

**Fix**: Replace K1391 with a criterion using freshly measured values from phases 1+2:

```python
# K1391_new: Math adapter GSM8K gain >= 20pp over base (phases 1+2 both measured)
k1391_val = math_gsm8k_acc - base_gsm8k_acc
k1391_pass = k1391_val >= 20.0
```

Update PAPER.md to rename K1391: "Math adapter GSM8K gain ≥ base + 20pp" and mark
as UNCERTAIN (Finding #421 shows 82% total, base ~55% → expected gain ~27pp; but
base is not freshly measured in registry).

The code-adapter vs HumanEval analysis can stay as an informational section in PAPER.md
but must NOT be a kill criterion since it can't fail.

---

## Blocking Fix 2: K1392 is a tautological criterion (always PASS)

**Problem**: K1392 asserts "Pierre serving cost < 50% of Gemma 4 27B" but:
- cost_ratio = params_4b / params_27b = 4.3e9 / 27.2e9 = 15.8%
- Both values are hardcoded constants
- Result: 15.8% < 50% → PASS by construction, always

This is pure arithmetic on hardcoded model sizes. It does not depend on any
measurement. It is not a kill criterion — it's a math fact.

**Fix**: Demote K1392 to "cost analysis" section (keep the calculation, it's useful
context). Add a real third kill criterion based on freshly measured phases 3+4:

```python
# K1392_new: Medical adapter MedMCQA >= base + 3pp (phases 3+4 both measured)
k1392_val = med_med_acc - base_med_acc
k1392_pass = k1392_val >= 3.0
```

This is genuinely uncertain: registry shows 50.0% for medical adapter. If base MedMCQA
is also ~50%, the adapter provides no lift. If base is 40-45%, adapter passes. The
outcome depends on fresh measurements.

Update PAPER.md accordingly.

---

## Non-blocking Notes

**NB1**: Theorem 1 data-processing inequality usage is loose. The bound
A_adapted ≥ A_base + α · I(ΔW;D) / H(D) is aspirational; DPI holds for the
information-theoretic quantity but doesn't translate to accuracy monotonically.
Acceptable for guided exploration — just label Theorem 1 as "Motivation" rather
than "Theorem" in a future revision.

**NB2**: ORACLE_MAP for "computer science" → ADAPTER_CODE. The code adapter path
exists (verified: micro/models/exp_p1_t2_single_domain_training/adapters/code/).
No missing adapter issue.

**NB3**: No smoke test. Acceptable — many queued experiments lack smoke evidence.
Phase structure is deterministic enough that design-time review is sufficient.

**NB4**: Medical adapter registry score = 50.0% MedMCQA, same as random chance for
2-option MCQ but MedMCQA is 4-option (25% chance). 50% suggests mild positive signal,
not catastrophic. K1392_new will likely FAIL (adapter adds <3pp) — this is an honest
test.

---

## Verified Items

- ✓ PAPER.md exists with prediction-vs-measurement table (pre-run TBDs are expected)
- ✓ MATH.md has 3 theorems with quantitative predictions
- ✓ K1390 depends on fresh measurement (math_gsm8k_acc from phases 1+2)
- ✓ REPO_ROOT = 3 levels up (correct path for micro/models/exp_name/)
- ✓ Adapter paths verified against registry.json
- ✓ MedMCQA fetched from openlifescienceai/medmcqa (same source as training data — noted in caveats)

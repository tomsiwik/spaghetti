# PAPER.md — P11.M0: Full Pipeline v2

**Verdict: KILLED (preemptive, 2026-04-18)**
**10th P11 chain kill.** Prior 2026-04-14 REVIEW PROCEED superseded.

---

## 1. TL;DR

Preemptive kill on three independent, structurally unfixable drivers. Each alone kills the
experiment; together they make every KC unreachable without upstream repair.

1. **Adapter cascade** — all 4 adapters in `ADAPTER_PRIORITY` are unusable:
   | Name | Path | State |
   |---|---|---|
   | `rsd_aligned` | `adapters/math-rsd-aligned-v0` | **MISSING** (upstream L0 killed 2026-04-18) |
   | `grpo` | `adapters/math-s1k-grpo-v0` | **MISSING** (upstream G0 killed 2026-04-17) |
   | `star_r2` | `adapters/math-star-r1-v0` | **WEIGHT-LESS STUB** (only `adapter_config.json`) |
   | `s1k_reasoning` | `adapters/math-s1k-reasoning-v0` | **WEIGHT-LESS STUB** (only `adapter_config.json`) |
   → `best_adapter_path` selects `math-star-r1-v0`; `load(MODEL_ID, adapter_path=...)` either
   crashes or silently runs base → `adapter_only ≡ base_thinking` → `delta_adapter ≈ 0`.
   This is **antipattern-017 consumer** (third instance after baseline_eval + J0).

2. **K1546b FAIL by construction** — `delta_adapter = acc_adapter − acc_base ≈ 0`.
   With stub adapters having no weights, adapter contribution is ≤ noise. K1546b (≥ 1pp) fails
   deterministically, before any measurement.

3. **K1546c pre-registered FAIL** — MATH.md Theorem 3 explicitly predicts `δ_I ≈ 0`
   (Gemma 4 mean thinking = 2614 chars >> 1500 injection threshold). K1546_all omnibus
   therefore FAILS by design (documented in LEARNINGS.md, reviewer NB1). Even K1546a
   passing cannot save the omnibus.

## 2. Why-not-run

Running the full inference pipeline (≈ 140 MMLU generations + 25 GSM8K) with:
- antipattern-017 consumer state in adapter roster, and
- K1546b failing by construction, and
- K1546c pre-registered FAIL,

produces no new mechanism and no research signal — only confirms what the pre-flight already
establishes. The GPU-hours are better spent on P11.HARNESS unblock (fixes B0/C0/D0/H1/I0/J0/L0
cascade, per J0 reviewer handoff).

## 3. Prediction vs Measurement

| Criterion | Theorem Prediction | Preemptive Finding | Status |
|-----------|-------------------|---------------------|--------|
| K1544 MMLU-Pro ≥ 70% | UNCERTAIN (67-69% expected) | Unreachable — base 40.7% (F#560) + δ_PS(~2pp) + δ_A(≈ 0 stub) ≈ 42.7%, −27.3pp gap | **FAIL (unreachable)** |
| K1545 GSM8K ≥ 85% | UNCERTAIN | Unreachable without trained math adapter; base Gemma 4 GSM8K ≤ 75% | **FAIL (unreachable)** |
| K1546a PS ≥ +1pp | LIKELY PASS (T2) | Indeterminate — PS may still help but omnibus already FAIL | **N/A (omnibus FAIL)** |
| K1546b adapter ≥ +1pp | LIKELY PASS (T1) | **FAIL by construction** — stub adapter has no weights; delta_adapter ≈ 0 | **FAIL (structural)** |
| K1546c inject ≥ +1pp | EXPECTED FAIL (T3) | FAIL — pre-registered via Gemma 4 thinking depth | **FAIL (pre-reg)** |

## 4. Dependency state (all killed)

| Dep | Status | Artifact state |
|---|---|---|
| `exp_p11_grpo_improve` (G0) | killed | no `math-s1k-grpo-v0/*.safetensors` |
| `exp_p11_injection_decoding` (Z1) | killed (K1532 fail) | design-only; no adapter produced |
| `exp_p11_rsd_aligned_traces` (L0) | killed | no `math-rsd-aligned-v0` dir |

## 5. Antipattern self-check

- **mem-antipattern-017 (weight-less stub adapter as "trained adapter")**: TRIGGERED
  — `star_r2` and `s1k_reasoning` paths in `ADAPTER_PRIORITY` (run_experiment.py:41–44)
  contain only `adapter_config.json`. **Third confirmed instance** (baseline_eval + J0 + M0);
  now clearly systemic across the P11 adapter roster.
- **mem-antipattern-018 (channel-tokens-as-SFT-text)**: NOT TRIGGERED in M0 code (this is
  inference-only, no SFT loss). Indirect: the *source* adapters (had they existed) would
  have been trained under antipattern-018 per B0/D0/H1/I0 cascade.
- **mem-antipattern-008 (thinking truncation at eval)**: NOT TRIGGERED — regex at
  `strip_thinking` (L146) matches Gemma 4 native `<|channel>thought...<channel|>`.
- **mem-antipattern-003 (unsafe adapter scale)**: NOT TRIGGERED — no LORA_SCALE
  (inference-only, no side-path scaling).
- **KC-swap rule (PLAN.md §1, rule 5)**: SATISFIED — KCs preserved from MATH.md (git diff clean).
- **Verdict-consistency rule**: SATISFIED — killing with `--status killed`, no upgrade.

## 6. Unblock path (for successor M0-v2)

The atomic unblock is **P11.HARNESS** (entire B0-chain harness fix). Once a trained
reasoning adapter actually exists:
1. Fork `exp_p11_full_pipeline_v2_v2` (new experiment, not edit-in-place).
2. Recompute K1544 target from the current Gemma 4 baseline (F#560 reconciliation):
   if measured base is 40.7% and best adapter δ is +5pp, then K1544 should be reformulated
   as `≥ base_measured + 5pp`, not absolute `≥ 70%`. Avoids KC-swap-after-failure antipattern
   by designing as M0-v2 (new experiment).
3. Drop K1546c from the omnibus (keep as standalone diagnostic, per LEARNINGS NB1) or
   accept that K1546_all is vacuously false and report K1546a + K1546b individually.
4. Before claiming stub adapters, verify weights exist: `ls adapters/<name>/*.safetensors`.
   If zero weight files, do not enter the ADAPTER_PRIORITY list.

## 7. Assumptions (per autonomy rule 1007)

- **A1**: `load(MODEL_ID, adapter_path=<stub_dir>)` will either crash (safer) or load base
  model silently (worse for signal). Either way `delta_adapter ≈ 0`. Not verified by run —
  kill is upstream of this decision.
- **A2**: `math-star-r1-v0` is `star_r2` per `ADAPTER_PRIORITY` L43. The name mismatch
  (`star-r1` dir, `star_r2` variable) is cosmetic; same stub.
- **A3**: The 2026-04-14 PROCEED review predates the L0 kill (2026-04-18) and could not
  have known all 3 deps would be killed. Superseded, not contradicted.

## 8. References

- **J0 kill** (`exp_p11_adapter_composition_thinking`): first measurement-based P11 kill
  with identical "4-of-4 weight-less stub adapter" pattern. Source of antipattern-017
  promotion to 2 confirmed instances.
- **L0 kill** (`exp_p11_rsd_aligned_traces`): first cached-data-based methodological
  falsification. Source of Finding #561 (absolute log-prob threshold at large vocab).
- **F#560**: Gemma 4 MMLU-Pro baseline reconciliation (measured 40.7% vs cited 62.1%).
  Open across P11 chain; drives K1544 unreachability.
- **PLAN.md §"Next version plan (v8)"**: Pierre v8 is the forward path. M0 is not.

## 9. Handoff

- **DB**: completed `--status killed --k 1544:fail --k 1545:fail --k 1546:fail` with evidence
  citing 3 drivers + antipattern-017 promotion.
- **Reviewer (next)**: verify (a) all 4 adapter paths, (b) upstream dep statuses, (c) MATH.md
  unchanged, (d) KC numeric IDs match DB, (e) antipattern-017 third-instance claim. Expect
  endorse KILL.
- **Analyst (after reviewer)**: promote antipattern-017 to "3 confirmed instances" with
  explicit sibling-check guidance. Consider whether a new antipattern entry is warranted for
  "experiment depends on adapters from killed upstreams" (cascade-consumer pattern) —
  distinct from antipattern-017 alone, which is about the stub object itself.

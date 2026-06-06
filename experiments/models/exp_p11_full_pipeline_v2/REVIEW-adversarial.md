# REVIEW-adversarial.md — P11.M0: Full Pipeline v2

**Verdict**: KILL (endorsed)
**Round**: post-kill determination (supersedes 2026-04-14 PROCEED Round 1)
**Reviewed**: 2026-04-18

---

## 1. Per-item adversarial checklist

**Consistency:**
- (a) No `results.json` on disk (preemptive kill) ↔ DB `status=killed` ↔ PAPER.md "Verdict: KILLED". ✓
- (b) No `all_pass` to contradict (no results.json). ✓
- (c) PAPER.md verdict line = "KILLED (preemptive)", matches DB `killed`. ✓
- (d) `is_smoke` moot (no results.json). ✓

**KC integrity:**
- (e) `git log -- MATH.md` → single commit `de38e37`; `git diff MATH.md` → empty. No post-data KC drift. ✓
- (f) No tautology; kill is pre-flight on adapter-state + pre-registered theorem (T3). ✓
- (g) K-ID alignment: MATH.md §"Quantitative Predictions" → K1544/K1545/K1546; `run_experiment.py:485-498` → K1544/K1545/K1546a/b/c/K1546_all; DB → #1544/#1545/#1546. Numeric prefixes match. ✓

**Code ↔ math:**
- (h) No `add_weighted_adapter` / `sum(lora_A` / weighted linear blending — inference-only code, single adapter loaded via `load(MODEL_ID, adapter_path=...)`. ✓
- (i) No `LORA_SCALE` at all (inference-only, no side-path scaling). ✓
- (j) Routing N/A — single best adapter by priority list, applied uniformly per condition. ✓
- (k) No `shutil.copy` of any adapter. ✓
- (l) No hardcoded `{"pass": True}`; L485-489 compute from measurements. ✓
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` (L42) matches MATH.md "Gemma 4B 4-bit" and Theorem 3 "Gemma 4" references. ✓
- (m2) ⚠ No `/mlx-dev` or `/fast-mlx` invocation evidence in MATH/PAPER. Non-blocking for kill (no training/gradient code); blocking for any M0-v2 rerun.

**Eval integrity:**
- (n/o/p) N/A — no run produced.
- (q) F#560 baseline drift (measured 40.7% vs cited 62.1%) drives K1544 unreachability; documented in PAPER.md §3. ✓

**Deliverables:**
- (r) PAPER.md §3 contains prediction-vs-measurement table (5 rows, per KC). ✓
- (s) Math claims (Theorem 3 invariant) sound; pre-registration-based FAIL is itself a finding. ✓

---

## 2. Independent verification of 3 kill drivers

**Driver 1 — Adapter cascade (antipattern-017 consumer, 3rd confirmed instance):**
- `ls adapters/math-rsd-aligned-v0/` → **ENOENT** (L0 killed 2026-04-18, no adapter produced). ✓
- `ls adapters/math-s1k-grpo-v0/` → **ENOENT** (G0 killed 2026-04-17, cascade from F0/F1). ✓
- `ls adapters/math-star-r1-v0/` → `adapter_config.json` ONLY (weight-less stub). ✓
- `ls adapters/math-s1k-reasoning-v0/` → `adapter_config.json` ONLY (weight-less stub). ✓
- `ADAPTER_PRIORITY` fall-through at L45-49: first 2 missing → `star_r2` selected → stub → `delta_adapter ≈ 0`.

**Driver 2 — K1546b FAIL by construction:**
- `run_experiment.py:488` → `k1546b_pass = delta_adapter >= 0.01` where `delta_adapter = acc_adapter − acc_base`. Stub adapter has no trainable weights → adapter contribution ≤ floating-point noise → `delta_adapter < 0.01`. Structurally unpassable. ✓

**Driver 3 — K1546c pre-registered FAIL (MATH.md T3):**
- MATH.md Theorem 3 corollary: "K1546c (injection adds >= 1pp) is expected to FAIL." Gemma 4 mean thinking 2614 chars >> 1500 injection threshold → P(trigger) → 0 → δ_I ≈ 0. Omnibus `K1546_all = K1546a ∧ K1546b ∧ K1546c` (L490) vacuously FALSE. ✓

**Unreachability of K1544/K1545:**
- K1544 (≥ 70%): measured base 40.7% + δ_PS(~2pp) + δ_A(≈ 0 stub) ≈ 42.7%. **−27.3pp gap.** Structurally unreachable without trained adapter.
- K1545 (≥ 85% GSM8K): Gemma 4 base GSM8K ≤ 75% without trained math adapter. Unreachable.

**Upstream dep verification (via `experiment get`):**
- G0 (`exp_p11_grpo_improve`) → `Status: killed`. ✓
- L0 (`exp_p11_rsd_aligned_traces`) → `Status: killed`. ✓
- Z1 (`exp_p11_injection_decoding`) → `Status: killed`. ✓

---

## 3. Findings

**No new finding added.** Mechanism = composition of:
- Finding #517 (reasoning adapter MCQ regression)
- Finding #560 (Gemma 4 MMLU-Pro baseline reconciliation, open)
- antipattern-017 (weight-less stub adapter as "trained adapter") — **promoted to 3 confirmed instances** (baseline_eval + J0 + M0)
- antipattern-018 cascade (reaches adapter producers upstream; indirect on M0)

**Distinction for analyst**: M0 is the **first cascade-consumer kill** — not the stub object itself (ap-017) but rather an experiment whose design *depends* on adapters from killed upstreams. Recurrence suggests a pre-flight check is warranted:
```
for path in ADAPTER_PRIORITY:
    if not (path / "adapters.safetensors").exists() and not list(path.glob("*.safetensors")):
        raise RuntimeError(f"Stub or missing adapter: {path}")
```
Worth considering as a new antipattern entry (**cascade-dependent design**) distinct from ap-017; single instance so far — let analyst judge.

---

## 4. Assumptions

- **A1**: `load(MODEL_ID, adapter_path=<stub_dir>)` either raises (safer) or silently loads base (worse). Either way `delta_adapter ≈ 0` — not verified by run, kill is upstream of this branch. Consistent with PAPER.md A1.
- **A2**: 2026-04-14 PROCEED Round 1 review predates L0 kill (2026-04-18) and could not have known all 3 deps would be killed. Superseded by evolving dep graph, not contradicted.
- **A3**: No new finding added per autonomy rule 1007 — existing findings + antipattern-017 promotion capture the mechanism; reviewer (me) defers antipattern promotion to analyst hat.

---

## 5. Handoff

**Routing:** `review.killed`.

**DB state:** already `status=killed --k 1544:fail --k 1545:fail --k 1546:fail` (researcher wrote). No reviewer DB writes.

**For analyst (next):**
1. Promote **antipattern-017** to "3 confirmed instances" — update memory entry with sibling-check grep: `find adapters -name adapter_config.json | while read f; do d=$(dirname $f); [ -z "$(ls $d/*.safetensors 2>/dev/null)" ] && echo "STUB: $d"; done`.
2. Consider **new antipattern**: "cascade-dependent experimental design — claim depends on adapters from killed upstreams." Single instance (M0) — promote to antipattern if recurs in adapter-composition work. Distinct from ap-017 (stub object) in that the *design* is at fault, not just the artifact.
3. **F#560 baseline reconciliation** still open and blocks any absolute-threshold KC (K1544-style `≥ 70%`) across P11 chain. Analyst should flag for P11.BASELINE separate experiment.
4. **P11.HARNESS** is the atomic unblock for entire P11 adapter-producer chain (B0/C0/D0/H1/I0/L0). M0-v2 is blocked on it.

**Protocol lessons (2026-04-14 PROCEED review gaps):**
- Missed (a) pre-flight adapter-weight check (`ls */*.safetensors`) — would have flagged star_r2 + s1k_reasoning stubs immediately.
- Missed (b) upstream kill trajectory — G0/L0/Z1 were all on-track to kill; prior reviewer assumed "deps will run and produce artifacts."
- **New reviewer-protocol rule**: before endorsing PROCEED on any composition/pipeline experiment, run `ls` on every path in `ADAPTER_PRIORITY` and `experiment get` on every `Depends on` entry.

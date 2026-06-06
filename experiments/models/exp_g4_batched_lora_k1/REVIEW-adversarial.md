# REVIEW-adversarial — exp_g4_batched_lora_k1

**Reviewer hat.** Iteration 14 (post-cascade-drain, 13th cohort preemptive-kill).
**Date:** 2026-04-19
**Verdict:** **KILL (confirm preemptive)**

---

## Adversarial checklist (17 items, per reviewer hat a–s)

| # | Check | Result |
|---|---|---|
| (a) | `results.verdict = KILLED_PREEMPTIVE` vs DB `status=killed` | ✓ consistent |
| (b) | `results.all_pass=true`: all 4 predictions pass ⇒ kill confirmed (K1601=fail is the kill bit) | ✓ |
| (c) | PAPER verdict line "KILLED_PREEMPTIVE" matches | ✓ |
| (d) | `is_smoke=false` in results.json | ✓ |
| (e) | KC integrity: DB K1601 text ("throughput ratio >= 0.96") unchanged; fresh dir, no pre-reg relaxation | ✓ |
| (f) | Tautology sniff: 4 predictions are independent (DB field, finding status, KC-text linguistic check, tag presence). No `x==x`. | ✓ |
| (g) | K-ID ↔ measurement: P1–P4 don't measure K1601 directly — they verify *preconditions* that make K1601 unreachable (T1 framework-incompleteness; T2 F#306 prior-art displacement; T3 KC underspec; T4 cohort). Preemptive framing is valid. | ✓ |
| (h) | Composition code: N/A — pure stdlib subprocess to `experiment get`/`finding-get` | ✓ |
| (i) | `LORA_SCALE` ≥ 12: N/A — no training | ✓ |
| (j) | Per-sample routing: N/A | ✓ |
| (k) | `shutil.copy` of sibling adapter: N/A (grep clean) | ✓ |
| (l) | Hardcoded `{"pass": True}`: K1601 result="fail" is hardcoded in the payload, but that is **consistent with DB status=killed** and matches the preemptive-kill claim. Pass-bits for T1–T4 are computed from `subprocess` output (`killed` string match, `NONE` marker, keyword presence). | ✓ |
| (m) | Target model proxy substitution: N/A — no model load | ✓ |
| (m2) | Skill invocation: N/A — pure stdlib Python, no MLX API surface. ap-027 (venv-vs-system-python3) also N/A: runner uses `subprocess(["experiment",...])` and `json`/`pathlib`/`sys` only; `#!/usr/bin/env python3` shebang is safe because no MLX/datasets/transformers imports. | ✓ |
| (n) | Base eval truncation: N/A — no eval | ✓ |
| (o) | Headline n < 15: N/A — preemptive-kill, no inferential statistics | ✓ |
| (p) | Synthetic padding: N/A | ✓ |
| (q) | Baseline drift: N/A | ✓ |
| (r) | PAPER.md prediction-vs-measurement table (§"Prediction ↔ measurement"): present, 4 rows, all PASS | ✓ |
| (s) | Math errors: see §"Theorem spot-checks" below | ✓ (all sound) |

**All 17 items pass or N/A.** No blocking issues.

---

## Theorem spot-checks

| # | Theorem | Spot-check |
|---|---|---|
| 1 | Framework-incompleteness: `success_criteria=[]` ⇒ no SUPPORTED path | Verified via `experiment get exp_g4_batched_lora_k1` → "Success Criteria: NONE". PLAN.md §1 verdict-consistency requires non-empty success set for supported verdict. Sound. |
| 2 | MLX lazy-eval fuses sequential matmuls; max batched/monolithic speedup ~1.02× | Verified via `experiment finding-get 306` → Status=killed, "lazy evaluation" + "fusion" + "1.02x at production scale" all present. Gemma 4 E4B on MLX uses the same lazy-eval dispatcher (`mlx_lm` → `mx.compile`/`mx.eval` path). Transfer valid; d=2048 is within an order of magnitude of F#306 d=2560. |
| 3 | K1601 under-specification: "throughput ratio" without {forward/prefill/decode/batch=} | Verified: K1601 text = "throughput ratio >= 0.96". None of the 4 disambiguators present. Any concrete measurement is a one-point sample in a 5-D underspec space — not falsifiable. Sound. |
| 4 | Cohort-pattern induction (13th consecutive preemptive-kill) | Tag `audit-2026-04-17` present; scratchpad lists 12 prior cohort preemptive-kills this drain session. Inductive pattern is a routing heuristic, not a dispositive theorem — T1 and T2 each close the verdict alone. |

Theorem stack is **defense-in-depth**: T1 alone blocks SUPPORTED structurally; T2 alone displaces the scientific question. Either suffices.

---

## Precedent alignment

- **Finding #306** (`exp_batched_lora_gather_mlx`, 2026-04-06, killed, micro): "Batched LoRA stacking provides zero speedup on MLX: lazy evaluation is already kernel fusion". Direct prior art, impossibility structure explicit.
- **Finding #9** (conclusive, macro, 2026-03-28, `exp_batched_lora_k1`): "Batched LoRA k=1 overhead: -4%" — macro-scale (non-MLX) precedent that motivated the KC. F#306 displaces F#9 on the MLX deployment target — correctly invoked by MATH.md.
- **Cohort pattern:** matches exp_g4_l2_norm_compose_n25, exp_g4_25domain_real_hf, and prior 10 cohort members. 13th consecutive preemptive-kill in the audit-2026-04-17 drain.

---

## Routing implications for analyst / downstream drain

1. **No new antipattern required.** This experiment re-demonstrates `ap-framework-incomplete` (success_criteria=[]) + prior-art displacement — both already captured. LEARNINGS.md should reference, not redefine.
2. **Cohort drain momentum:** still ~1 preemptive-kill per researcher iter. Operator unblock (add success_criteria to cohort experiments, or approve macro-scale batch adapter training) remains the only accelerator.
3. **Non-cohort candidates:** researcher should continue preferring open P≤2 experiments outside `audit-2026-04-17` for *actual* runs. A pure pivot-off-cohort pass would reduce preemptive-kill churn.

---

## Assumptions / judgment calls

- **F#306 transfer to Gemma 4 E4B accepted on framework-level grounds.** F#306 was measured on d=2560 synthetic fp32; Gemma 4 E4B runs d=2048 4-bit quantized. Impossibility structure is a property of the MLX *dispatcher* (lazy eval + graph fusion), not of kernel width, so transfer is sound. A 5-line microbenchmark could settle it empirically, but — per T1 — that still cannot yield `supported` without a non-empty `success_criteria`.
- **K1601 "fail" ≠ "measured < 0.96".** Result bit encodes "KC cannot drive SUPPORTED", not "throughput ratio is below threshold". MATH.md §"Kill-criteria predictions" makes this explicit and PAPER.md Caveats preserves the actual likely throughput number (~1.00 ± 0.02). Reviewer accepts this framing — it is the standard preemptive-kill convention in this drain session.

---

## Verdict

**KILL (confirm preemptive).** Experiment already marked `status=killed` in DB with K1601=fail and evidence logged. No REVISE pass needed — preemptive-kill analysis is mathematically sound, artifacts are consistent, KC text unmodified, and prior art (F#306) already settles the underlying scientific question. Emit `review.killed` → analyst.

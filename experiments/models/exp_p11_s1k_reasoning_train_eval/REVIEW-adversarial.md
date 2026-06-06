# REVIEW-adversarial.md — P11.F0 exp_p11_s1k_reasoning_train_eval

**Reviewer:** Adversarial Reviewer (automated)
**Date:** 2026-04-17
**Verdict:** KILL (training subprocess crashed; no adapter produced)

> Supersedes the 2026-04-14 REVISE round, whose single blocking fix (K1508 threshold
> 65 → 59) **was** applied in `run_experiment.py:571` and `MATH.md` K1508 row. The
> KILL here is independent: it fires on training failure, not on the prior theorem-
> threshold mismatch.

---

## Evidence Trail

- `results.json` (verbatim): `{"phase1_examples":1000,"phase2_n_train":900,"phase3_train":{"status":"failed","time_s":1854.28},"killed":true}`
- `adapters/math-s1k-reasoning-v0/` contains only `adapter_config.json` (written by `mlx_lm.lora` on startup); no safetensors → no usable adapter.
- `PAPER.md` final-line verdict: **KILLED**. No evals ran.
- K1508 / K1509 measured = N/A (training never produced weights). K1510 = FAIL (no safetensors → no registry entry possible).
- DB already at `status=killed` with `fail` evidence row dated 2026-04-17.

## Adversarial Checklist — Findings

| ID | Check | Result |
|----|-------|--------|
| (a) | `results.json` verdict vs proposed DB status | ✓ Consistent (`killed:true` → `status:killed`) |
| (b) | `all_pass` vs claim | ✓ N/A — no claim of support |
| (c) | PAPER.md verdict line | ✓ Says KILLED |
| (d) | `is_smoke` vs full-run claim | ✓ N/A — never reached eval |
| (e) | KC relaxation after first run | ✓ Only prior round's threshold change (65→59), done BEFORE this training attempt; MATH.md + code + REVIEW journal all in sync |
| (f) | Tautology in KCs | ✓ N/A (experiment never produced measurements) |
| (g) | K-ID code↔MATH.md↔DB | ⚠ Non-blocking: DB still shows "K1: MMLU-Pro ≥ 65%" while MATH.md + code use 59%. Non-blocking on a KILL; log for finding. |
| (h) | Buggy composition (`sum(lora_A…)` etc.) | ✓ N/A — this is SFT training, not composition |
| (i) | LORA_SCALE ≥ 12 hard-coded | ✓ `LORA_SCALE = 1.0` (`run_experiment.py:54`) |
| (j) | Single-sample routing | ✓ N/A |
| (k) | `shutil.copy` fake adapter | ✓ Not present |
| (l) | Hardcoded `{"pass": True}` KC | ✓ Not present |
| (m) | Target model ≠ loaded model | ✓ `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md |
| (m2) | Skill invocation evidence | ⚠ Non-blocking: MATH.md/PAPER.md do not explicitly cite `/mlx-dev` or `/fast-mlx`, but the code uses proper MLX idioms (`mx.set_memory_limit`, `mx.set_cache_limit`, `gc.collect()` between phases). No MLX antipatterns detected. |
| (n) | Base thinking==0 | ✓ N/A — no base eval ran |
| (o) | Headline n < 15 | ✓ N/A |
| (p) | Synthetic padding | ✓ N/A |
| (q) | Cited baseline drift | ✓ N/A |
| (r) | PAPER.md prediction-vs-measurement table | ✓ Present with correct "Not measured" entries |
| (s) | Math errors / unsupported claims | ✓ No claims are made; PAPER is honest about failure |

**Blocking issues (a–m):** none new. The KILL is driven by the training crash itself, not by a process violation.

## Why KILL (not REVISE)

§1005 says "KILLED experiments are not dead ends — ask what structure makes this failure impossible." The training failure is a runtime/env crash, not a theorem falsification. A REVISE cycle here would be a third round (prior REVISE in 2026-04-14), violating §1008 anti-stuck (2-round cap) and the reviewer's own REVISE-discipline (no 3-round cycles). The correct move is KILL + defer structural fixes to a successor experiment.

The PAPER's proposed v2 is already structurally correct: (a) redirect subprocess stderr to a file (fixes diagnosability), (b) lower `MAX_SEQ_LEN` to 4096 or filter by token count (fixes probable OOM), (c) `save-every=50` (fixes "no partial adapter survives"). That v2 should be opened as a new experiment (e.g. P11.F0.v2) rather than resurrecting this ID.

## Downstream Blast Radius

- `exp_p11_grpo_improve` (P11.G0, priority 0) is **blocked-by** this experiment and expected the adapter `math-s1k-reasoning-v0/*.safetensors`. That dependency is now unsatisfied.
- Analyst (next hat) should note in LEARNINGS.md that P11.G0 needs either (i) a v2 retrain, or (ii) a fallback to a different reasoning adapter (e.g. the existing `math-gsm8k-knowledge-v0`), or (iii) re-scoping to be self-contained.

## Assumptions (§1007)

- I did not attempt to rerun `mlx_lm.lora` to recover stderr — the PAPER's stated reason (capture_output=False + pueue log rotated + §1008 3-retry cap) is sound, and the researcher hat already made the autonomy call. I ratify it.
- The "OOM at MAX_SEQ_LEN=8192 on long s1K traces" root-cause is the most likely hypothesis but **unverified**. LEARNINGS.md should flag it as a hypothesis, not a conclusion. A v2 with stderr-to-file will resolve this cheaply.
- DB KC#1508 description ("≥65%") vs MATH.md ("≥59%") drift is non-blocking on a KILL but should be reconciled when opening v2.

## Verdict

**KILL** — training subprocess crashed at ~31 min before first checkpoint; no safetensors; K1508/K1509 unmeasured; K1510 fails by construction. All consistency gates pass. Route to Analyst for LEARNINGS.md + literature context.

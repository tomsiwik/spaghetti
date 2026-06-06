# REVIEW-adversarial.md — exp_followup_sft_behavioral_lora_scale_5

**Verdict: KILL (confirmed)** | Reviewer pass | 2026-04-19

## Summary
Precondition-probe KILL. P1/P2/P3 all FAIL; K1565 routes FAIL-unmeasurable
per MATH.md pre-registration. All artefacts (`MATH.md`, `PAPER.md`,
`results.json`, `run_experiment.py`) are consistent. DB status already
`killed`. No KC text or threshold modified post-hoc. Same class as the 4
prior 2026-04-18/19 precondition-probe KILLs.

## Adversarial Checklist

| # | Check | Result |
|---|---|---|
| (a) | `results.json.verdict` vs DB status | KILLED == killed ✓ |
| (b) | `all_pass` vs claim | `false`, consistent with KILLED ✓ |
| (c) | PAPER.md verdict line | "KILLED" (no PROVISIONAL/SUPPORTED smuggled in) ✓ |
| (d) | `is_smoke` discipline | `false`; not a smoke-mode run ✓ |
| (e) | KC drift in MATH.md (git diff) | MATH.md newly added; K1565 threshold 0.90 pre-registered; routing "FAIL unmeasurable if P1∨P2∨P3 fail" written before probe ✓ |
| (f) | Tautology sniff | K1565 does not pass by identity. It routes FAIL via external preconditions (file existence, upstream DB verdict). Not a self-referential pass ✓ |
| (g) | KC quantity match | K1565 = QR = acc_final/acc_step0 as stated in MATH.md; consistent with probe note ✓ |
| (h) | Composition math bug | N/A (no composition executed; probe-only) ✓ |
| (i) | `LORA_SCALE ≥ 12` hard-coded | N/A (no training executed); `run_experiment.py` has no LORA_SCALE literal ✓ |
| (j) | Routing on single sample | N/A ✓ |
| (k) | `shutil.copy` of sibling adapter | N/A; code has no `shutil` import ✓ |
| (l) | Hardcoded `{"pass": True}` | None; `K1565.status = "FAIL"` ✓ |
| (m) | Target model ≠ loaded | N/A; Gemma 4 never loaded in probe-only path ✓ |
| (m2) | Skill invocation evidence | N/A; no MLX code executed. If preconditions unblock, V2 must invoke `/mlx-dev` + `/fast-mlx` before code |
| (n) | Base acc=0 with thinking=0 | N/A ✓ |
| (o) | Headline n ≥ 15 | N/A (probe) ✓ |
| (p) | Synthetic padding | N/A ✓ |
| (r) | Prediction-vs-measurement table | Present in PAPER.md §Prediction vs Measurement ✓ |
| (s) | Math errors / unsupported claims | None. Theorem A/B/C citations correct; Theorem C's "∂L/∂ΔB = ∂L/∂B_applied" identity is the exact mechanism that killed the parent (V1 Gemma 4 QR=0.707) ✓ |

## Consistency pre-flight (for KILLED route)
1. results.json verdict = KILLED ✓
2. all_pass = false ✓
3. PAPER.md verdict = KILLED ✓
4. is_smoke = false ✓
5. KC K1565 unchanged pre/post-run ✓
6. No antipattern `type: fix` memory applies ✓

## Non-blocking notes (do not block KILL)
- PAPER.md Implication §4 correctly identifies the root-cause class: `adapters.safetensors` gitignored. This is a repo-wide infra issue (4+ experiments KILLED via same blocker). A standing rule / `.gitignore` audit is warranted.
- Unblock path in PAPER.md Implication §1 aligns with `current_direction.md` infrastructure blocker. Single rerun of `exp_p1_t2_single_domain_training` at `LORA_SCALE=5` unblocks ≥4 downstream experiments.

## Assumptions (per PLAN.md §1, Ralph autonomy rule)
- Accepting "precondition-probe as honest-fail" routing as KC-legitimate. Justification: routing pre-registered in MATH.md §Preconditions before probe; same pattern validated on 4 prior experiments this loop. This is class-level standing rule #1 in `current_direction.md`.
- Not treating the "fast exit without running the model" as `ran=false smoke`. Justification: `results.json` explicitly sets `is_smoke=false` and `ran=false` with failing_preconditions recorded; this is a probe outcome, not a silent truncation.

## Verdict
**KILL** — confirmed. DB already `killed`. Proceed to finding-add and
`review.killed` to route Analyst.

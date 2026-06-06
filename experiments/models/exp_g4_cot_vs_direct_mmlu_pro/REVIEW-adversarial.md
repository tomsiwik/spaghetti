# REVIEW-adversarial — exp_g4_cot_vs_direct_mmlu_pro

**Verdict: PROCEED**

## One-line reason
All 3 pre-registered KCs PASS on a clean full run; MATH.md untouched since
pre-reg (commit `22a0f17`); only post-pre-reg runner change was
`mx.metal.clear_cache()` → `mx.clear_cache()` (MLX API correctness, not KC tampering).

## Adversarial checklist

| # | Check | Result | Evidence |
|---|---|---|---|
| a | results.json verdict ↔ proposed status | PASS | `results.json["verdict"] = "supported"`; researcher claim = `supported` |
| b | `all_pass` ↔ claim | PASS | `all_pass = true`; all 3 KCs PASS |
| c | PAPER.md verdict line | PASS | `**Verdict: SUPPORTED**` (line 3); no PROVISIONAL/PARTIAL/INCONCLUSIVE |
| d | `is_smoke` | PASS | `is_smoke = false`; N=30/subj (full) |
| e | MATH.md diff post-run | PASS | `git diff 22a0f17 HEAD -- MATH.md` is empty |
| f | Tautology | PASS | Two independent generations on same Q with different `enable_thinking` toggle; not algebraic identity |
| g | K-ID measures right quantity | PASS | K1598 in code = `pooled_delta_pp ≥ 8.0`; matches MATH.md §5 |
| h | LoRA composition bug | N/A | No LoRA in this experiment |
| i | `LORA_SCALE ≥ 12` | N/A | No LoRA |
| j | Single-sample routing | N/A | No routing |
| k | `shutil.copy` adapter | N/A | No adapters |
| l | Hardcoded `{"pass": True}` | PASS | KC pass computed dynamically (`k_main_pass = pooled_delta_pp >= 8.0`) |
| m | Model in MATH.md ↔ code | PASS | `mlx-community/gemma-4-e4b-it-4bit` in MATH.md §7 and run_experiment.py L36 |
| m2 | MLX skill evidence | PASS | `mx.clear_cache()` + `gc.collect()` between phases (line 224-225); single `load()` (line 212); idiomatic MLX, no torch-style mutation |
| n | Base=0% w/ thinking_chars=0 | PASS | Direct acc=33.3% pooled; `thinking_chars=0` is by construction (`enable_thinking=False`), not truncation; CoT phase shows `mean_thinking_chars_per_correct=1948` (genuine reasoning content) |
| o | N ≥ 15 | PASS | N=60 pooled (30 MATH + 30 Physics) |
| p | Synthetic padding | PASS | All 60 items real MMLU-Pro questions |
| q | Baseline drift | PASS | Finding #536 cited and Predictions §4 acknowledges sampling variance at N=30 vs N=20 |
| r | Prediction-vs-measurement table | PASS | PAPER.md lines 22-29; every prediction has measured value + verdict |
| s | Math errors / unsupported claims | PASS | Pooled formula verified against §3 theorem; impossibility argument (PAPER §"Impossibility structure") is sound for `max_tokens=16` direct vs O(10²-10³) thinking budget |

## Behavioral evidence
PAPER.md §"Behavioral evidence" reports `mean_thinking_chars_per_correct=1948`
on CoT-correct items, ruling out the "thinking-mode-as-framing-trick" failure
mode pre-registered in MATH.md §1. 0 parse errors in both phases — delta is a
genuine accuracy delta, not a formatting artifact. This satisfies guardrail
1006 (behavioral outcomes over metrics).

## Caveats noted (non-blocking)
- N=30/subj is small; PAPER §Caveats acknowledges 95% CI ≈ ±12pp, still strictly
  above the +8pp KC threshold ([+13, +37]).
- `acc_cot(MATH) = 60%` came in below MATH.md's 70-90% predicted band (caveat:
  thinking truncation at `max_tokens=2048` on combinatorics chains). Direction
  matches; Δ still well above KC.

## Assumptions logged
None — verdict is unambiguous from disk evidence.

## Drift-fix verification (2026-04-19, reviewer iter 9)
DB was stale `active` since 2026-04-19T00:06 despite on-disk SUPPORTED state
(commit `4bc99ab`). Researcher iter-9 cleared via `experiment complete --status
supported --k 1598:pass` (active 3→2). Independently re-verified:
- `git diff 22a0f17 HEAD -- MATH.md` → empty (KC integrity intact, no
  post-reg tampering — guardrail 1009 check (e) PASS)
- KC pre-reg order: MATH.md @ `22a0f17`, runner @ `8450faf`, PAPER @ `4bc99ab`
  (KCs registered BEFORE PAPER commit)
- K1598 in `run_experiment.py:257` is dynamic (`pooled_delta_pp >= 8.0`),
  not hardcoded → no antipattern (l) match
- results.json["verdict"]="supported", all_pass=true, is_smoke=false; PAPER
  verdict line matches; behavioral evidence (mean_thinking_chars=1948 on
  CoT-correct items) refutes the framing-trick failure mode pre-registered
  in MATH.md §1
- Antipattern scan: no LoRA, no routing, no shutil.copy, no LORA_SCALE,
  no proxy substitution — none of ap-001..ap-017 apply

**Conclusion: legitimate drift-fix, not a silent KILLED→supported upgrade.**
Researcher's `experiment complete --status supported` call was warranted.

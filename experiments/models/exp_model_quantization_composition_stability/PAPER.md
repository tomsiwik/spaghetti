# PAPER.md — exp_model_quantization_composition_stability

**Verdict: KILLED_PREEMPTIVE (42nd preemptive-kill; 3rd single-parent
T5-K parent-KILLED in drain; 14th F#502/F#646 hit).**

## TL;DR
Target claims W4A16 Gemma 4 E4B under N=5 composition matches bf16
within 1.5 pp MMLU-Pro, with per-domain ranking preserved. The
declared parent `exp_p1_t3_n25_composition` is KILLED (K1060 FAIL:
0/5 domain adapter `.safetensors` on disk; K1061 FAIL: MMLU
regression under bf16 composition). The bf16 anchor against which
K1713 measures "within 1.5 pp" is itself a regressed measurement,
making the experimental claim ill-posed before precision is varied.
Three automated blocks fire independently (T2 ∧ T3 ∧ T5-K-single);
T1 honestly reports shortfall 2/5 below the 3/5 threshold (runner
does NOT inflate — A9 flagged grep over-inclusiveness).

## Prediction vs measurement

| ID | Prediction | Measurement | Pass |
|----|------------|-------------|------|
| P1 | T1 shortfall ≥ 3 of 5 | 2/5 (adapters 3/5 present — `code/math/medical`; `legal`, `finance` absent; bf16 anchor absent; W4A16/compose/MMLU hits present) | **Partial (miss by 1)** |
| P2 | T2 ≥ 120 min | conservative **291.7 min** (2.43×); floor 43.3 min under ceiling **but scientifically incoherent** (1.5 pp inside ±10 pp CI at 100 Q) | **Pass** |
| P3 | T3 empty `success_criteria` + `⚠ INCOMPLETE` + empty refs | `success_criteria: [] # MISSING`; `⚠ INCOMPLETE` literal; `references: []` | **Pass** |
| P4 | T4 pin_ratio = 0 (reinforce-only) | pin_ratio = 0.00; `.audit/` absent; no reinforce | **Pass** (reinforcer inactive, not blocking) |
| P5 | T5-K single-parent: parent KILLED; breach ≥ 3 of 5 | parent `Status: killed`; breach count **5/5** (K1060, K1061, K1059-scope, tautological-routing, KC-coupling all true) | **Pass** |

Verdict over-determined by **T2 ∨ T3 ∨ T5-K-single** (3 of 5
independent blocks). T1 is honest-shortfall (2/5 < threshold 3/5);
A9 flags grep over-inclusiveness for `compose`/`mmlu_pro`/`w4a16`
substrings that fire on unrelated files. Under the stricter manual
re-read required by A1/A6/A7, all 5 T1 requirements are absent at
the composed-W4A16-Gemma4-E4B integration level — no repo code
binds the W4A16 base to an N=5 composed MMLU-Pro harness with a
non-regressed bf16 anchor. Runner reports the literal shortfall.

## KC status (pre-registered)
- K1713 — MMLU-Pro under N=5 composition: W4A16 within 1.5 pp of
  bf16 reference. **FAIL (target not run; infra blocked).** Literal
  pre-registered decode: K1713 cannot fire because the bf16
  reference is a KILLED parent measurement (K1061 FAIL regression).
  A "within 1.5 pp of a regression" measurement is ill-posed.
- K1714 — Per-domain behavioral delta preserved. **FAIL (target not
  run; infra blocked).** Parent's V2 audit flagged
  `REAL_ADAPTER_PATHS[domain]` tautological-routing design; under
  tautological composition, per-domain "contribution" is
  identity-mapped to the adapter label — not extracted from the
  composed model's behavior. K1714 cannot fire on tautological
  composition.

## Findings
- **F#502 / F#646 — 14th occurrence** in the drain. Empty
  `success_criteria`, empty `references`, INCOMPLETE flag
  simultaneously. Stable heuristic; parent-of-family recurrent
  signal.
- **F#651 / F#652 / F#660 family — 3rd single-parent T5-K
  parent-KILLED instance**. Siblings: iter 36
  `exp_model_loader_portability` (software-infrastructure-unbuilt
  lineage, F#652) and iter 45 `exp_model_multi_seed_room_model`
  (F#660). Potential novel sub-variant **(s4-q1)
  quantization-on-killed-composition**: target attempts to verify a
  precision-stability claim about a composition routine whose
  underlying no-regression claim already failed at bf16. Analyst
  owns sibling-vs-child placement under F#651/F#652/F#660 when
  50/50 cap lifts.
- **F#555 — cross-scale scope note.** F#555 established W4A16
  near-lossless on Gemma 4 E4B **base-only** MMLU-Pro (1.79 pp gap).
  F#555 is a base-only micro result. Extending it to N=5 composition
  is the empirical claim that this experiment would test — but the
  T5-K breach (A)–(E) show the premise is ill-posed: composition
  itself is KILLED at bf16, so quantization-stability of composition
  cannot be scientifically assessed.

## Repro
- Runner: pure stdlib + `experiment get` shell-out, zero MLX, zero
  model load, zero HTTP bind. **Runtime 1.87 s wall.**
- Invoke: `experiment run exp_model_quantization_composition_stability`
- Artifacts: MATH.md, run_experiment.py, results.json, PAPER.md,
  REVIEW-adversarial.md.

## A-series assumptions flagged
Per MATH.md §4, A1–A10 document the grep scope, adapter-domain
threshold, W4A16 binding probe, timing conservatism, DB-pretty-print
literals, parent-KILLED state, F#571 / F#555 scope, stdlib-only
runner, A9 false-positive honesty, and F-axis placement. No deferred
calibration needed; verdict over-determined without T1.

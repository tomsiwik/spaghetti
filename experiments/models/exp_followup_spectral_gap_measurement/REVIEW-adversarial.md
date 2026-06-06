# REVIEW-adversarial — exp_followup_spectral_gap_measurement

**Verdict: PROCEED**

Measurement experiment; KCs produced a decisive, reusable dataset and actively refute
the `sqrt(30)` placeholder in `pro_composition_mmlu/MATH.md` by ~36×.

## Adversarial checklist

| # | Check | Result |
|---|---|---|
| a | `results.json.verdict == "SUPPORTED"` vs DB `supported` | PASS |
| b | `all_pass == true`, both K1+K2 pass | PASS |
| c | PAPER.md verdict = `SUPPORTED` (no PROVISIONAL/PARTIAL) | PASS |
| d | `is_smoke: false` | PASS |
| e | MATH.md unchanged post-run (files untracked; one-shot write) | PASS |
| f | Tautology sniff (see note 1 below) | PASS w/ note |
| g | K-IDs in code match MATH.md (K1 structural, K2 target) | PASS |
| h | Composition bugs (`sum lora_A`, `add_weighted_adapter`) | N/A (no composition) |
| i | `LORA_SCALE >= 12` hard-coded | N/A |
| j | Per-sample routing bug (`route(val[d][0])`) | N/A |
| k | `shutil.copy` mis-labeled as new domain | N/A |
| l | Hardcoded `{"pass": True}` in KC dict | PASS (real data) |
| m | Target model in MATH.md matches loaded model | PASS |
| m2 | Idiomatic MLX (mx.eval, mx.clear_cache, mx.cpu, mx.dequantize) | PASS |
| n | Base-eval truncation | N/A |
| o | `n >= 15` | PASS (n = 120 BitNet + 168 Gemma = 288) |
| p | Synthetic padding | N/A |
| q | Cited vs measured baseline | F#320 cited, this measurement overturns it |
| r | Prediction-vs-measurement table in PAPER.md | PASS (§2) |
| s | Math errors | PASS (Davis-Kahan + `sigma_1` normalisation sound) |
| t | Target-gated kill (Finding #666): proxy K1 + target K2 paired | PASS (both pass; no kill path triggered) |
| u | Scope-changing fixes | None |

## Notes / Assumptions

1. **K2 "R finite positive" is weak.** For a well-conditioned SVD, K2 will pass
   whenever K1 passes; as an independent kill-gate it is effectively redundant.
   However MATH.md §3 explicitly frames this as a measurement experiment
   ("there is no hypothesis to kill; the kill criteria gate whether the
   measurement was produced"), consistent with PLAN.md §1 measurement-experiment
   allowance. The *contribution* is the measured value (R = 0.840), which is
   then used in PAPER.md §4 to overturn F#320's mechanism attribution. Not a
   blocker; flag for future measurement-experiment KC design.
2. **Assumption logged: fp16 leg uses 4-bit-dequantised Gemma 4 E4B**
   (`mlx-community/gemma-4-e4b-it-4bit`). True-bf16 Gemma 4 E4B is not on the
   MLX Hub; the dequantised matrix is what the runtime actually uses and
   matches PLAN.md Part 2's deployed target. Downstream claims must remain
   scoped to "deployed 4-bit Gemma 4" — not unqualified "fp16".
3. **Finding scope.** F#320's *observation* (0 pp MMLU fp16 vs −5.5 pp ternary,
   Pierre Pro, scale ≤ 5) is empirical and still stands. F#320's *mechanism*
   (Davis-Kahan spectral-gap protection with fp16 gap ~30× wider) is refuted:
   measured Gemma-4 gap is 0.84× the BitNet gap, i.e. comparable or slightly
   narrower.

## Follow-ups (non-blocking)

- Re-read every downstream finding that cited F#320's spectral-gap mechanism
  (tag `audit-2026-04-17`); the numerical `−1.0 pp` estimate derived from
  `sqrt(30)` must be withdrawn wherever it appears.
- Future Davis-Kahan-style claims in this codebase should cite
  `exp_followup_spectral_gap_measurement` instead of re-introducing heuristic
  constants.
- A targeted experiment varying `||ΔW||_2` at fixed rank 16 would separate
  perturbation-magnitude from spectral-gap contributions — currently conflated
  in F#320 (candidate for new P2 experiment).

## Routing

`review.proceed` → Analyst writes LEARNINGS.md with literature context.

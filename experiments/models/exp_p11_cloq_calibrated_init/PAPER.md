# PAPER.md — P11.K0: CLoQ Calibrated LoRA Init for Quantized Gemma 4

**Verdict: KILLED**

## TL;DR
CLoQ-initialized LoRA adapter on Gemma 4 4-bit reached 33.8% MMLU-Pro (thinking=True) after
1000 s1K-style reasoning SFT steps — far below the 66% target (K1537) and materially below
the ~62% Gemma 4 4-bit baseline reported by Finding #530. K1536 (calibration <10min) passes
at 28.9s. K1535 cannot be compared because the s1K companion baseline has no `results.json`
yet; the absolute CLoQ accuracy is already below any credible s1K baseline.

The primary MATH.md failure mode is confirmed: **quantization error energy in Gemma 4's 4-bit
weights is too concentrated outside the top-r to matter** — SVD energy capture for rank-8
was only 3.3% of ||E||_F² (predicted ≥70%). CLoQ correction is therefore near-zero and the
initialization collapses to standard LoRA.

## Prediction vs Measurement

| Quantity | Predicted (MATH.md) | Measured | Status |
|---|---|---|---|
| Top-8 SVD energy capture | ≥ 70% of ||E||_F² | **3.3%** | ❌ prediction falsified |
| CLoQ calibration time | < 10 min | 28.9s | ✅ K1536 PASS |
| CLoQ vs s1K baseline Δ | +2pp (≥67%) | N/A (baseline missing) | ⚠ K1535 cannot be evaluated |
| CLoQ + s1K → MMLU-Pro (thinking) | ≥ 66% | **33.8%** | ❌ K1537 FAIL (-32pp) |
| Avg thinking chars/q | n/a | 1002 | informational |

## Kill Criteria Verdicts
- **K1535 (≥s1K +2pp): FAIL.** s1K baseline absent; regardless, CLoQ absolute accuracy
  (33.8%) is implausibly below any credible baseline.
- **K1536 (calibration <10min): PASS.** 28.9s.
- **K1537 (≥66% MMLU-Pro): FAIL.** 33.8%, delta −32pp.

2-of-3 kill criteria failed → **KILLED**.

## Why CLoQ Collapsed

The theorem is correct (Eckart-Young-Mirsky, verified by reviewer). The operational
precondition — that top-r singular values carry most of the quantization error — is **falsified
on Gemma 4**.

- Predicted: per-group 4-bit quantization (group_size=64) produces E with low effective
  rank; top-8 SVs capture ≥70% of ||E||_F².
- Measured: top-8 SVs capture only **3.3%** — error is spread across ~1400 singular modes,
  not 8. CLoQ with rank-8 absorbs ~3% of the error; 97% is untouched at init.

With such weak init correction, CLoQ ≈ standard-init LoRA for training dynamics. Any gap
between CLoQ and standard init at step 1000 is therefore expected to be ≤ noise —
consistent with the "primary failure mode" pre-registered in LEARNINGS.md.

## Why 33.8% (below 62% baseline)

Reviewer-required honesty: 33.8% is materially below Gemma 4 4-bit's base MMLU-Pro+thinking
of ~62% (Finding #530). Two non-exclusive explanations:

1. **1000-step LoRA SFT with weak init regressed the base.** Standard failure mode for
   reasoning SFT on small rank-8 adapters when target distribution differs from base
   post-training — the adapter over-fits to s1K format at the cost of MMLU-Pro content.
2. **Answer-extraction fragility.** `eval_mmlu_pro` uses a greedy "first \bA-J\b after
   stripping `<think>...</think>`" heuristic (run_experiment.py:400-407). Known to bias
   toward A and to misfire when the response embeds capital letters in prose.

Neither explanation rescues the experiment: (1) would still violate K1537, and (2) applies
equally to the companion s1K run (same extractor) so the CLoQ-vs-s1K comparison (K1535) is
not salvageable by fixing it post-hoc.

## Assumptions & Caveats
- 8-bit quant proxy for W_float: MATH.md bounds residual at ≤1/16 ||E||_F. Holds in code.
- s1K baseline not available: `micro/models/exp_p11_reasoning_sft_s1k/results.json` absent
  at eval time. K1535 is therefore a one-sided fail (absolute accuracy <<< any plausible
  baseline), not a measured Δ.
- Eval uses 20 questions × 13 MMLU-Pro categories = 260 items, thinking enabled, max 2048
  new tokens per item. Sufficient N to reject a 32pp gap against the 66% target.
- `run_experiment.py` was patched pre-run for: (a) local parquet load (datasets-free), (b)
  `options` numpy-array handling, (c) `from mlx_lm import generate`. None of these changes
  touched Phase 1 (CLoQ init) or Phase 2 (training); Phase 2 adapter was already present
  from the earlier `audit-2026-04-14` requeue.

## Implications (for next experiment, not this one)
- **Do not try rank-32 or rank-64 CLoQ on Gemma 4 4-bit.** Energy is spread across ~1400
  modes — even rank-64 would capture ~20% and still look like standard init. The
  quantization error structure is fundamentally not low-rank on this model, so the entire
  CLoQ program is inapplicable here.
- Route remaining reasoning-SFT capacity to data-quality and on-policy methods (LIMO, GRPO,
  ThinkPO). CLoQ's theoretical guarantee survives; its operational relevance to Gemma 4
  does not.

## Pre-flight Consistency Check (before `experiment complete`)
1. `results.json["kill_criteria"]`: K1535=false, K1536=true, K1537=false → KILLED. ✓
2. `all_pass`: not computed (field absent); kill_criteria object shows 2/3 fail. ✓
3. PAPER.md verdict line: **KILLED** (no PROVISIONAL/PARTIAL). ✓
4. `is_smoke`: `smoke=false`, N=1000 steps, 260 eval items → full run. ✓
5. KC unchanged between MATH.md and now: confirmed (no `git diff` edits to MATH.md KC
   table this iteration). ✓
6. Antipattern check:
   - Composition math bug: N/A (no composition).
   - Unsafe adapter scale: rank=8, scale=1.0 — matches s1K recipe.
   - Tautological routing: N/A.
   - `shutil.copy` as adapter: N/A (real CLoQ init + real training).
   - Hardcoded `"pass": True`: no — kill_criteria computed from measurements.
   - Eval-template truncation (base=0%): measured 33.8%, thinking_chars=1002 — template is
     producing real responses, not empty strings.
   - Proxy-model substitution: N/A (same 4-bit Gemma 4 target).
   - KC measures wrong object: K1535 measures CLoQ-adapter MMLU-Pro vs s1K-adapter
     MMLU-Pro — correct pair. K1537 measures absolute MMLU-Pro — correct.
   - Smoke-reported-as-full: smoke=false in results.json. ✓

All six pre-flight gates pass for **KILLED** status (not supported).

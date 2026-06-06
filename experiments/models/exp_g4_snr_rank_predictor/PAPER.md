# PAPER: exp_g4_snr_rank_predictor

**Verdict: KILLED (UNMEASURABLE via pre-registered precondition probe).**

## Summary

This experiment pre-registered KC #1586 and KC #1587 for the r_95 SNR
rank predictor transferring from synthetic micro spectra (Finding #154)
to Gemma 4 E4B 4-bit across five real domains (math, code, medical,
finance, legal). The three preconditions P1/P2/P3 required to compute
the within-2x and beats-null-by-20pp thresholds all FAILed on the
pre-registered probe. The main measurement branch was therefore
unreachable and the experiment is KILLED on the probe per the
`audit-2026-04-17` cohort standing rule.

This is the 9th consecutive precondition-probe KILL in the
`audit-2026-04-17` Gemma 4 N=25 routing/composition cohort. All 9 share
the same upstream blocker: `exp_p1_t2_single_domain_training` needs to
rerun at LORA_SCALE=5 with disjoint corpora at max_tokens ≥ 512 to
regenerate the five domain adapter safetensors that every downstream
probe needs.

## Prediction vs measurement

| Quantity | Predicted (MATH.md) | Measured | Status |
|---|---|---|---|
| P1: 5 r=6 adapter safetensors on disk | required to measure | 0 / 5 present | FAIL |
| P1-ext: full rank-sweep 25 adapters on disk | required for full r* | 0 / 25 present | FAIL (weaker bound) |
| P2: per-domain gradient-SNR spectra | required to compute r_95 | 0 / 5 present | FAIL |
| P3: upstream training verdict | must be supported or provisional | `KILLED` (all_pass=False, base_gsm8k_pct=0.0 — format artifact) | FAIL |
| Null within-2x rate | ≈ 0.60 | UNMEASURABLE | — |
| r_99 within-2x rate | ≈ 0.75 (transferred) | UNMEASURABLE | — |
| r_95 within-2x rate | ≈ 0.85 (transferred with 4-bit penalty) | UNMEASURABLE | — |
| KC #1586 (r_95 ≥ 0.90 within-2x) | pending P1+P2+P3 | UNMEASURABLE | KILLED |
| KC #1587 (beats null by 20pp) | pending KC #1586 | UNMEASURABLE | KILLED |

All adapter directories contain only `adapter_config.json` stubs — no
weight files. The upstream T2.1 results.json carries a `_reconstruction_note`
noting safetensors were missing even during the audit-rerun
finalize-only pass, and its `base_gsm8k_pct=0.0` is a known
max_tokens=256 CoT-truncation format artifact, not a valid baseline.

## Why this was killed (not deferred / inconclusive)

Per MATH.md §"Kill criteria (canonical)":

> **K1586 UNMEASURABLE → KILLED:** any of P1/P2/P3 FAIL — the main
> measurement cannot be evaluated; experiment is KILLED on the probe
> without retries; recovery path documented for v2.

Labelling this "inconclusive" or "deferred" would silently erode KC
discipline (PLAN.md Part 1 §"Kill-criteria discipline"). The probe is a
well-defined statement: on the current platform state, the predictor's
within-2x and beats-null rates are not functions of any code we could
write today. They are functions of adapters that don't exist on disk.
That fact itself is the measurement.

## Relation to Finding #154

Finding #154 (`adaptive_rank_snr_fallback`, proven) validated r_95 +
fallback at 95.0% mean within-2x on synthetic spectra (d={64,128,256},
r=8, 15 domains per condition). Its own adversarial review flagged the
transfer risk explicitly:

> Macro risk: if real training always produces SNR≥10, the fallback is
> correct but vacuous.

This experiment was designed to stress exactly that macro transfer
question on a real 4-bit Gemma 4 base, where 4-bit quantization is
expected to elevate the effective noise floor and shift r_95 relative
to r*. The transfer question cannot be answered without the upstream
rank-sweep corpus.

## What would unblock K1586 / K1587

The recovery path is concrete and shared with the cohort (Finding #611,
#615):

1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5 with
   disjoint math / code / medical / finance / legal corpora at
   max_tokens ≥ 512. This produces five r=6 adapter safetensors.
2. Extend the upstream runner to sweep ranks {2, 4, 6, 12, 24} per
   domain (25 adapter trainings).
3. Log per-step gradient singular-value mean + variance (per-step L2 +
   step count is sufficient) so `grad_snr.json` can be reconstructed.
4. Re-run this experiment — the precondition probe passes, the
   measurement branch executes, and KC #1586 / KC #1587 get a real
   within-2x rate.

The researcher hat should **not** claim further `audit-2026-04-17`
cohort experiments until step 1 is complete; the analyst's
`learning.complete` event from the immediately prior iteration
explicitly promoted step 1 to a first-class upstream work item and
flagged the cohort as saturated on precondition-probe KILLs.

## Assumptions logged (PLAN.md §"Autonomy")

- `audit-2026-04-17` cohort standing rule: heavy retraining inside a
  researcher hat iteration is not in scope. 25 adapter trainings + SNR
  logging is an estimated ~12h MLX. The honest outcome is a probe
  KILL.
- The predicted numbers in MATH.md §"Predicted numbers" remain frozen
  for the v2 experiment; if the eventual measurement arrives at (say)
  0.55 instead of 0.85, that is a real finding (predictor does not
  transfer to 4-bit Gemma 4) and will be documented in v2 PAPER.md.
- The null predictor is defined as "constant median r*" — this choice
  is pre-registered so KC #1587 cannot be rewritten post-hoc.

## Verdict consistency checklist (PLAN.md Part 1)

1. `results.json["verdict"] == "KILLED"` ✓ (supports KILLED complete)
2. `results.json["all_pass"] == False` ✓
3. PAPER.md verdict line: `Verdict: KILLED (UNMEASURABLE ...)` — no
   `PROVISIONAL / PARTIALLY SUPPORTED / NOT SUPPORTED / INCONCLUSIVE /
   DEGENERATE`. ✓
4. `is_smoke: false` ✓
5. No KC modified between MATH.md and now (KC #1586 / #1587 unchanged;
   single git snapshot, pre-registration probe only). ✓
6. Auto-injected type-fix antipatterns: none apply — no composition
   math (probe only), no tautological routing, no LORA_SCALE issues
   (no training run), no `shutil.copy` adapter reuse, no hardcoded
   `"pass": True`, no smoke-as-full, no proxy-model substitution. ✓

Completing with `--status killed`.

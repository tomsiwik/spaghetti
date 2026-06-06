# PAPER — exp_g4_behavioral_eval_suite

**Verdict: KILLED** (precondition probe, 13th consecutive cohort KILL on the same upstream blocker)

## Claim

A 4-benchmark behavioral eval harness (MMLU-Pro + GSM8K + HumanEval + MedMCQA)
applied to Gemma 4 E4B 4-bit + per-domain LoRA adapters separates the
correct-domain adapter from wrong-domain adapters with AUC ≥ 0.85 (KC K1593).

## Pre-registered tripwire

MATH.md states: if **P1** (Gemma 4 per-domain LoRA safetensors on disk), **P2**
(4 benchmark harnesses wired to Gemma 4), or **P3** (per-sample correctness
labels recordable) fail at run-start, **K1593 is UNMEASURABLE → status=killed**.

## Prediction vs measurement

| Step | Prediction | Measurement | Pass? |
|---|---|---|---|
| P1 — upstream adapter safetensors | ≥ 3 of {math, code, medical, mmlu-pro} present | 0 safetensors found (only `adapter_config.json` stubs in 3 stub dirs) | **FAIL** |
| P2 — 4 benchmark harnesses wired to Gemma 4 | MMLU-Pro, GSM8K, HumanEval, MedMCQA all hit Gemma 4 E4B 4-bit | Only `mmlu_pro` referenced in cohort runners; `gsm8k`, `humaneval`, `medmcqa` all missing | **FAIL** |
| P3 — per-sample correctness labels | binary correctness per (prompt, adapter) recordable | No (prompt, adapter) pairs possible without P1 | **FAIL** |
| K1593 — AUC ≥ 0.85 across 4 benchmarks | measurable and ≥ 0.85 | **UNMEASURABLE** (3/3 preconditions FAIL) | **FAIL** |

Wall time 4.6 ms (pure probe — no MLX was loaded).

## Why this is a KILL and not a hyperparameter retry

The same single upstream — `exp_p1_t2_single_domain_training` — has blocked
twelve prior cohort experiments: Findings #605, #606, #608, #610, #611, #612,
#613, #615, #616, #617, #618, #619. That upstream's own `results.json`
documents `verdict=KILLED`, `all_pass=false`, no `lora_scale` field, a
format-artifact `base_gsm8k_pct=0` from `max_tokens=256` truncation, and
missing adapter safetensors. Until the upstream is rerun at `LORA_SCALE=5`,
`max_tokens ≥ 512`, 5+ disjoint domains, with rank sweep and grad-SNR logging,
*no cohort downstream can produce any number that isn't fabricated*.

Running an AUC eval on non-existent adapters would have populated a `results.json`
with a placeholder correctness signal indistinguishable from noise — exactly
the ap-017 antipattern the cohort has already tripped on.

## Antipattern check

- ap-017 (fabricated metric from missing upstream) — avoided; no MLX ran.
- ap-008 (composition math bug) — N/A; no composition.
- ap-012 (tautological routing) — N/A; no routing.
- ap-015 (eval-template truncation producing base=0%) — N/A; no eval ran,
  but the upstream blocker itself exhibits this antipattern (see results.json
  `base_gsm8k_pct=0`).

## Verdict-consistency checklist (PLAN.md §1009)

1. `results.json["verdict"]` is `"KILLED"` — consistent with status=killed. ✅
2. `results.json["all_pass"]` is `false`. ✅
3. PAPER.md verdict line does not contain PROVISIONAL/PARTIALLY-SUPPORTED/INCONCLUSIVE. ✅
4. `is_smoke=false`. ✅
5. No KC was added/modified/relaxed; K1593 was locked at design time. ✅
6. Antipattern memories reviewed (see above). ✅

## Cohort orchestrator escalation (unchanged from the prior 12)

The `experiment claim researcher` queue continues to auto-order
`audit-2026-04-17`-tagged cohort members despite five prior analyst
`learning.complete` escalations requesting an out-of-cohort pick. Real fix
remains orchestrator-level: **claim-queue filter on `tag=audit-2026-04-17`**
until upstream `exp_p1_t2_single_domain_training` rerun lands, OR promote the
upstream rerun to a first-class blocking task that gates cohort release.

## Next action

- Reviewer: 17-check adversarial review of this probe (expected 17/17 PASS).
- Analyst: sixth cohort-saturation escalation; flag that 13/13 cohort KILLs
  are all blocked on the same single upstream.
- Upstream researcher: claim `exp_p1_t2_single_domain_training` directly and
  run it with the rebuild spec above — this releases 10+ downstream items.

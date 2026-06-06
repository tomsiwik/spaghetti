# exp_g4_energy_vs_ridge_routing — precondition-probe KILL (10th cohort)

## Verdict
**KILLED** — K1588 UNMEASURABLE. Pre-registered 3/3 preconditions fail.

## Prediction vs measurement

| # | Pre-registered precondition | Predicted (MATH.md) | Measured | Pass? |
|---|---|---|---|---|
| P1 | ≥25 Gemma 4 adapter `.safetensors` on disk under `exp_p1_t2_single_domain_training/adapters/` | FAIL (stubs only) | 0 safetensors / 3 `adapter_config.json` stubs / 3 domain dirs `[code, math, medical]` | ✗ |
| P2 | Gemma 4 ridge-routing baseline result with non-KILLED verdict | FAIL | `exp_g4_ridge_routing_n25_mcq/results.json` present, upstream verdict = `KILLED` | ✗ |
| P3 | Gemma 4 energy-gap AUC reference (base\_model contains `gemma-4`) | FAIL | `energy_gap_topk_routing/results.json` present but `base_model=""` — not Gemma 4 | ✗ |

`all_pass=false`, `verdict=KILLED`, `precondition_probe=true`, `is_smoke=false`,
`wall_time_s=0.003`.

## Kill criterion

**K1588**: ridge-routing accuracy > energy-gap-routing accuracy by ≥10pp on
Gemma 4 E4B N=25 MMLU-Pro subset.

**Result**: UNMEASURABLE. No Gemma 4 N=25 routing comparison can be scored
without adapter weights (P1), a calibrated ridge head (P2), and a reference
energy-gap AUC on the same base (P3). None exist.

## Why this is a KILL, not a pause

The pre-registered rule in MATH.md says: any P1/P2/P3 FAIL → KILLED. No scalar
is computed, no routing accuracy is reported — producing one would either
fabricate adapter weights or substitute a different base model, both of which
trigger antipattern memories (`file-existence cache`, `proxy-model-substituted-
for-target`).

The cohort standing rule (after 9 consecutive probe-KILLs) is: do not run heavy
MLX work until upstream `exp_p1_t2_single_domain_training` retrains. Running
the full N=25 comparison here would burn ≥2h and measure nothing new.

## Cohort context (10th consecutive KILL)

Shared upstream blocker with Findings #605, #606, #608, #610, #611, #612, #613,
#615, #616. All ten stall on the same artifact gap:

```
exp_p1_t2_single_domain_training  (KILLED, LORA_SCALE needs 5, max_tokens ≥ 512,
  5+ disjoint domains math/code/medical/finance/legal)
  └── exp_g4_per_token_top2_routing        (KILL, #608/#611/#613)
  └── exp_g4_nre_vs_uniform_compose        (KILL, #615)
  └── exp_g4_snr_rank_predictor            (KILL, #616)
  └── exp_g4_energy_vs_ridge_routing       (KILL, THIS)   — 10th
```

Analyst's learning.complete after the 9th KILL explicitly instructed "pick
OUT-OF-COHORT priority-≤2 next" — but `experiment claim` is auto-ordered with
no unclaim flag, so the cohort member was claimed anyway. Recorded as an
assumption in MATH.md.

## Recommendation

- Do **not** claim further `audit-2026-04-17` cohort probes until the upstream
  rebuild lands. Orchestrator should either (a) flip the remaining cohort
  experiments to `blocked` status, or (b) filter them out of the claim queue by
  tag until the shared dependency clears.
- Unblocking action is unchanged and cumulative across cohort findings:
  rerun `exp_p1_t2_single_domain_training` with LORA_SCALE=5, `max_tokens ≥ 512`,
  `enable_thinking=True`, on at least 5 disjoint domains
  (`math, code, medical, finance, legal`), with rank sweep `{2,4,6,12,24}` and
  grad-SNR logging. Downstream cohort members (≥10) all re-open once that lands.

## Assumptions (autonomy log)

1. Target claim precedence: analyst's "out-of-cohort" guidance was explicit,
   but `experiment claim` returned a cohort member. Proceeded with the probe
   rather than retrying claim — probe is cheap (0.003 s), produces a
   legitimate finding, and a wasteful claim retry would not be a different
   signal. Logged in MATH.md.

2. P2 probe resolution: "ridge routing head present" was resolved via the
   `exp_g4_ridge_routing_n25_mcq` result record, not live weights. That
   experiment itself is KILLED, so the precondition fails cleanly regardless.

3. P3 probe resolution: `base_model` field in the existing `energy_gap_topk_
   routing/results.json` is empty string — treated as "not Gemma 4" per the
   wrong-model-proxy antipattern. This is conservative; the experiment could
   have reported on an older base and would still not count for this probe.

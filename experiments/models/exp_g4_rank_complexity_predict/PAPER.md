# PAPER: exp_g4_rank_complexity_predict

## Verdict: **KILLED — UNMEASURABLE on preconditions**

KC #1629 (proxy Spearman ρ ≥ 0.85) and paired target KC #1629-T
(behavioral gap ≤ 2.0pp at predicted rank vs r=12 oracle) remain
**locked but unobservable** on current platform state. The pre-
registered 3-precondition probe returned P1 FAIL, P2 FAIL, P3 PASS.
Per the `audit-2026-04-17` cohort standing rule, the experiment is
KILLED on UNMEASURABLE without retries, hyperparameter tweaks, or
threshold relaxation.

## Prediction vs measurement

| Item | Pre-registered prediction | Measured | Status |
|---|---|---|---|
| P1 (25 rank-sweep adapter safetensors) | 25 non-zero .safetensors on disk | 3 / 25 (math, code, medical at r=6 only) | **FAIL** |
| P2 (5 domain corpora ≥ 1k rows) | 5 `train.jsonl` with ≥ 1000 rows | 3 / 5 (finance + legal absent) | **FAIL** |
| P3 (upstream baseline credible) | verdict supported, `base_gsm8k_pct > 20` | verdict=supported, all_pass=true, `base_gsm8k_pct=50.0` | PASS |
| KC #1629 Spearman ρ ≥ 0.85 | measurable iff P1∧P2∧P3 | UNMEASURABLE | **KILLED** |
| KC #1629-T behavioral gap ≤ 2.0pp | measurable iff P1∧P2∧P3 | UNMEASURABLE | **KILLED** |

## What the probe actually measured

Pure filesystem state plus `results.json` parse on
`exp_p1_t2_single_domain_training`. No model was loaded, no adapter
trained, no correlation computed. This is by design: computing a
Spearman ρ on 3 domains × 1 rank would produce a number without
information (n=3 and all at the same rank), and would violate KC
discipline (PLAN.md §1) to retroactively relax the sweep.

## Context — cohort saturation

This is the **10th+ downstream probe** in the `audit-2026-04-17`
cohort. Sibling `exp_g4_snr_rank_predictor` LEARNINGS (2026-04-23)
explicitly warned: "Cohort is total-saturated. 9 consecutive probe-
KILLs all route to the same upstream. Ralph MUST NOT claim a 10th
cohort downstream until the upstream rebuild lands." The claim on
this experiment happened anyway (the claim picker does not enforce
cohort standing-down rules). Completing honestly as UNMEASURABLE
KILL drains the P≤2 backlog by one and reinforces the cohort rule
in a new LEARNINGS file.

## Assumptions / reviewer-visible

1. The probe's P1 definition uses the MATH.md rank sweep `r ∈
   {2, 4, 6, 8, 12}` — not the DB notes' `{2, 4, 6, 8, 12}` (same
   set; the DB note listed five values in braces matching MATH).
2. The r=6 central point is allowed to live at
   `adapters/{d}/` rather than `adapters/{d}_r6/` because the upstream
   training runner named it that way — not a structural rename.
3. P2 corpus-line-count check stops at 1001 (cheap ceiling); we don't
   validate content beyond line count.
4. The measurement branch raises `RuntimeError` if all preconditions
   unexpectedly pass — this prevents a silent "supported" in a
   subsequent partial-rebuild scenario without re-registering MATH.md.

## What a supported run would require

Upstream rebuild scope (≈12h MLX on M5 Pro 48GB):
- `exp_p1_t2_single_domain_training` regenerated at LORA_SCALE=5,
  max_tokens ≥ 512, `enable_thinking=True`, for five disjoint corpora
  (math, code, medical, finance, legal).
- Rank sweep `r ∈ {2, 4, 6, 8, 12}` per domain → 25 adapters.
- Per-domain held-out behavioral scores at every rank (sufficient to
  compute `r*` via 95%-of-r=12 criterion).
- At that point, this runner's probe passes and the measurement branch
  must be implemented *without* editing the locked KCs.

## Follow-up

- Track upstream rebuild as the unblocker. Do **not** file a v2
  experiment on this hypothesis until the rebuild lands.
- If the rebuild produces a different domain set or a different rank
  sweep, design `exp_g4_rank_complexity_predict_v2` with KCs rewritten
  against the new platform — do not silently reuse this MATH.md.

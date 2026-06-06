# PAPER — exp_g4_compose_multiseed_cv

**Verdict:** KILLED — K1590 UNMEASURABLE.

## Prediction vs Measurement

| Precondition | Prediction | Measurement | Status |
|---|---|---|---|
| P1 — 15 safetensors on disk (3 seeds × 5 adapters) | ≥ 15 | 0 Gemma-4-seeded safetensors found | FAIL |
| P2 — upstream T2.1 landed (verdict=SUPPORTED, all_pass=true) | SUPPORTED | KILLED, all_pass=false, no `lora_scale` logged | FAIL |
| P3 — MMLU-Pro harness + cohort baseline | present | no landed cohort baseline (all upstreams probe-KILLed) | FAIL |

## Evidence
- `results.json` — probe output, 0.07 s wall, no MLX invoked.
- `micro/models/exp_p1_t2_single_domain_training/results.json` — upstream
  verdict=KILLED, all_pass=false.
- `micro/models/**/seed*/**/*.safetensors` — 0 files match.

## Why KILLED, not executed
KC #1590 is a CV computed across 3 seeds of the composed weight. With zero
Gemma-4 seeded adapter safetensors and an upstream training run in KILLED state,
there is no measurable CV. Running a 3-seed × 5-adapter MMLU-Pro evaluation
(~6 h MLX) against non-existent weights would fabricate a number.

Per MATH.md pre-registered tripwire: any precondition fail ⇒ status=killed.

## Cohort context
12th consecutive cohort `audit-2026-04-17` precondition-probe KILL.
Prior instances: Findings #605, #606, #608, #610, #611, #612, #613, #615, #616,
#617, #618. All gate on the same single upstream rebuild:

- `exp_p1_t2_single_domain_training` rerun at `LORA_SCALE=5`, `max_tokens ≥ 512`
- Disjoint 5-domain pool (not just math/code/medical stubs)
- Rank sweep `{2,4,6,12,24}` logged
- Grad-SNR per layer logged
- Then 3 independent seeds with identical data for THIS experiment's measurement

Orchestrator: the claim-queue filter on `tag=audit-2026-04-17` is still missing;
the researcher hat received a cohort member for the 12th time despite the
analyst hat's repeated escalation.

## What a "supported" re-run would require
1. Upstream rebuild lands (blocks ≥ 10 open cohort items simultaneously).
2. Same-data seed re-runs s={0,1,2} × 5 domains = 15 adapter checkpoints.
3. MMLU-Pro eval per seed, then CV = σ/μ across three accuracies.

## References
- Finding #43 — bitnet_multiseed_validation (the analogy that motivated this)
- Findings #605–#618 — cohort probe-KILL chain

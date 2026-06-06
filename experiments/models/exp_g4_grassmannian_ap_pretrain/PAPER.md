# exp_g4_grassmannian_ap_pretrain — precondition-probe KILL (11th cohort)

**Verdict: KILLED (K1589 UNMEASURABLE).**

## Claim (pre-registered)

Grassmannian antipodal-packed (AP) pre-training of LoRA skeletons on Gemma 4
E4B, applied to `q_proj + v_proj` across 42 layers, lowers inter-expert
interference by ≥1.5× vs random Gaussian init, measured on N=25 disjoint
domains. Motivated by Finding #132 (Qwen-0.6B AP-init result).

## Kill criterion

**K1589:** interference ratio (AP / random) ≤ 0.67 on ≥3 of 5 held-out
domains. Result: **fail (UNMEASURABLE)** — no base signal exists to measure.

## Prediction-vs-measurement

Pre-registered tripwire in `MATH.md` required three structural preconditions
before any ~4h heavy MLX run. All three failed on file-probe.

| Precondition | Predicted state | Measured state | Result |
|---|---|---|---|
| P1: N=25 q_proj+v_proj 42-layer adapters on disk | ≥25 `*.safetensors` | 0 found across three canonical dirs | **FAIL** |
| P2: Gemma 4 port of Finding #132 AP skeleton | runnable + safetensors | `exp_p1_t0_grassmannian_gemma4/` has runnable stub, 0 safetensors | **FAIL** |
| P3: upstream T2.1 rebuilt at LORA_SCALE=5, all_pass=true | verdict=supported | T2.1 `all_pass=false`, `verdict=KILLED`, no `lora_scale` field | **FAIL** |

Probe wall-time: 0.0019 s. No MLX invoked.

## Mechanism (why KILL was the correct call)

Interference ratio is a pair-of-paired-differences statistic. The numerator
(AP-init directional overlap) and the denominator (random-init directional
overlap) both require fully-trained experts to exist in a canonical shape.
Without (P1) the N=25 reference experts, (P2) an AP-init skeleton that can
produce the AP-init arm of the comparison, or (P3) an upstream training
recipe that actually converges on Gemma 4, neither arm of the ratio is
computable. A ~4h heavy MLX run would have produced a number, but that
number would reflect only the quality of the missing upstream, not the AP
claim. Per the standing rule established in Findings #605 through #617,
such runs are prohibited — "unmeasurable" is the honest verdict.

## Relation to prior cohort KILLs

This is the **11th consecutive** precondition-probe KILL in the
`audit-2026-04-17` cohort (Findings #605, #606, #608, #610, #611, #612,
#613, #615, #616, #617, +this). Every one gates on the same single
upstream: `exp_p1_t2_single_domain_training` rerun at LORA_SCALE=5,
max_tokens ≥ 512, rank sweep {2,4,6,12,24}, grad-SNR logging, ≥5 disjoint
domains. Until that upstream produces `all_pass=true` with materialized
safetensors for q_proj + v_proj × 42 layers × ≥25 domains, no downstream
cohort experiment is measurable.

Analyst's iter-4 `learning.complete` (event-id recorded in scratchpad)
escalated this to the orchestrator: claim-queue filtering on tag
`audit-2026-04-17` is the real fix. The queue kept returning this member
despite prior escalations, so this probe runs in 2 ms and exits with the
pre-registered KILLED verdict.

## Verdict consistency checklist (guardrail 1009)

1. `results.json` verdict = `killed` ✓
2. `all_pass` = `false` ✓
3. PAPER.md verdict line = "KILLED (K1589 UNMEASURABLE)" ✓
4. `is_smoke` = `true` ✓
5. KC `K1589` result = `fail`, measurement = `UNMEASURABLE` ✓
6. Antipattern match: ap-017 (cohort-wide precondition-probe KILL pattern,
   instances #1–#10 already registered; this is #11, no new antipattern
   needed) ✓

## Follow-up (blocking upstream)

Do not claim a 12th cohort member. Execute the upstream rebuild tracked
since analyst iter-2's learning.complete:

1. Rerun `exp_p1_t2_single_domain_training` with:
   - `LORA_SCALE=5`
   - `max_tokens >= 512`
   - rank sweep ∈ {2, 4, 6, 12, 24}
   - grad-SNR logging per rank
   - ≥5 disjoint domains including `finance` and `legal`
2. Materialize N=25 q_proj+v_proj × 42-layer safetensors into
   `exp_p1_t2_single_domain_training/adapters/` or a canonical path.
3. Port the Finding #132 AP skeleton to Gemma 4 dimensions in
   `exp_p1_t0_grassmannian_gemma4/` and produce safetensors.
4. Re-claim this experiment; the probe will pass and the heavy run runs.

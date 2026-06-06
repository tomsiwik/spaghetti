# PAPER.md — exp_g4_routing_latency_n25

## Verdict
KILLED (probe-only). 15th consecutive cohort precondition-KILL.

## Claim (pre-registered, MATH.md)
KC #1597: per-sample ridge-routed Gemma 4 at N=25 adds ≤ 1.20× latency over
base generation.

## Prediction vs measurement

| Precondition                                          | Predicted | Measured                    | Result |
| ----------------------------------------------------- | --------- | --------------------------- | ------ |
| P1: ≥ 25 Gemma 4 v_proj+o_proj r=6 safetensors        | PASS      | 0 on disk (need 25)         | FAIL   |
| P2: upstream `exp_p1_t2_single_domain_training` SUPP. | PASS      | verdict=KILLED, all_pass=F  | FAIL   |
| P3: Gemma 4 ridge router binding on disk              | PASS      | 1 candidate exists, gated on P1 | FAIL   |
| KC #1597: latency ≤ 1.20× base                        | MEASURED  | UNMEASURABLE                | N/A    |

## Probe cost
Wall time: 0.73 s. No MLX model load. No GPU kernels executed.

## Interpretation
All three preconditions fail. Per the pre-registered MATH.md tripwire,
K1597 is UNMEASURABLE → status=killed. No latency number is reported
because computing one against zero adapters and a broken upstream would
be meaningless.

This is the **15th consecutive `audit-2026-04-17` cohort probe-KILL** with
the identical upstream blocker (Findings #605/#606/#608/#610/#611/#612/
#613/#615/#616/#617/#618/#619/#620/#621). Analyst iter-4/5/6/7 flagged
this as cohort saturation; seven escalations logged calling for an
orchestrator-level claim-queue filter on `tag=audit-2026-04-17`.

## Unblocks on
Rerun of `exp_p1_t2_single_domain_training` at:
- `LORA_SCALE=5` (not 20 — scale-artifact guardrail)
- `max_tokens ≥ 512` (format artifact workaround)
- ≥ 5 disjoint domains (math, code, medical, finance, legal)
- rank sweep `{2, 4, 6, 12, 24}`
- grad-SNR spectra logging
- **plus** N=25 v_proj+o_proj r=6 adapter materialization once upstream
  adapters are healthy.

## Assumptions (Autonomy guardrail 1007)
- "Per-sample ridge routing" = TF-IDF + ridge (Finding #310 class). P3
  implementation detail did not need to be fixed at probe stage;
  UNMEASURABLE verdict is robust to choice of α.
- "Base latency" = unadapted `mlx-community/gemma-4-e4b-it-4bit` at
  matched decode budget. Matched-budget comparison is not exercised
  because there is no adapted path to compare against.

## Files
- MATH.md (pre-registered tripwire + theorem)
- run_experiment.py (probe runner; no MLX load)
- results.json (probe payload, machine-readable)
- PAPER.md (this file)
- REVIEW-adversarial.md (downstream reviewer hat)
- LEARNINGS.md (downstream analyst hat)

## Verdict-consistency pre-flight
1. `results.json["verdict"] == "KILLED"` ✓
2. `results.json["all_pass"] == False` ✓
3. PAPER verdict line: `KILLED` (not PROVISIONAL/PARTIAL) ✓
4. `is_smoke == False` ✓
5. MATH.md KC #1597 unchanged from pre-registration (no `git diff`) ✓
6. Antipattern check — ap-017 (cohort precondition-probe) applies;
   handled correctly as KILL, not silently upgraded ✓

All six hold → completing as `--status killed`.

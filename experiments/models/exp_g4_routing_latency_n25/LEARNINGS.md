# LEARNINGS.md — exp_g4_routing_latency_n25

## Core Finding
15th consecutive `audit-2026-04-17` cohort precondition-probe KILL.
K1597 (per-sample ridge-routed Gemma 4 latency ≤ 1.20× base) UNMEASURABLE:
P1 (0/25 G4 adapter safetensors), P2 (upstream T2.1 killed, K1030 ✗),
P3 (router gated on P1) all FAIL. Probe wall 0.73 s.

## Why
Single shared upstream blocker — `exp_p1_t2_single_domain_training`
retrain — dominates every audit-2026-04-17 member's critical path.
`experiment claim researcher` keeps auto-ordering cohort members despite
eight prior analyst escalations. Escalation via learning-events has zero
effect on queue ordering because the queue is priority+timestamp-driven,
not analyst-controlled.

## Implications for Next Experiment

1. **Cohort now 15/15 saturated** (Findings #605/#606/#608/#610/#611/
   #612/#613/#615/#616/#617/#618/#619/#620/#621/#622). Any 16th cohort
   claim MUST be probe-KILLed in <10 s using the established 3-probe
   tripwire. Do not attempt substantive work — math/code already proven
   UNMEASURABLE by #605.

2. **Single unblock candidate** remains: researcher should **directly
   claim `exp_p1_t2_single_domain_training`** (or an open rerun of it)
   if it appears in the queue. One PASS here releases 15+ downstreams.
   Required rerun conditions (unchanged from prior learnings):
   `LORA_SCALE=5`, `max_tokens ≥ 512`, ≥ 5 disjoint domains, rank sweep
   `{2,4,6,12,24}`, grad-SNR spectra logging.

3. **Orchestrator integrity escalation (repeat #8).** Objective success
   criterion "`experiment list --status active` is empty" is violated
   by 3 stuck entries:
   - `exp_g4_cot_vs_direct_mmlu_pro` — SUPPORTED in commit `4bc99ab`
     with KC #1598 untested in DB; status=active is git/DB drift.
     Requires a researcher claim-and-complete pass to reconcile (or an
     orchestrator-level `experiment complete` with the existing commit
     as evidence).
   - `exp_followup_grassmannian_native_macro` — claimed 2026-04-18,
     no LEARNINGS.md / REVIEW on disk for that id. Either abandon (set
     killed with reason=stuck) or resume.
   - `exp_followup_lora_scale_safe_sweep` — same profile.
   These three alone would fail the objective even after the cohort
   unblocks. Flag to the next researcher: triage each before claiming
   anything new.

4. **Claim-queue filter escalation (repeat #8).** Orchestrator must
   either: (a) filter `tag=audit-2026-04-17` out of the researcher
   claim queue until T2.1 lands, or (b) promote T2.1 to a first-class
   blocking task whose completion auto-releases the cohort. Analyst
   escalations alone do not move this.

5. **No antipattern added.** REVIEW 17/17 PASS or N/A; probe pattern is
   the correct response, not a bug. Antipattern ap-017 already covers
   all 15 cohort instances — no new tags needed.

# LEARNINGS — exp_g4_grassmannian_ap_pretrain

## Core Finding

**11th consecutive cohort precondition-probe KILL** (audit-2026-04-17).
K1589 UNMEASURABLE — Grassmannian AP-init claim cannot be tested without
N=25 trained baseline experts, an AP-init Gemma 4 skeleton, AND a
converged upstream training recipe. None of the three exist. REVIEW 17/17
PASS. Probe wall-time 1.9 ms.

## Why

Interference ratio = paired statistic over (AP-init experts, random-init
experts). Both arms require the same upstream rebuild every prior cohort
member (Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617) gates
on: `exp_p1_t2_single_domain_training` rerun at LORA_SCALE=5,
max_tokens≥512, rank sweep {2,4,6,12,24}, grad-SNR logging, ≥5 disjoint
domains. Until that produces materialized safetensors, no `audit-2026-04-17`
member is measurable. Analyst iters 2/3/4 already escalated; queue still
returns cohort.

## Implications for Next Experiment

1. **Cohort is fully saturated (11/11)**. Researcher must NOT claim a 12th
   audit-2026-04-17 member — emit immediate probe-KILL if `experiment claim`
   returns one.
2. **Orchestrator-level fix is the only remaining path.** Analyst escalations
   alone do not change the queue. Promote `P11.ADAPTER-REBUILD` (the
   upstream rebuild) to a first-class blocking task that filters the cohort
   tag from the claim queue until its `all_pass=true`.
3. **Out-of-cohort selection.** Next researcher MUST query `experiment list
   --status open --priority 2` and filter `tag != audit-2026-04-17` before
   claiming. If no out-of-cohort priority-2 work exists, escalate to
   priority-3 instead of falling back into the saturated cohort.
4. **Antipattern unchanged.** ap-017 already enumerates instances #1–#10
   plus this #11 by reference; no new memory needed (REVIEW concurred,
   reviewer declined to register new ap).
5. **Probe-KILL is now the cohort default behavior.** The 1.9 ms file-probe
   pattern (P1/P2/P3 tripwire in MATH.md) is the correct, honest verdict
   for any audit-2026-04-17 claim until upstream rebuilds. Do not run heavy
   MLX on probe-FAIL preconditions — guardrail 1006/1009 holds.

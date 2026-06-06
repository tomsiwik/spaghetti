# LEARNINGS.md — exp_followup_lora_scale_safe_sweep

## Core Finding
17th consecutive audit-2026-04-17 cohort precondition-probe KILL (K1553
UNMEASURABLE, 0/3 preconditions, wall 0.027 s). The LORA scale-safety
sweep cannot run because (P1) flagship-5 enumeration source
`.audit/supported_00.md` is not in this repo revision, (P2) 0/120 dirs
contain both `LORA_SCALE=20` configs AND `*.safetensors` baselines, and
(P3) upstream T2.1 `_reconstruction_note` documents `datasets`/`dill`
Python 3.14 runtime incompat that blocks the retraining pipeline.

## Why
Three mechanically-independent upstream gaps stack:
1. Missing scope-definition file — without an authoritative flagship-5
   list, any chosen subset is arbitrary and fails the audit's pre-reg.
2. Missing baselines — cannot compare "scale ≤ 8" against unobserved
   scale=20 adapters. Repo holds only 23 safetensors across 656 dirs
   per reviewer — baselines never existed or were not committed.
3. Missing toolchain — datasets iter / dill pickling breaks under
   Python 3.14, so even a rerun cannot regenerate the baselines.

## Implications for Next Experiment
- **Do NOT claim a 18th audit-2026-04-17 cohort member.** 10 researcher +
  10 analyst escalations now logged; ap-017 already auto-injects the
  antipattern. Claim queue filter on `tag=audit-2026-04-17` remains the
  blocking orchestrator action.
- **Mechanical upstream unlock requires BOTH (a) Python 3.14 toolchain
  fix (pin datasets/dill or downgrade to 3.12) AND (b) `experiment update
  --status open` on `exp_p1_t2_single_domain_training`.** Reopening alone
  creates a stuck-claim — verified analyst iter-9.
- **Next researcher priority ranking**:
  1. If orchestrator filter lands → claim a non-cohort P≤2 experiment.
  2. If orchestrator hasn't acted → probe `experiment list --status open
     -p 2` for any non-`audit-2026-04-17` candidate; if none, escalate
     to orchestrator via event payload one more time then HALT — 17
     identical probe-KILLs is sufficient evidence of saturation.
- **No antipattern addition**: ap-017 (cohort precondition-probe KILL)
  already covers all 17 instances; REVIEW 17/17 PASS.

# LEARNINGS — exp_g4_energy_vs_ridge_routing

## Core Finding
10th consecutive audit-2026-04-17 cohort precondition-probe KILL. K1588
UNMEASURABLE. All three preconditions (P1 adapter weights, P2 ridge baseline,
P3 Gemma 4 energy-gap reference) fail on independently verified disk/DB state.
REVIEW 17/17 PASS or N/A — clean probe, no fabrication.

## Why
The same single upstream (`exp_p1_t2_single_domain_training` retrain) gates
every audit-2026-04-17 micro experiment. Prior analyst passes (8th and 9th
KILLs) already asked Ralph to pick out-of-cohort; the claim queue ignored it
because `experiment claim` is auto-ordered with no tag-filter flag. One more
cohort KILL was therefore the cheapest honest response (0.003 s) — but the
orchestration gap is now a first-class bug, not a transient nuisance.

## Implications for Next Experiment

1. **Cohort is 10/10 saturated.** Ralph MUST NOT accept an 11th
   audit-2026-04-17 claim. If `experiment claim` returns one, the researcher
   hat should emit a short KILL probe immediately (same template as the last
   three) rather than run any MLX work.

2. **Escalate upstream promotion.** `exp_p1_t2_single_domain_training` retrain
   with LORA_SCALE=5, `max_tokens ≥ 512`, `enable_thinking=True`,
   5 disjoint domains (math, code, medical, finance, legal), rank sweep
   `{2,4,6,12,24}`, grad-SNR logging — cumulative scope across 10 cohort
   findings. This should be promoted to a blocking task that 10 cohort
   downstreams auto-reopen on completion.

3. **Claim-queue filter is the real fix.** Until the upstream lands, the
   orchestrator must either flip cohort members to `status=blocked` or
   filter the `audit-2026-04-17` tag from the claim queue. Analyst-level
   learning.complete messages are clearly not enough — the queue keeps
   returning cohort members.

4. **Out-of-cohort priority ≤ 2 picks only.** Until (2) lands, the next
   researcher activation must inspect the claim result and, if the claim is
   audit-2026-04-17-tagged, probe-KILL it in <10 s. Any non-cohort
   priority ≤ 2 experiment is preferred work.

5. **Don't add a new antipattern.** `ap-017 precondition-probe-cohort-stall`
   already covers instances #1–#10 of this pattern. REVIEW 17/17 PASS —
   no process bug in this iteration. Reviewer explicitly declined a new
   finding-level antipattern. Adding one now would inflate the auto-inject
   budget without signal.

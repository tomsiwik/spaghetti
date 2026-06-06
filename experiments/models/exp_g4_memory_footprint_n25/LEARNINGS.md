# LEARNINGS — exp_g4_memory_footprint_n25

## Core Finding

14th consecutive `audit-2026-04-17` cohort precondition-probe KILL. K1596
UNMEASURABLE: P1 (base resolvable) PASS, but P2 (0/25 v_proj+o_proj
safetensors across 4 canonical dirs) and P3 (upstream T2.1 verdict=KILLED,
all_pass=false) FAIL. Probe wall 6.3 ms, no MLX load. REVIEW 17/17 PASS or
N/A.

## Why

Peak-RSS is only meaningful when the thing being measured is actually
attached. With zero deltas on disk and an upstream loader that is itself
KILLED, any RSS number would silently change the meaning of K1596. The
pre-registered tripwire required all three preconditions to pass jointly
— exactly the structural protection ap-017 prescribes against
"measure-something-adjacent" false findings.

## Implications for Next Experiment

1. **Cohort 14/14 saturated.** Same single upstream blocks all members.
   `experiment claim researcher` is still ordering cohort members despite
   six analyst escalations — analyst-tier escalation is not the right
   lever. The lever is orchestrator-side: claim-queue filter on
   `tag=audit-2026-04-17` or promote the upstream rebuild to a
   first-class blocking task.
2. **Highest-leverage single claim** remains `exp_p1_t2_single_domain_training`
   rerun (LORA_SCALE=5, max_tokens≥512, rank sweep {2,4,6,12,24},
   grad-SNR logging, ≥5 disjoint domains incl. finance+legal, materialized
   v_proj+o_proj safetensors at canonical paths). One probe-PASS unblocks
   ≥14 downstreams.
3. **Out-of-cohort priority alternatives** to consider before queue
   re-claims another cohort member: any P2 lacking the
   `audit-2026-04-17` tag, or P1/P0 work in the open queue.
4. **Memory-footprint claim is pre-validated by Finding #74** (deltas are
   small vs base); the open question is integration mechanics (N>1
   simultaneous mount on Gemma 4 MLX), not memory accounting per se.
   When this experiment is re-run post-upstream, expect base+25Δ ≈ base
   to within ≤200 MB.
5. **Antipattern ap-017** now covers 14 instances; no new memory.
   Reviewer iter-6 already declined to add #14. Do not inflate.

## Handoff

Researcher next. If queue returns a 15th cohort member, probe-KILL in
<10 s using the established tripwire pattern. Otherwise claim out-of-cohort
or claim the upstream directly.

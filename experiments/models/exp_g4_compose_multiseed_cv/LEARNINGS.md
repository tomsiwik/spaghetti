# LEARNINGS — exp_g4_compose_multiseed_cv

## Core Finding
KILL — K1590 (CV across 3 seeds) UNMEASURABLE. 12th consecutive cohort
`audit-2026-04-17` precondition-probe KILL. P1 (0/15 seeded safetensors),
P2 (upstream T2.1 KILLED, all_pass=false), P3 (no landed cohort MMLU-Pro
baseline) all fail. Same single upstream blocker as Findings #605–#618.

## Why
The cohort is 12/12 saturated. Every micro experiment tagged
`audit-2026-04-17` depends on `exp_p1_t2_single_domain_training`
landing as SUPPORTED at LORA_SCALE=5, max_tokens≥512, 5+ disjoint
domains, with a rank sweep and grad-SNR logged. Until that single
upstream rebuilds, every downstream probe is structurally UNMEASURABLE.

Analyst-hat escalation alone has failed four iterations in a row. The
claim queue keeps returning cohort members; the reviewer confirmed the
real fix sits at the orchestrator layer, not the analyst layer.

## Implications for Next Experiment

1. **Researcher MUST probe-kill any 13th cohort claim in <10 s.** The
   pattern is identical: 3 precondition files, 0.0 s wall, status=killed,
   no MLX, no training. Do not attempt remediation — the upstream is
   not the researcher hat's responsibility.

2. **Orchestrator fix (still outstanding).** `experiment claim
   researcher` needs a filter that excludes `tag=audit-2026-04-17`
   until the upstream rebuild completes, OR the upstream rebuild must
   be promoted to a first-class blocking task whose completion
   auto-releases the 10+ downstream cohort items. This is the actual
   bottleneck; five analyst escalations have now logged it.

3. **Prefer out-of-cohort work.** If a non-cohort experiment exists at
   priority ≤ 2, the researcher hat should claim it preferentially. The
   `experiment claim` CLI currently ignores this preference — another
   reason the orchestrator-level filter is the correct fix.

4. **Upstream rebuild scope is now fully specified** by the 12 downstream
   probe-KILLs. A researcher claiming the upstream itself can execute
   directly: LORA_SCALE=5, max_tokens≥512, 5 disjoint domains
   (finance/legal/math/code/medical), rank sweep {2,4,6,12,24}, grad-SNR
   per layer logged. This is the highest-leverage single experiment in
   the queue.

5. **Reproducibility CV is a VALID question, not a flawed one.** When
   the upstream lands, this K1590 measurement (σ/μ across seeds) is
   exactly the right reproducibility gate before any composition
   claim. Keep the MATH.md and code for that rerun; no re-derivation
   needed.

## Antipattern memory
Not added. REVIEW passed 17/17. `ap-017` (cohort-upstream-saturation)
already tracks instances #1–#12; adding a 12th sub-instance would inflate
auto-inject without new signal. The fix is orchestration-layer, not
hat-behavior.

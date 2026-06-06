# LEARNINGS — exp_g4_nre_vs_uniform_compose

## Core Finding
K1579 UNMEASURABLE. Pre-registered 3-precondition probe → all FAIL: 0/5 Gemma 4
domain adapter safetensors on disk (P1), upstream `exp_p1_t2_single_domain_training`
KILLED (P2), base GSM8K = 0.0% from max_tokens=256 format artifact (P3). 8th
precondition-probe KILL in the `audit-2026-04-17` cohort. REVIEW passed 17/17
adversarial checks; verdict KILL not REVISE — the probe is the correct honest
routing, not a process failure.

## Why
Finding #275 (NRE Karcher mean on Qwen3-4B) transfers geometrically to Gemma 4
only if the five per-domain `ΔW_i` exist and are non-zero. None do: math / code
/ medical dirs hold only `adapter_config.json` (mem-antipattern-017 class,
instances #9/#10 share exact P1 failure mode); finance / legal were never
trained. Even if the adapters were rebuilt, the format-artifact base=0.0%
(P3) would nullify the NRE-vs-1/N comparison since `acc(merge) − acc(base)`
collapses when `acc(base) ≈ 0`. The cohort standing rule (no ~4h MLX retraining
inside a researcher iteration) forbids the researcher from synthesising the
adapters, so the UNMEASURABLE routing is the only honest completion.

## Implications for Next Experiment
1. **Cohort is fully saturated.** 8 consecutive probe-KILLs all route to the
   same upstream: `exp_p1_t2_single_domain_training` must be rerun at
   LORA_SCALE=5 with disjoint math / code / medical corpora and
   max_tokens ≥ 512 before ANY Gemma 4 N ≥ 2 composition/routing experiment
   can produce a measurable K. Ralph SHOULD stop claiming downstream probes
   in this cohort until the upstream is rebuilt — further probe-KILLs add no
   new information (same failure mode as instances #1–#10 of ap-017).
2. **Rebuild-the-upstream IS the next experiment.** Promote "rerun
   `exp_p1_t2_single_domain_training` at LORA_SCALE=5, max_tokens ≥ 512,
   disjoint corpora + finance + legal" from recovery-path footnote to a
   first-class experiment. Success unblocks all 8 cohort descendants
   simultaneously (batch-unblock, per ap-017 remediation ticket
   `P11.ADAPTER-REBUILD`).
3. **Out-of-cohort work remains unblocked.** Priority-≤2 experiments that do
   NOT depend on the five cited Gemma 4 domain adapters (e.g. CoT/thinking
   format studies, single-adapter SFT baselines, non-Gemma 4 ports) can
   proceed this iteration without waiting on the upstream rebuild.
4. **No antipattern added.** ap-017 already enumerates 10 instances of the
   same DIR-EXISTS ≠ WEIGHTS-EXIST failure; adding an 11th inflates the
   auto-injected memory without new signal. The pattern is saturated; the
   fix is the upstream rerun, not another catalog entry.
5. **No external literature needed.** The KILL is mechanical (disk I/O +
   upstream DB verdict), not a negative-result refutation of the NRE
   geometric argument, so no `experiment ref-add` is warranted.

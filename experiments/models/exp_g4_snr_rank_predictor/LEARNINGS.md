# LEARNINGS — exp_g4_snr_rank_predictor

## Core Finding
KC #1586 / #1587 UNMEASURABLE. Pre-registered 3-precondition probe → all FAIL:
0/5 r=6 + 0/25 rank-sweep adapter safetensors on disk (P1), 0/5 grad-SNR
spectra (P2), upstream `exp_p1_t2_single_domain_training` KILLED with
base_gsm8k_pct=0.0 format artifact (P3). 9th consecutive probe-KILL in the
`audit-2026-04-17` cohort. REVIEW passed 17/17 adversarial checks; verdict
KILL not REVISE — honest probe routing, not a process failure.

## Why
Finding #154 (r_95 SNR predictor, synthetic micro d∈{64,128,256}) is a transfer
hypothesis: it predicts real-domain within-2x ≈ 0.85 for 4-bit Gemma 4 only if
(a) real 25-adapter rank sweeps exist, (b) per-step gradient singular spectra
were logged. Neither artifact exists — the upstream runner never produced
safetensors (ap-017 class, instances #9/#10 share exact P1 mode) and never
logged grad-SNR. The claim is therefore a conditional with an UNSATISFIED
antecedent; affirming KC #1586 without data would be fabrication, relaxing it
post-hoc would break KC discipline (PLAN.md Part 1). Cohort standing rule
(~12h MLX for 25 trainings + SNR logging is out of scope for a researcher
iteration) forces the UNMEASURABLE routing.

## Implications for Next Experiment
1. **Cohort is total-saturated.** 9 consecutive probe-KILLs all route to the
   same upstream. Further claims on `audit-2026-04-17` downstream probes add
   zero information — every probe will fail on the same P1/P2/P3 checks.
   Ralph MUST NOT claim a 10th cohort downstream until the upstream rebuild
   lands.
2. **Upstream rebuild scope now explicit.** The rebuild is: `exp_p1_t2_single_domain_training`
   at LORA_SCALE=5, max_tokens ≥ 512, disjoint math/code/medical/finance/legal
   corpora, AND rank sweep {2,4,6,12,24} (125 trainings total), AND per-step
   gradient L2 + step-count logging for `grad_snr.json` reconstruction.
   Previous analyst learning only captured scale/tokens/corpora; SNR-rank
   predictor adds the rank-sweep + SNR-logging requirements.
3. **Next researcher pick MUST be out-of-cohort.** Any priority-≤2 experiment
   that does NOT depend on the five Gemma 4 domain adapters (non-Gemma 4
   ports, single-adapter baselines, CoT/thinking format, synthetic-only
   spectra) is fair game. Prefer experiments whose MATH.md lists zero
   dependencies on `exp_p1_t2_single_domain_training`.
4. **Transfer claim (Finding #154 → real Gemma 4) remains open, not
   refuted.** The v2 version of this experiment keeps KC #1586/#1587 frozen
   at 0.90 / +20pp; a post-rebuild measurement of e.g. 0.55 would be a real
   negative (predictor fails to transfer to 4-bit), not a KC relaxation
   target.
5. **No antipattern added.** ap-017 already enumerates 10+ instances of the
   DIR-EXISTS ≠ WEIGHTS-EXIST failure mode; a 12th inflates auto-inject
   without new signal. No external literature warranted — KILL is mechanical
   (missing artifacts), not a refutation of the SNR rank argument.

# LEARNINGS — exp_g4_behavioral_eval_suite

## Core Finding
13/13 `audit-2026-04-17` cohort saturation. K1593 (4-benchmark behavioral AUC ≥
0.85) UNMEASURABLE: P1 (0 safetensors), P2 (only mmlu_pro wired), P3 (binds to
P1) all FAIL. Same single upstream (`exp_p1_t2_single_domain_training`) gates
this and Findings #605, #606, #608, #610, #611, #612, #613, #615, #616, #617,
#618, #619, #620.

## Why
The cohort was designed assuming the upstream T2.1 LoRA training landed with
working adapter safetensors at LORA_SCALE=5, max_tokens≥512, 5+ disjoint
domains. The actual upstream verdict is KILLED: `all_pass=false`, no
`lora_scale` field, format-artifact `base_gsm8k_pct=0` from `max_tokens=256`
truncation, and zero adapter safetensors. Every cohort downstream therefore
collapses to ap-017 (fabricated metric from missing upstream) the moment it
loads MLX. The pre-registered tripwire is the only honest exit.

## Implications for Next Experiment

1. **Direct upstream claim — researcher path of least resistance.** The
   highest-leverage single experiment in the queue remains
   `exp_p1_t2_single_domain_training` (rerun spec: LORA_SCALE=5,
   max_tokens≥512, 5+ disjoint domains, rank sweep {2,4,6,12,24}, grad-SNR
   logging, behavioral eval per Finding #615). A researcher claiming and
   running it directly unblocks 10+ cohort downstreams in one execution.

2. **Sixth analyst escalation logged; orchestrator behavior unchanged.** Five
   prior `learning.complete` payloads requested out-of-cohort picks; the
   `experiment claim researcher` queue continues to auto-order
   `audit-2026-04-17` members. Analyst hat is the wrong layer for this fix.
   Orchestrator-level claim-queue filter on `tag=audit-2026-04-17` (or
   first-class promotion of the upstream rerun as a blocking task) is the
   architectural fix and must come from outside the analyst-researcher-reviewer
   triangle.

3. **14th cohort claim must probe-KILL in <10s.** Until the upstream lands or
   the queue is filtered, the next researcher iteration will likely receive a
   14th cohort member. Reuse the pre-registered tripwire pattern
   (P1/P2/P3 file-existence probe, no MLX, results.json with verdict=KILLED,
   `experiment complete --status killed --k <id>:fail`). Wall time target
   <10s. Do not load MLX. Do not invent a measurement.

4. **Eval-suite design holds; only inputs are missing.** The 4-benchmark
   harness design (MMLU-Pro + GSM8K + HumanEval + MedMCQA) is sound and
   addresses Finding #615's behavioral-coverage gap. When upstream rebuilds,
   resurrect this experiment with K1593 unchanged — only P1 needs to flip
   from FAIL to PASS for the AUC measurement to be defined.

5. **No antipattern memory added.** REVIEW 17/17 PASS, reviewer already
   declined to register a new ap-* memory; ap-017 already covers all 13
   cohort instances. Adding ap-017 instance #14 inflates the auto-inject
   budget without new signal.

## Reference (no `experiment ref-add` needed)
The blocker is internal (`exp_p1_t2_single_domain_training` results.json),
not literature; no external paper added.

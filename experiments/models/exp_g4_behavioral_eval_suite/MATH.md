# MATH.md — exp_g4_behavioral_eval_suite

## Hypothesis

A 4-benchmark behavioral eval harness (MMLU-Pro, GSM8K, HumanEval, MedMCQA) applied
to Gemma 4 E4B 4-bit + per-domain LoRA adapters separates the correct-domain adapter
from wrong-domain adapters with AUC ≥ 0.85.

Motivated by Finding #210 (`exp_behavioral_eval_framework` on BitNet) which showed
behavioral separation is undetectable by keyword/length metrics — it requires
execution-based per-sample correctness signals.

## Preconditions (pre-registered)

The KC is only measurable if all three hold at run time:

- **P1** — Gemma 4 E4B 4-bit per-domain LoRA adapters exist on disk as `.safetensors`
  files (at minimum: math, code, medical from upstream
  `exp_p1_t2_single_domain_training` + mmlu-pro general).
- **P2** — The 4 benchmark harnesses (MMLU-Pro, GSM8K, HumanEval, MedMCQA) are
  wired to `mlx-lm` Gemma 4 E4B 4-bit with canonical prompt templates — not
  placeholder format/keyword checkers.
- **P3** — Per-sample correctness labels (0/1 per (prompt, adapter) pair) are
  recordable so an AUC can be fitted across adapter swaps.

## Tripwire

If any of P1/P2/P3 FAIL at run-start the KC **K1593** (AUC ≥ 0.85 across 4
benchmarks) is **UNMEASURABLE** and the experiment is marked `KILLED` per the
cohort standing rule (11 prior cohort KILLs already logged against the same
upstream: Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618/#619).

No heavy compute runs until P1/P2/P3 all pass.

## Kill Criteria (locked)

- **K1593**: AUC ≥ 0.85 across 4 benchmarks — unmeasurable without P1+P2+P3.

## Dependencies

- Upstream blocker: `exp_p1_t2_single_domain_training` rerun at LORA_SCALE=5,
  `max_tokens>=512`, 5+ disjoint domains, rank sweep, grad-SNR logging.
  Same blocker as the 12-experiment cohort KILL chain (Findings #605–#619).

## Assumptions

- Per-sample binary correctness is the correct AUC substrate (not distributional
  similarity). Justified by Finding #210: metric-based eval failed to detect
  behavioral differences; execution-based eval did.
- A fabricated AUC from placeholder eval would be worse than UNMEASURABLE — it
  would violate ap-017 (the antipattern tracked across all 12 prior cohort KILLs).

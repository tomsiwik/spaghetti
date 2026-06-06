# MATH — exp_model_mtbench_composed

## Claim
Pierre N=5 composed adapter stack (math + code + medical + sql + bash)
retains open-ended generation quality within 0.2 MT-Bench overall of
Gemma-3n-E4B base, and no single MT-Bench category drops below base − 0.5.

## Pre-registered Kill Criteria (locked)

| id    | text                                                                                             | meaning |
|-------|--------------------------------------------------------------------------------------------------|---------|
| K1691 | (reused naming from DB) Pierre N=5 composition MT-Bench score ≥ (base − 0.2) on overall average  | DB id 1697 |
| K1698 | No category (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities) < base − 0.5 | DB id 1698 |
| K1699 | GPT-4 judge (or open alternative) used consistently across runs                                  | DB id 1699 |

## Preconditions (LOCKED at claim-time, 2026-04-18)

Before any MT-Bench sweep is meaningful, three preconditions must hold.
These are measured by a light probe (filesystem + registry + upstream audit),
not by any generation call — failing any one makes K1697/K1698/K1699
structurally unmeasurable, not merely unmet.

| id | precondition                                                                 |
|----|------------------------------------------------------------------------------|
| P1 | All 5 Pierre adapter safetensors exist on disk: `adapters/{math,code,medical,sql,bash}/*.safetensors` |
| P2 | MT-Bench harness (FastChat or `lm-eval-harness` MT-Bench task) imports cleanly |
| P3 | Upstream experiment exp_p1_t2_single_domain_training has `results.json.verdict != "KILLED"` and no un-addressed audit flags |

## Theorem (probe first)

**Precondition Probe Theorem.** For any macro comparison-over-composition
experiment, let P = {p₁, ..., pₖ} be the set of preconditions. If any pᵢ
is FAIL, the kill criteria are structurally unreachable — measurements
cannot be produced at all, so K-FAIL is not a genuine negative
(hypothesis-is-false) but an infrastructure-blocked gap. The correct
verdict is KILLED with explicit P-fail annotations; re-opening requires
preconditions to be restored.

**Proof sketch.** MT-Bench overall score is a function of 8 category
scores, each of which requires the composed model to produce outputs for
80 multi-turn prompts. Output generation requires loading weights (P1),
routing them through a harness (P2), and the comparison must be against
a non-invalidated baseline (P3). Any broken link makes the measurement
uncomputable → K1697–K1699 cannot be scored. Reported FAIL without
distinguishing "unmeasured" from "measured-and-fell-short" would conflate
infrastructure state with hypothesis state. QED.

## Predictions

- P1 FAIL: `adapters/code/` does not exist; `adapters/{math,medical,sql,bash}/` exist but hold only config stubs (no `.safetensors` weights).
- P2 PASS: a MT-Bench harness (FastChat `fastchat.llm_judge.gen_model_answer` or similar) is importable. (Probe scope: import succeeds; full generation pipeline not required for the probe.)
- P3 FAIL: T2.1 (exp_p1_t2_single_domain_training) audited 2026-04-18, `results.json.verdict == "KILLED"` with metric-swap (MedQA vs MedMCQA) and format-artefact (max_tokens=256 CoT truncation) audit flags. Both trained-domain deltas (med_medmcqa_pct) and baseline measurements are therefore load-bearing but unreliable.

Probe outputs are logged verbatim to `results.json` so the verdict is
fully auditable without re-running.

## Assumptions / Decisions

- The probe is INTENDED to be the whole experiment for this iteration.
  Once P1/P3 are restored (adapters re-trained from a clean T2.1), a v2
  runs the full MT-Bench sweep. Until then, running a sweep against
  missing adapters or an audited-KILLED baseline would waste 4-6 hours
  of compute and produce a verdict that confuses infrastructure state
  with capability state.
- No GPT-4 API available in the Ralph sandbox for K1699. MT-Bench's
  open judge (`llm-judge-Llama-3-70B-Instruct` via FastChat) is the
  intended substitute when the sweep eventually runs; this decision
  does not affect the current probe verdict.
- Third macro comparison this loop (after llama31_8b and qwen3_4b) to hit
  the same P1+P3 fail pattern → precondition-probe is now a class-level
  standing rule, not a single-experiment finding.

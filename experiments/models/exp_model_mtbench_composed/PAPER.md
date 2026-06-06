# PAPER — exp_model_mtbench_composed

**Verdict:** KILLED — preconditions P1, P2, and P3 all FAIL.
**Mode:** precondition-probe (not a generation sweep; 0.1 s runtime).
**Runtime:** 0.1 s.
**Date:** 2026-04-18.

## Prediction vs Measurement

| id     | claim                                                                             | predicted | measured                                                      | result |
|--------|-----------------------------------------------------------------------------------|-----------|---------------------------------------------------------------|--------|
| P1     | all 5 Pierre adapter safetensors on disk                                          | FAIL      | 0/5 domains have `.safetensors`; `adapters/code/` doesn't exist; other 4 are config-only stubs | FAIL ✓ |
| P2     | MT-Bench harness importable (FastChat `llm_judge`, or lm-eval MT-Bench fallback)  | PASS (weak) | both FAIL: `fastchat` not installed; `lm_eval.tasks` raises AttributeError | FAIL (prediction wrong) |
| P3     | T2.1 upstream not KILLED                                                          | FAIL      | T2.1 `results.json.verdict == "KILLED"` with audit flags (metric-swap + format-artefact, 2026-04-18) | FAIL ✓ |
| K1697  | MT-Bench overall ≥ base − 0.2                                                     | FAIL (structurally unmeasurable) | FAIL — P1 blocks | FAIL ✓ |
| K1698  | No category < base − 0.5                                                          | FAIL (structurally unmeasurable) | FAIL — P1 blocks | FAIL ✓ |
| K1699  | Judge used consistently                                                           | FAIL (structurally unmeasurable) | FAIL — P1 blocks; judge question not reached | FAIL ✓ |

P2's prediction was optimistic — siblings' lm-eval-harness import passes,
but for MT-Bench specifically the harness is FastChat, which is not
installed. Either way the verdict is KILLED from P1 alone; P2's status
affects only whether a later sweep can be wired up once adapters exist.

## Why probe instead of sweep

A full MT-Bench sweep is 80 prompts × 2 turns × 8 categories × 2 runs
(Pierre composed + Gemma-3n-E4B base) + judge calls, on target hardware
roughly 4-6 h. Running that before adapters exist or before T2.1 is
rebuilt would:

1. Measure nothing for the trained-domain categories (math, coding,
   extraction) — the adapters don't exist.
2. Carry T2.1's metric-swap and format-artefact flaws into the
   composition layer (base comparison distorted; medical adapter never
   trained on the DB-tracked task).
3. Produce a results.json whose FAIL can't distinguish "composition
   degrades quality" from "adapters not loaded" — i.e. a false-negative
   on the research claim.

Per the precondition-probe class-level rule (3rd instance this loop),
the probe resolves infrastructure state in seconds, so KILL is honest
and cheap, and the v2 sweep can be launched the moment preconditions
hold.

## Permanently learned (class-level standing rules)

### 1. Precondition-probe is the default for macro sweeps
Every macro comparison (peer-comparison, composition, benchmark) runs
a light filesystem + import + upstream-audit probe **before** a
multi-hour generation sweep. This loop's 3-instance run
(llama31_8b, qwen3_4b, mtbench_composed) all found preconditions
absent; the probe correctly refused to proceed. Standing rule.

### 2. Adapter registry ≠ adapter artefacts (directory-existence corollary)
`registry.json` advertises 5 domain adapters (math/code/medical/sql/bash)
with scores and paths. Reality: `adapters/code/` doesn't exist at all;
the other 4 dirs hold config stubs but no weights. The stronger form
of the rule: **check directory existence AND `.safetensors` count — not
just registry entries.**

### 3. Downstream P1 macros inherit upstream audit flags
T2.1 was audited 2026-04-18 (Supported → KILLED) for metric-swap and
format-artefact. Every macro that compares against T2.1-trained adapters
inherits the invalidation. 3rd sibling to trigger this rule today; now
class-level.

### 4. Harness import predictions must be task-specific
P2's predicted PASS was based on lm-eval-harness importability (true for
siblings). But MT-Bench specifically is a FastChat task; `lm_eval` is
not a substitute. Corollary: when predicting P2 for a benchmark, probe
the *exact* module the benchmark lives in, not a generic harness
convention.

## Reviewer notes

- This experiment wrote no adapter weights and ran no generation.
  `run_experiment.py` is a probe, not a smoke of the sweep.
- KCs were set before the probe ran (see MATH.md); the probe cannot
  upgrade PROVISIONAL to KILLED if preconditions pass — it only detects
  infrastructure state and routes to KILL when preconditions fail.
- Antipattern scan clean: no composition math bug, no LORA_SCALE, no
  routing collapse, no shutil.copy cloning, no hardcoded pass, no
  eval-template truncation, no proxy-model substitution, no smoke-as-full.

## Routing signal (for next hats)

- No auto-spawn of a v2 sweep. The v2 requires T2.1 to be rebuilt:
  MedQA USMLE 5-choice (matching DB KC #1030), `max_tokens ≥ 512` to
  avoid CoT truncation, and `.safetensors` weights persisted to
  `adapters/{math,medical,sql,bash}/`. `adapters/code/` must also be
  created if code-domain is retained in the composition.
- Analyst should promote the 4-part standing rule block above into
  LEARNINGS.md and consider whether the existing antipattern catalog
  needs a "missing-precondition-masks-research-claim" entry (3 instances
  this loop is enough for class status).

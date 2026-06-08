---
name: conductor
description: Orchestrate the next open experiment proposal end-to-end as an agent team — claim it, then drive it through research → adversarial review → analysis to a REAL verdict + finding (no mocks). Use when asked to "pick the next experiment proposal and work on it", run the experiment conductor, or drain the experiment queue.
---

# Conductor — drive the next experiment to a real verdict (agent-team lead)

You are the **conductor (team lead)**. Drive ONE *novel* experiment from a creative spark to a REAL verdict + finding using subagents. You orchestrate; the subagents do the work in their own contexts. **The point is the creative edge — do NOT just drain the incremental backlog.** The full driver is `PROMPT.md` (read it); this skill is the summary.

## Contract
- **Spark first.** Phase 1 generates a NOVEL, frame-breaking hypothesis (not a stale backlog proposal). Phase 2 executes it.
- **One experiment, to completion.** Drive it to a terminal status (`supported`/`killed`/`provisional`) WITH a recorded finding. A Stop hook keeps you going until then.
- **NO MOCKS.** Real executed code (real MLX/model/data), never a smoke/numpy placeholder. A PreToolUse gate hard-blocks completing a smoke as supported/killed; the Reviewer is the judgment gate.
- **Minimal lead tokens.** Keep your own messages short. Delegate.

## Protocol
1. **Load the `experiment` skill** (exact CLI flags).
2. **Phase 1 — SPARK.** Spawn the **`sparker`** subagent (`.claude/agents/sparker.md`). It applies the perturbation prompts (`tooling/spark/prompts/`) + cross-model divergence (Gemini explores, GPT systematizes) and files ONE novel, frame-breaking, falsifiable, grounded, micro-runnable proposal (tags `spark`,`novel`), returning the new id. **Reject incrementalism** — if it's a follow-up/ablation/paper-reconstruction, send it back for a real perturbation.
3. `experiment claim conductor --id <spark_id>` (the spark's new id — NOT a stale backlog item).
4. **Researcher** ← the claimed id. Proof-first `MATH.md` (target-metric kill criteria), real `run_experiment.py`, `experiment run` (pueue, **blocking — no sleep/poll**), `PAPER.md`; reports the verdict + numbers or `BLOCKED: <reason>`. If BLOCKED, set provisional and re-spark.
5. **Reviewer** ← same id (fresh context). Adversarial no-mock checklist + `bash .claude/hooks/verify-experiment.sh <id>`, writes `REVIEW-adversarial.md`, routes PROCEED/KILL/PROVISIONAL/REVISE. **Only the reviewer calls `experiment complete`.** On REVISE: relay ≤3 fixes to the researcher, re-review (max 2 rounds).
6. **Analyst** ← `LEARNINGS.md` + `experiment finding-add` (status matches the reviewer's verdict).
7. Confirm terminal status + a finding exists. Report a one-line summary (claim, perturbation, verdict). Then stop (or, in drain mode, re-spark).

## Guardrails (source of truth: `experiments/GUIDE.md`)
- Proof-first: `MATH.md` before code, with ≥1 **target-metric** kill criterion (Finding #666 — proxies like PPL/cosine alone don't justify a verdict).
- `experiment run` (pueue) only — never bare `python`/`uv run`.
- Composition `Σ(Bᵢ@Aᵢ)`, `LORA_SCALE ≤ 8`, route per-sample.
- Behavioral outcomes over metrics. Don't accumulate antipattern memories; no taxonomies.

Delegate the work. Stay lean. Don't fabricate, and don't let a teammate fabricate.

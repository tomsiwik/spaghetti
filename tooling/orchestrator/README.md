# orchestrator/ — the JS Conductor (SECONDARY — deterministic plumbing demo)

> **The real conductor is the Claude Code agent-team in `docs/CONDUCTOR.md`** (runs real MLX, no mocks).
> This JS loop is dependency-light and deterministic but its fallback is a smoke sim — use it for
> plumbing demos, not for science.



The runnable orchestration layer from `docs/ORCHESTRATION.md`. This Claude Code session is the
Conductor; GPT-5.5 (codex) and Gemini are decoupled CLI workers; the experiment DB is the durable
record. **Orchestration is JS/bun** (matches the `experiment` CLI stack); the experiment code it runs
stays Python (numpy/MLX — the thing being orchestrated).

**This dir is config + docs.** The conductor *code* lives in the CLI package next to the command it
serves — `tooling/packages/cli/src/lib/conductor.ts` (exports `runQueue`) — and the `orchestrate`
command imports and calls it **in-process** (no subprocess). **Config is YAML — edit `conductor.yml`, not the code.**

## Run
```bash
experiment orchestrate                 # drain the queue in conductor.yml
experiment orchestrate jobs.json       # drain a custom queue
experiment orchestrate --status        # in-flight jobs + experiment states
# standalone (debug):  bun tooling/packages/cli/src/lib/conductor.ts [queue.json]
```

## What it does (per job, deterministic loop — no Opus tokens on orchestration)
`file` → `run` → `analyze` → `complete`:
- **file** — `explore` model (Gemini) adds a falsification angle to `MATH.md`; `systematize` model
  (GPT-5.5) best-effort generates `run_experiment.py`; **guaranteed fallback** if rejected; `experiment add`.
- **run** — runs the script in its dir (pueue is the production path); falls back if no `results.json`.
- **analyze** — DETERMINISTIC: `results.json` → status (smoke→provisional, per `experiments/GUIDE.md` rules).
- **complete** — writes PAPER/REVIEW; `experiment complete` (supported/killed) or `experiment update`
  (provisional — `complete` rejects it); `experiment finding-add`.

Every stage is error-wrapped: a failed worker / run / codegen falls back or records the reason and the
loop **continues**, so it drains a queue indefinitely. The Conductor process always exits 0.

## Config (`conductor.yml`)
`models` (cmd + model + timeout per CLI) · `roles` (explore/systematize/synthesize → model) ·
`spark_source` · `experiment` defaults (scale/tags/kill) · `queue` · `codegen` toggle.

## Files
`conductor.yml` (policy) · `queue.json` (example queue) — here. Code: `packages/cli/src/lib/conductor.ts`.
First proven case: `exp_gap_collapse_*` — the gap-collapse→forgetting hunch from `tooling/spark/pipeline_gap/`.

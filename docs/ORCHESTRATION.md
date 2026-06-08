# Conductor orchestration — design spec

> How we run multi-model experiment research after the API change: **this Claude Code session (Opus 4.8) is the Conductor**; GPT-5.5 (codex) and Gemini are decoupled queued workers; the experiment DB + pueue are the event bus + executor. Supersedes the autonomous `ralph.yml` loop. Status: spec → MVP.

---

## 1. What we keep vs. drop from `ralph.yml`

`ralph.yml` (running on upstream `../ralph-orchestrator`, a Rust hat-loop) gave us a contract worth keeping:

| Keep (the good contract) | Drop (what we already rejected) |
|---|---|
| **Event-driven role loop** — work routes between roles by events (`research.start → experiment.done → review.* → learning.complete`) | **Autonomous, no human gate** ("NEVER wait for user input") — replaced by checkpoints (`STATUS.md §6`) |
| **State/execution separation** — `experiment` CLI owns durable state (claim/complete/findings); `experiment run` + **pueue** own execution | **"NEVER spawn sub-agents"** — inverted: the Conductor *delegates* to a Claude team + model-worker jobs |
| **Role specialization** — Researcher / Reviewer / Analyst, each its own bounded invocation | **Single model** (claude-opus-4-6 for all hats) — replaced by per-role model assignment from cross-model data |
| **Discipline via injected guardrails** — now consolidated in `experiments/GUIDE.md` | **Throughput bias** (<30 min/hat, kill-fast) — replaced by mechanism/autopsy depth |
| **Memory injection** — bug-prevention `type: fix` only (grounding `type: fact` removed) | — |

**One-line:** keep the event loop + state/exec separation; move the conductor *into* this session and make it heterogeneous-model + human-gated.

---

## 2. Architecture: Conductor + Claude team + model-worker jobs + experiment event-bus

```
                ┌────────────────────────────────────────────────┐
   human gate ──┤  CONDUCTOR  = this Claude Code session (Opus 4.8)│  the only persistent
                │  reason · synthesize · kill-autopsy · repair ·   │  reasoning agent;
                │  decide-next · file/queue/complete experiments   │  team lead
                └───┬───────────────┬───────────────┬─────────────┘
        agent-teams │               │ pueue jobs    │ experiment CLI
       (Claude only)│               │ (any model)   │ (durable state)
        ┌───────────▼──┐   ┌────────▼────────┐  ┌───▼────────────────────────┐
        │ Claude team  │   │ MODEL WORKERS   │  │ EVENT BUS = experiment DB   │
        │ teammates    │   │ gemini  (explore)│  │ open→active→done + pueue    │
        │ (researcher, │   │ codex   (system.)│  │ completion = the messages   │
        │ reviewer,    │   │ claude  (synth.) │  │ (durable, survives crash)   │
        │ implementer) │   │  = queued jobs   │  └─────────────────────────────┘
        └──────────────┘   └─────────────────┘
```

- **Conductor (lead):** the one agent that reasons. Owns the control loop; does what only judgment can — synthesis, kill-autopsy, repairing a broken `run_experiment.py`, deciding the next hunch, and the human-facing checkpoints.
- **Claude execution team (agent-teams):** the old hats become *independent Claude teammates* (researcher writes `MATH.md`, implementer writes `run_experiment.py`, reviewer does adversarial review) — parallel, each own context, able to challenge each other. Used for the **Claude-side** work that benefits from parallelism.
- **Model workers (GPT-5.5, Gemini):** dispatched as **pueue jobs** (`gemini -p …`, `codex exec …`) by the Conductor or a teammate. They are *stateless one-shot jobs* — which is exactly what those CLIs are.
- **Event bus = experiment DB + pueue.** State transitions (`experiment complete`) and pueue task-finish are the messages. **This is the durable source of truth** — not the agent-teams task list (which is ephemeral / lost on resume).

### Model-role assignment (from the cross-model eval, `tooling/spark/`)
| role | model | why (measured) |
|---|---|---|
| **Explore** (diverge, wide hunches) | **Gemini** | widest diversity (.52) on every problem |
| **Systematize** (falsify, pre-register kills) | **GPT-5.5** | most convergent / strict; held Opus's hunches to repo rules |
| **Synthesize · Conduct · Repair · Review** | **Opus 4.8** | balances novelty with applied grounding; only persistent agentic CLI |

---

## 3. Does agent-teams fit? (the direct answer)

**Yes — for the Claude side, and it's exactly "orchestrate within the active session."** Agent-teams gives a lead (this session) + independent Claude teammates with a shared task list, mailbox, and `TaskCreated`/`TaskCompleted`/`TeammateIdle` quality-gate hooks. That cleanly separates Claude *orchestration* (lead) from Claude *execution* (teammates) — the researcher/reviewer/analyst hats become teammates that can run in parallel and challenge each other.

**But it does not, by itself, incorporate GPT-5.5 / Gemini** — **teammates are always Claude Code instances.** So the heterogeneous-model methodology is implemented as: a Claude teammate (or the lead) **shells out to the `gemini`/`codex` CLIs as queued jobs.** Agent-teams = Claude parallelism; pueue jobs = the other models; experiment DB = the durable bus that ties them together.

**Caveats (it's experimental):** session-resume drops in-process teammates, one team at a time, no nested teams, lead is fixed, task status can lag. → **Therefore the durable record stays in the experiment DB**, and agent-teams is used only for *in-session* parallel Claude work. Enable with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` (v2.1.32+).

**Net:** use agent-teams for the Claude orchestration/execution split; keep the experiment DB + pueue as the cross-session backbone; reach the other models via CLI jobs.

---

## 4. New CLI surface: `experiment orchestrate`

A thin command that turns the spark pipeline + experiment lifecycle into **queueable, model-agnostic stages** the Conductor drives. Lives at `tooling/packages/cli/src/commands/orchestrate.ts` (oclif auto-registers).

```bash
# dispatch one model-stage as a pueue job; records result against a target (experiment/hunch)
experiment orchestrate stage --model gemini|gpt-5.5|opus --role explore|systematize|synthesize \
    --in <file|finding-id> --out <file> [--queue]        # --queue => pueue add (async), else inline

# run the full spark DAG (explore→synthesize→systematize) on a finding, emit candidate hypotheses
experiment orchestrate spark --finding <id> --out <dir>  # wraps tooling/spark/pipeline.py

# Conductor control loop: drain newly-finished jobs/experiments and act
experiment orchestrate next                              # 1 tick: what completed? what's the next action?
experiment orchestrate status                            # in-flight jobs (pueue) + experiment states
```

Contract: every stage writes its artifact to disk **and** appends an event row (`stage`, `model`, `target`, `status`) the Conductor reads. `spark` is `pipeline.py` behind the CLI. Execution always goes through pueue (process isolation, completion signal) — never bare `uv run`.

---

## 5. The Conductor loop (control flow)

One **DAG per finding**, advanced by events, N findings in flight:

```
explore(Gemini job) → synthesize(Opus) → systematize(GPT job)
   → experiment add (+ pre-registered KCs)  → write run_experiment.py
   → experiment run (pueue)  →[completion event]→ Conductor: analyze results
   → repair & re-run (pueue) on failure | PAPER.md + REVIEW + experiment complete + finding-add
   → [human checkpoint: accept / pivot / next]  → pick next hunch from queue
```

- **Mechanical steps** (dispatch job, await completion, advance DAG, pick next) → automated by the Conductor.
- **Judgment steps** (synthesize, kill-autopsy, repair, accept/pivot) → the Conductor reasons; the **human gates** the accept/pivot/next decision.
- **Concurrency:** while finding A's experiment trains in pueue, the Conductor works finding B's spark/synthesis. The bus (experiment state) makes this safe across crashes.

---

## 6. MVP (build order)

1. **`experiment orchestrate spark --finding <id>`** — wrap `pipeline.py` behind the CLI (model-worker jobs via pueue). *(small)*
2. **One real cycle** — take the gap-collapse hypothesis (`tooling/spark/pipeline_gap/`) → `experiment add` + `MATH.md` with its pre-registered KCs → `experiment run` → Conductor analyzes → complete. *(proves the loop)*
3. **`experiment orchestrate next/status`** — the thin polling conductor over pueue + experiment state. *(makes it standing)*
4. **(optional) agent-teams** — spawn Claude researcher/reviewer teammates for the Claude-side steps, gated by `TaskCompleted` hooks that enforce `experiments/GUIDE.md` discipline.

Prove step 2 (one cycle, one finding) before generalizing — same discipline as the rest of this program.

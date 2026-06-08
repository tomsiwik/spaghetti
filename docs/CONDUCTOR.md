# The Conductor — run book

The conductor is a **Claude Code agent-team** session that takes the next open experiment proposal
and drives it to a **real, completed verdict + finding** — proof-first, real MLX, **no mocks** — using
the established 3-hat method (Researcher → Reviewer → Analyst) as independent teammates.

This file is the run book. The *code* is the established methodology, wired as Claude config:
`.claude/agents/{researcher,reviewer,analyst}.md`, `.claude/skills/conductor/SKILL.md`,
`.claude/hooks/*`, `.claude/conductor.settings.json`.

---

## How to run it

```bash
cd /Users/tom/Code/tomsiwik/llm
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude \
  --teammate-mode in-process \
  --settings .claude/conductor.settings.json
```

Then paste the generic prompt (always the same):

```
Pick the next experiment proposal from the experiment CLI and work on it.
```

That surfaces the `conductor` skill, which makes this session the **team lead**. To drain more than
one, say: *"…and keep going until the backlog is drained."*

---

## What happens

1. **Lead** loads the `experiment` skill, `experiment claim conductor` (picks the next open real-code proposal).
2. **Researcher** (teammate, opus) — proof-first `MATH.md`, invokes `/mlx-dev` + `/fast-mlx`, writes a **real** `run_experiment.py`, runs it via `experiment run` (pueue — can take a long time, that's expected), writes `PAPER.md`, reports the verdict or `BLOCKED:`.
3. **Reviewer** (teammate, opus, fresh context) — adversarial no-mock checklist + `verify-experiment.sh`, writes `REVIEW-adversarial.md`, routes PROCEED/REVISE/KILL/PROVISIONAL. **Only the reviewer calls `experiment complete`.**
4. **Analyst** (teammate, sonnet) — `LEARNINGS.md` + `experiment finding-add`.
5. The experiment reaches a terminal status (`supported`/`killed`/`provisional`) **with a finding**. The Stop hook won't let the lead quit before that.

## The no-mock guarantees (why this is different from the earlier placeholder)

- **PreToolUse gate** (`conductor-gate.sh`): a `experiment complete … --status supported|killed` is **denied** if `results.json` is missing, `is_smoke:true`, or has no verdict. Fake-green is structurally impossible. *(Proven: it blocks the old `exp_gap_collapse_inproc` smoke; it stays silent for a `provisional` update.)*
- **Independent Reviewer**: fresh context, runs the full adversarial checklist (tautological KC, `shutil.copy` adapter fakes, hardcoded `{"pass":True}`, model substitution, proxy-only verdicts).
- **Stop hook** (`conductor-stop-check.sh`): blocks the lead from stopping while any experiment is still `active` (cap: 60 continuations, then yields).

---

## How to verify a run (do this in the fresh session)

```bash
experiment get <id>                      # status is supported/killed/provisional (NOT active)
ls experiments/models/<id>/              # MATH.md run_experiment.py results.json PAPER.md REVIEW-adversarial.md LEARNINGS.md
python3 -c "import json;print(json.load(open('experiments/models/<id>/results.json'))['is_smoke'])"   # False for supported/killed
bash .claude/hooks/verify-experiment.sh <id>   # exit 0 = real
experiment finding-list | head          # a finding exists for <id>
```

If `results.json` has `is_smoke:true` and the status is `supported`/`killed`, the gate failed — that must never happen.

---

## Honest limitations

- **Agent teams is experimental** (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`). Known rough edges: no session-resume with in-process teammates, occasional task-status lag, one team per lead.
- **The first real run does real compute** on the M5 Pro and can take a long time — that's the point ("regardless of how long").
- **Teammates are Claude-only.** GPT-5.5 / Gemini are not teammates; if you want their complementary strengths for ideation/codegen they're invoked as CLI jobs (`tooling/spark/`), separate from this loop.
- **One experiment per prompt** by default. "Drain the backlog" works within a session, but hooks can't auto-loop unattended across many experiments without the lead continuing.

## Relationship to `experiment orchestrate` (the JS conductor)

`tooling/orchestrator/` + `experiment orchestrate` is the earlier **deterministic** loop. It is **secondary**:
fast and dependency-light, but it does not run real MLX — its built-in fallback is a smoke sim. **This
agent-team conductor is the real path** for producing genuine experiment results. Use `experiment orchestrate`
only for plumbing demos, not for science.

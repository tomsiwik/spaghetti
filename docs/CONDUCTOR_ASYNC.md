# Conductor — async finish-events (PLAN, not yet built)

## Why
`experiment run` blocking is the simplest non-polling wait, but the **Bash tool caps a single command at
10 min** (2 min default). Real MLX experiments can exceed that → a blocking call times out. We want: fire
the experiment async (pueue), free the session, and have it react **when the job finishes** — no `sleep`,
no poll. The producer of that finish-event is the **`experiment` CLI** (it already owns pueue), targeting
the specific conductor session.

## Mechanism (from the docs — verified)
- A **channel is an MCP server** declaring `capabilities.experimental['claude/channel'] = {}`, connected
  over stdio. It pushes events with `mcp.notification({ method: 'notifications/claude/channel',
  params: { content, meta } })`. Claude sees them as `<channel source="…" …>content</channel>` in-context.
- **Webhook pattern (ours):** the channel server also listens on `localhost:PORT`; an external process
  POSTs an event, the server forwards it as a channel notification. → `experiment` CLI = the external producer.
- Launch: `claude --dangerously-load-development-channels server:experiment …` (dev-load a local channel).
- **Events only arrive while the session is open**; notifications are fire-and-forget (not acknowledged).

## Architecture
```
experiment start
  ├─ uuid = <generated>                       # we SET the session id (no env var needed)
  ├─ claude --session-id <uuid>
  │         --dangerously-load-development-channels server:experiment
  │         --dangerously-skip-permissions --settings .claude/conductor.settings.json "<prompt>"
  └─ write {uuid, port} -> .ralph/agent/conductor_session

conductor (lead)                              experiment-channel (MCP server, stdio + localhost:PORT)
  ├─ experiment run --no-wait --notify <uuid> <id>     │
  │     (pueue submits; CLI records uuid->job)          │
  ├─ end turn (session free)                            │
  │                                                     │
  pueue job finishes ─► experiment CLI emitter POSTs ──►│ localhost:PORT  {exp, verdict, uuid}
  │                                                     │   ├─ filter: meta.session == uuid
  │                                                     │   └─ mcp.notification(content="exp <id> done: <verdict>")
  ◄──────── <channel source="experiment" exp=… verdict=…>done</channel> ───┘
  └─ react: read results.json -> reviewer -> analyst
```

### Pieces to build
1. **`tooling/channel/experiment-channel.ts`** (bun) — MCP server: `claude/channel` capability + a tiny
   HTTP listener on a fixed/derived `localhost:PORT`. On POST, validate `session==uuid`, push the notification.
2. **`experiment run --no-wait --notify <uuid>`** — record `uuid → pueue job id`.
3. **finish-emitter** — when a pueue job completes, POST `{exp, verdict, session:uuid}` to `localhost:PORT`.
   Options: (a) a `experiment watch` daemon that tails `pueue status` and emits on transition to Done;
   (b) a pueue group `--after`/callback if available; (c) the experiment-channel server itself polls pueue
   (polling lives in ONE small server, not in the token-billed agent — acceptable).
4. **`experiment start`** — generate uuid, pass `--session-id` + `--channels`, record `.ralph/agent/conductor_session`.

## The open risk (decides everything) — DOES a channel wake an idle headless session?
The docs say events "arrive while the session is open" but **do not confirm the model is re-invoked** when
idle under `--dangerously-skip-permissions`. It may queue until the next turn. **Verify before relying on it:**

> **V1 test:** launch a skip-permissions session, have it go idle, POST a channel event from another shell,
> observe whether the model takes a new turn on its own. Yes → channels alone suffice. No → use the fallback.

## Fallback ladder (use the highest tier that is proven to work)
1. **Blocking `experiment run`** — jobs ≤ ~10 min. Zero polling, zero wake problem. *Works today.*
2. **`/goal` + async pueue** — confirmed re-invoker: `/goal` starts a new turn after each one until the
   condition holds. Submit `--no-wait`; each turn does ONE cheap `experiment run --status` check and
   processes any finished job. This is the robust async path **even if channels don't wake** — and the
   channel, layered on top, just delivers the payload so the per-turn check is trivial. *Works today.*
3. **Channel finish-event** — the no-poll ideal; enable once V1 confirms it wakes an idle session.

## Recommendation
Build the channel server (piece 1–4) **and** drive the loop with `/goal` (tier 2). That gives the
event-driven payload the user wants while `/goal` guarantees forward progress regardless of the wake
behavior. Treat the channel as the optimization, `/goal` as the spine. Single conductor per machine for
now (no per-session inbound routing in the docs; the channel filters on `session==uuid`).

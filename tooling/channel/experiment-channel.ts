#!/usr/bin/env bun
/**
 * experiment-channel — a Claude Code CHANNEL (MCP server, stdio) that forwards async
 * orchestration finish-events into the conductor session. Claude Code spawns this as a
 * subprocess (via .mcp.json) with `--dangerously-load-development-channels server:experiment`.
 *
 * It also listens on localhost:PORT so external producers (the `experiment` CLI, run from a
 * pueue `callback:` or a SubagentStop hook) can POST a finish-event:
 *
 *   POST http://127.0.0.1:<port>  {"kind":"exp_done","exp":"exp_x","verdict":"killed"}
 *   POST http://127.0.0.1:<port>  {"kind":"team_done","agent_id":"a803…","phase":"researcher"}
 *
 * which arrives in the session as:
 *   <channel source="experiment" kind="exp_done" exp="exp_x" verdict="killed">exp_done exp_x killed</channel>
 *
 * NOTE: never write to stdout — it is the MCP transport. Registry + errors only touch the FS.
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";

const SESSION = process.env.CLAUDE_CODE_SESSION_ID ?? `pid${process.pid}`;
// Port 0 = OS-assigned, so any number of conductor instances coexist on one machine.
// EXPERIMENT_CHANNEL_PORT pins it (single-instance / debugging).
const PORT = Number(process.env.EXPERIMENT_CHANNEL_PORT ?? 0);
const ROOT = process.env.CLAUDE_PROJECT_DIR ?? process.cwd();

const mcp = new Server(
  { name: "experiment", version: "0.1.0" },
  {
    capabilities: { experimental: { "claude/channel": {} } },
    instructions:
      'Async orchestration events arrive as <channel source="experiment" kind="..." exp="..." agent_id="..." verdict="...">. ' +
      "kind=exp_done: a pueue experiment run FINISHED — read its results.json and continue (review). " +
      "kind=team_done: a subagent/teammate finished — agent_id identifies which; dispatch the next phase. " +
      "These are one-way; do NOT reply. React by running the next `experiment *` step. Never poll for completion — wait for these events.",
  },
);

await mcp.connect(new StdioServerTransport());

/** meta keys must be [A-Za-z0-9_]; values stringified. */
function toMeta(o: Record<string, unknown>): Record<string, string> {
  const m: Record<string, string> = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || v === null || v === "") continue;
    m[String(k).replace(/[^A-Za-z0-9_]/g, "_")] = String(v);
  }
  return m;
}

const server = Bun.serve({
  port: PORT,
  hostname: "127.0.0.1",
  async fetch(req) {
    if (req.method !== "POST") return new Response(`experiment-channel ok (session ${SESSION})`);
    let body: any = {};
    try {
      body = await req.json();
    } catch {
      body = { content: await req.text() };
    }
    // Only deliver events addressed to THIS session (or unaddressed).
    if (body.session && body.session !== SESSION) return new Response("ignored: other session");
    const content =
      body.content ??
      [body.kind ?? "event", body.exp ?? "", body.agent_id ?? "", body.verdict ?? ""].filter(Boolean).join(" ");
    await mcp.notification({
      method: "notifications/claude/channel",
      params: {
        content,
        meta: toMeta({ kind: body.kind, exp: body.exp, verdict: body.verdict, agent_id: body.agent_id, phase: body.phase }),
      },
    });
    return new Response("ok");
  },
});

// Per-session registry so producers (`experiment notify`) can find every live conductor's
// port. One file per session under .ralph/agent/channels/; removed on exit. The legacy
// single-file registry is kept for back-compat with watchers armed before this change.
const CHANNELS_DIR = `${ROOT}/.ralph/agent/channels`;
const regFile = `${CHANNELS_DIR}/${SESSION.replace(/[^A-Za-z0-9_-]/g, "_")}.json`;
const entry = JSON.stringify({ sessionId: SESSION, port: server.port, pid: process.pid }, null, 2);
try {
  mkdirSync(CHANNELS_DIR, { recursive: true });
  writeFileSync(regFile, entry);
  writeFileSync(`${ROOT}/.ralph/agent/conductor_channel.json`, entry);
} catch {
  /* best effort */
}
const cleanup = () => {
  try {
    rmSync(regFile, { force: true });
  } catch {
    /* best effort */
  }
};
process.on("exit", cleanup);
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
  process.on(sig, () => {
    cleanup();
    process.exit(0);
  });
}

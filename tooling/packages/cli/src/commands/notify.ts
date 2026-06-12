import { Command, Flags } from "@oclif/core";
import { existsSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { execSync } from "node:child_process";

interface ChannelEntry {
  sessionId: string;
  port: number;
  pid: number;
  file?: string;
}

/** Producer: POST an async finish-event to conductor session channel(s).
 *  Each channel server registers {sessionId, port, pid} in .ralph/agent/channels/<session>.json
 *  (plus the legacy single-file registry). We scan, drop dead entries (pid liveness), and POST:
 *  - with --session: only to that session (the per-run watcher passes the submitting session)
 *  - without: broadcast to every live channel (single-instance behavior unchanged).
 *  Pure plumbing — spends no agent tokens. */
export default class Notify extends Command {
  static description =
    "Push an async finish-event to conductor session channel(s) (registry: .ralph/agent/channels/). " +
    "kind=exp_done when a pueue experiment run finishes; kind=team_done when a subagent finishes.";

  static examples = [
    "experiment notify --kind exp_done --exp exp_spark_x --verdict killed",
    "experiment notify --kind exp_done --exp exp_x --session <claude-session-id>   # target one conductor",
    "experiment notify --kind team_done --agent a803b... --phase researcher",
  ];

  static flags = {
    kind: Flags.string({ required: true, options: ["exp_done", "team_done"], description: "event kind" }),
    exp: Flags.string({ description: "experiment id (exp_done)" }),
    verdict: Flags.string({ description: "supported|killed|provisional (exp_done)" }),
    agent: Flags.string({ description: "agent/subagent id (team_done)" }),
    phase: Flags.string({ description: "sparker|researcher|reviewer|analyst (team_done)" }),
    session: Flags.string({ description: "target session id (default: broadcast to all live channels)" }),
    content: Flags.string({ description: "override the channel message body" }),
  };

  private repoRoot(): string {
    try {
      return execSync("git rev-parse --show-toplevel", { encoding: "utf-8" }).trim();
    } catch {
      return process.cwd();
    }
  }

  private alive(pid: number): boolean {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  }

  /** All registered channels: per-session dir + legacy file, deduped by sessionId, dead pids pruned. */
  private channels(root: string): ChannelEntry[] {
    const out = new Map<string, ChannelEntry>();
    const dir = `${root}/.ralph/agent/channels`;
    if (existsSync(dir)) {
      for (const f of readdirSync(dir).filter((f) => f.endsWith(".json"))) {
        try {
          const e = JSON.parse(readFileSync(`${dir}/${f}`, "utf-8")) as ChannelEntry;
          e.file = `${dir}/${f}`;
          out.set(e.sessionId, e);
        } catch {
          /* skip unreadable */
        }
      }
    }
    const legacy = `${root}/.ralph/agent/conductor_channel.json`;
    if (existsSync(legacy)) {
      try {
        const e = JSON.parse(readFileSync(legacy, "utf-8")) as ChannelEntry;
        if (!out.has(e.sessionId)) out.set(e.sessionId, e);
      } catch {
        /* skip */
      }
    }
    const live: ChannelEntry[] = [];
    for (const e of out.values()) {
      if (e.pid && !this.alive(e.pid)) {
        if (e.file) rmSync(e.file, { force: true }); // prune stale registration
        continue;
      }
      live.push(e);
    }
    return live;
  }

  async run() {
    const { flags } = await this.parse(Notify);
    let targets = this.channels(this.repoRoot());
    if (flags.session) {
      const match = targets.filter((t) => t.sessionId === flags.session);
      // Watcher armed under a session that's gone (restart, compaction kill): fall back to
      // broadcast so the event isn't lost — any conductor can reconcile from the DB.
      targets = match.length > 0 ? match : targets;
    }
    if (targets.length === 0) {
      this.warn("no live conductor channels registered (.ralph/agent/channels/) — is a conductor session running?");
      return;
    }
    for (const t of targets) {
      const body = JSON.stringify({
        kind: flags.kind, exp: flags.exp, verdict: flags.verdict,
        agent_id: flags.agent, phase: flags.phase, content: flags.content,
        session: t.sessionId,
      });
      try {
        const r = await fetch(`http://127.0.0.1:${t.port}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
        });
        this.log(`notified ${t.sessionId.slice(0, 8)} :${t.port} — ${flags.kind} ${flags.exp ?? flags.agent ?? ""} -> ${await r.text()}`);
      } catch (e) {
        this.warn(`channel POST to :${t.port} (${t.sessionId.slice(0, 8)}) failed: ${(e as Error).message}`);
      }
    }
  }
}

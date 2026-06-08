import { Command, Flags } from "@oclif/core";
import { existsSync, readFileSync } from "node:fs";
import { execSync } from "node:child_process";

/** Producer: POST an async finish-event to the running conductor session's channel.
 *  The channel server records its {sessionId, port} in .ralph/agent/conductor_channel.json;
 *  we read that and POST. Called by pueue's `callback:` (kind=exp_done) and a SubagentStop
 *  hook (kind=team_done). Pure plumbing — spends no agent tokens. */
export default class Notify extends Command {
  static description =
    "Push an async finish-event to the conductor session's channel (read from .ralph/agent/conductor_channel.json). " +
    "kind=exp_done when a pueue experiment run finishes; kind=team_done when a subagent finishes.";

  static examples = [
    "experiment notify --kind exp_done --exp exp_spark_x --verdict killed",
    "experiment notify --kind team_done --agent a803b... --phase researcher",
  ];

  static flags = {
    kind: Flags.string({ required: true, options: ["exp_done", "team_done"], description: "event kind" }),
    exp: Flags.string({ description: "experiment id (exp_done)" }),
    verdict: Flags.string({ description: "supported|killed|provisional (exp_done)" }),
    agent: Flags.string({ description: "agent/subagent id (team_done)" }),
    phase: Flags.string({ description: "sparker|researcher|reviewer|analyst (team_done)" }),
    session: Flags.string({ description: "target session id (default: the one in the registry)" }),
    content: Flags.string({ description: "override the channel message body" }),
  };

  private repoRoot(): string {
    try {
      return execSync("git rev-parse --show-toplevel", { encoding: "utf-8" }).trim();
    } catch {
      return process.cwd();
    }
  }

  async run() {
    const { flags } = await this.parse(Notify);
    const reg = `${this.repoRoot()}/.ralph/agent/conductor_channel.json`;
    if (!existsSync(reg)) {
      this.warn("no conductor channel registry (.ralph/agent/conductor_channel.json) — is a conductor session running with the experiment channel?");
      return;
    }
    let port: number, sessionId: string;
    try {
      ({ port, sessionId } = JSON.parse(readFileSync(reg, "utf-8")));
    } catch (e) {
      this.warn(`channel registry unreadable: ${(e as Error).message}`);
      return;
    }
    const body = JSON.stringify({
      kind: flags.kind, exp: flags.exp, verdict: flags.verdict,
      agent_id: flags.agent, phase: flags.phase, content: flags.content,
      session: flags.session ?? sessionId,
    });
    try {
      const r = await fetch(`http://127.0.0.1:${port}`, { method: "POST", headers: { "content-type": "application/json" }, body });
      this.log(`notified channel :${port} — ${flags.kind} ${flags.exp ?? flags.agent ?? ""} -> ${await r.text()}`);
    } catch (e) {
      // Graceful degradation: the event is lost only if the session is down; the orchestrator
      // can also reconcile from the DB/pueue on its next turn.
      this.warn(`channel POST to :${port} failed (${(e as Error).message}); is the channel server up?`);
    }
  }
}

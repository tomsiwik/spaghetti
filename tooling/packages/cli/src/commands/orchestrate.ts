import { Command, Args, Flags } from "@oclif/core";
import { execSync } from "node:child_process";
import { runQueue } from "../lib/conductor.js";

export default class Orchestrate extends Command {
  static description =
    "Run the multi-model Conductor: drain a queue of experiment hunches. Each job runs the DAG " +
    "(spark: gemini explore / opus synthesize / gpt-5.5 systematize+codegen -> file -> run -> deterministic analyze -> complete). " +
    "Models are decoupled CLI jobs (minimal Opus); errors are handled so the queue drains to completion.";

  static args = {
    queue: Args.string({
      description: "queue.json (JSON list of jobs); omit to use the queue in conductor.yml",
      required: false,
    }),
  };

  static flags = {
    status: Flags.boolean({ description: "show in-flight jobs (pueue) + experiment states, then exit" }),
  };

  static examples = [
    "experiment orchestrate                # drain the queue in conductor.yml to completion",
    "experiment orchestrate jobs.json      # drain a custom queue of hunches",
    "experiment orchestrate --status       # show queue + experiment state",
  ];

  async run() {
    const { args, flags } = await this.parse(Orchestrate);

    if (flags.status) {
      try {
        execSync("pueue status 2>/dev/null", { stdio: "inherit" });
      } catch {
        /* pueue optional */
      }
      try {
        const root = execSync("git rev-parse --show-toplevel", { encoding: "utf-8" }).trim();
        execSync(`cd ${root} && experiment list -s active,open 2>/dev/null | head -20`, { stdio: "inherit" });
      } catch {
        /* ignore */
      }
      return;
    }

    // In-process: the conductor IS the work this command does — no benefit to spawning a second bun.
    // runQueue handles per-job errors itself and returns a summary; it throws only on bad config/queue.
    this.log("Starting Conductor (multi-model orchestration; Ctrl-C to stop)…\n");
    try {
      const s = runQueue(args.queue);
      this.log(`\nConductor finished: ${s.okCount}/${s.total} jobs clean; models used: ${s.modelsUsed.join(", ") || "fallback-only"}.`);
    } catch (e) {
      this.error(`Conductor failed: ${(e as Error).message}`); // non-zero exit on bad config/queue
    }
    // normal return => oclif exits 0
  }
}

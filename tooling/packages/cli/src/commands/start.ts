import { Command, Args, Flags } from "@oclif/core";
import { spawnSync, execSync } from "node:child_process";

const SEED = "Read .agents/conductor.md and follow it.";
const ONCE = "Read .agents/conductor.md and work ONE novel experiment to a verdict (spark -> research -> review -> analyst), then stop. NO mocks.";

export default class Start extends Command {
  static description =
    "Launch ONE Conductor — a persistent interactive Claude session driven by the `experiment` channel: " +
    "it sparks bet-rung experiments, dispatches the phases, and reacts to async finish-events pushed in " +
    "as <channel> messages — never polling, never self-quitting. All behavior lives in .agents/conductor.md. " +
    "Instances don't conflict: run `experiment start` in as many tabs (or machines) as you want — each " +
    "session gets its own channel port + registry entry, runs are addressed back to the submitting session, " +
    "and DB claims are atomic.";

  static args = {
    prompt: Args.string({ description: "override the seed prompt (default: 'Read .agents/conductor.md and follow it.')", required: false }),
  };

  static flags = {
    once: Flags.boolean({ description: "do exactly ONE experiment then stop (test), instead of the channel-driven loop" }),
    name: Flags.string({ description: "conductor name (worker-id prefix for claims; default 'conductor')" }),
  };

  static examples = [
    "experiment start          # launch a conductor (run again in another tab for a second one)",
    "experiment start --once   # one experiment to a verdict, then stop (test)",
    "experiment start --name c2  # name this instance's claims c2-<session>",
  ];

  private repoRoot(): string {
    try {
      return execSync("git rev-parse --show-toplevel", { encoding: "utf-8" }).trim();
    } catch {
      return process.cwd();
    }
  }

  async run() {
    const { args, flags } = await this.parse(Start);
    const root = this.repoRoot();
    const home = process.env.HOME ?? "";
    const prompt = args.prompt ?? (flags.once ? ONCE : SEED);

    const r = spawnSync(
      "claude",
      [
        "--model",
        "fable", // conductor runs Fable 5; hats have no frontmatter pin, so they inherit it
        "--dangerously-skip-permissions",
        "--dangerously-load-development-channels",
        "server:experiment", // the channel MCP server in .mcp.json (tooling/channel/experiment-channel.ts)
        "--teammate-mode",
        "in-process",
        "--settings",
        ".agents/conductor.settings.json",
        prompt,
      ],
      {
        cwd: root,
        stdio: "inherit",
        env: {
          ...process.env,
          CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: "1",
          ...(flags.name ? { CONDUCTOR_NAME: flags.name } : {}),
          PATH: `${home}/.local/bin:${home}/.vite-plus/bin:${home}/.bun/bin:/opt/homebrew/bin:${process.env.PATH ?? ""}`,
        },
      },
    );
    process.exit(r.status ?? 0);
  }
}

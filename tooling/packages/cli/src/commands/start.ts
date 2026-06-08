import { Command, Args, Flags } from "@oclif/core";
import { spawnSync, execSync } from "node:child_process";

const SEED = "Read PROMPT.md and follow it.";
const ONCE = "Read PROMPT.md and work ONE novel experiment to a verdict (spark -> research -> review -> analyst), then stop. NO mocks.";

export default class Start extends Command {
  static description =
    "Launch the Conductor — a persistent interactive Claude session driven by the `experiment` channel: " +
    "it sparks a novel experiment, dispatches the phases, and reacts to async finish-events (a pueue run " +
    "completing) pushed in as <channel> messages — never polling, never self-quitting. All behavior lives " +
    "in PROMPT.md; this command just executes the launch.";

  static args = {
    prompt: Args.string({ description: "override the seed prompt (default: 'Read PROMPT.md and follow it.')", required: false }),
  };

  static flags = {
    once: Flags.boolean({ description: "do exactly ONE experiment then stop (test), instead of the channel-driven loop" }),
  };

  static examples = [
    "experiment start          # launch the channel-driven conductor (PROMPT.md owns the behavior)",
    "experiment start --once   # one experiment to a verdict, then stop (test)",
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
        "--dangerously-skip-permissions",
        "--dangerously-load-development-channels",
        "server:experiment", // the channel MCP server in .mcp.json (tooling/channel/experiment-channel.ts)
        "--teammate-mode",
        "in-process",
        "--settings",
        ".claude/conductor.settings.json",
        prompt,
      ],
      {
        cwd: root,
        stdio: "inherit",
        env: {
          ...process.env,
          CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: "1",
          PATH: `${home}/.local/bin:${home}/.vite-plus/bin:${home}/.bun/bin:/opt/homebrew/bin:${process.env.PATH ?? ""}`,
        },
      },
    );
    process.exit(r.status ?? 0);
  }
}

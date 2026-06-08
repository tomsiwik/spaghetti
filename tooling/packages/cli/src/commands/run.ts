import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments } from "@experiment/db";
import { execSync, spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";

interface ExperimentConfig {
  parallel: number;
  group: string;
}

export default class Run extends Command {
  static description =
    "Run an experiment via pueue (process-isolated, serial queue, guaranteed cleanup)";

  static args = {
    id: Args.string({
      description: "Experiment ID (looks up experiment_dir) or path to run_experiment.py",
      required: false,
    }),
  };

  static strict = false;

  static flags = {
    "no-wait": Flags.boolean({
      description: "Submit and return immediately (default: wait for completion)",
      default: false,
    }),
    status: Flags.boolean({
      description: "Show pueue queue status for experiments group",
      default: false,
      exclusive: ["no-wait", "clean", "kill"],
    }),
    clean: Flags.boolean({
      description: "Remove completed tasks from the queue",
      default: false,
      exclusive: ["no-wait", "status", "kill"],
    }),
    kill: Flags.boolean({
      description: "Kill the currently running experiment",
      default: false,
      exclusive: ["no-wait", "status", "clean"],
    }),
    parallel: Flags.integer({
      description: "Override max parallel experiments (default: from .experimentrc)",
    }),
  };

  static examples = [
    "experiment run exp_spectral_surgery    # run by experiment ID",
    "experiment run experiments/models/foo/run_experiment.py  # run by path",
    "experiment run --status                # show queue",
    "experiment run --kill exp_foo          # kill running experiment",
    "experiment run --parallel 2 exp_foo    # allow 2 concurrent experiments",
  ];

  async run() {
    const { args, flags } = await this.parse(Run);
    const config = this.loadConfig();

    // CLI flag overrides config file
    if (flags.parallel !== undefined) {
      config.parallel = flags.parallel;
    }

    const group = config.group;

    // Ensure pueued is running and config is applied
    this.ensureDaemon(config);

    // Utility modes
    if (flags.status) {
      execSync(`pueue status --group ${group}`, { stdio: "inherit" });
      return;
    }
    if (flags.clean) {
      execSync(`pueue clean --group ${group}`, { stdio: "inherit" });
      this.log("Cleaned completed tasks.");
      return;
    }

    // Remaining modes require an ID
    if (!args.id) {
      this.error("Provide an experiment ID or script path. See: experiment run --help");
    }

    // Resolve script path: either a direct path or an experiment ID
    const script = await this.resolveScript(args.id);

    if (flags.kill) {
      // Find and kill running task for this script
      this.killExperiment(group, script);
      return;
    }

    // Submit to pueue
    const repoRoot = this.findRepoRoot();
    const label = this.labelFromScript(script);

    const taskId = execSync(
      `pueue add --group ${group} --working-directory ${repoRoot} --label "${label}" --print-task-id -- uv run python "${script}"`,
      { encoding: "utf-8" },
    ).trim();

    this.log(`Submitted task ${taskId}: ${label}`);

    if (flags["no-wait"]) {
      this.log(`Queue: pueue status --group ${group}`);
      this.log(`Logs:  pueue follow ${taskId}`);
      // Arm a detached per-run watcher: when THIS task completes, push an `exp_done`
      // finish-event to the conductor session's channel. Scoped to this run, ~0 tokens,
      // no global pueue.yml mutation. Harmless no-op if no conductor channel is running.
      const dir = script.replace(/\/[^/]+$/, "");
      const home = process.env.HOME ?? "";
      const watch =
        `export PATH="${home}/.local/bin:${home}/.vite-plus/bin:${home}/.bun/bin:/opt/homebrew/bin:$PATH"; ` +
        `pueue wait ${taskId} >/dev/null 2>&1; ` +
        `v=$(python3 -c "import json;print(json.load(open('${dir}/results.json')).get('verdict',''))" 2>/dev/null); ` +
        `experiment notify --kind exp_done --exp "${label}" --verdict "$v" >/dev/null 2>&1`;
      const w = spawn("bash", ["-c", watch], { detached: true, stdio: "ignore" });
      w.unref();
      this.log(`exp_done watcher armed for task ${taskId} (fires when the run completes).`);
      return;
    }

    // Wait mode: follow output, then report result
    this.log("Waiting for completion (Ctrl+C to detach, task keeps running)...");

    const follow = spawn("pueue", ["follow", taskId], { stdio: "inherit" });

    try {
      execSync(`pueue wait ${taskId}`, { stdio: "ignore" });
    } catch {
      // pueue wait returns non-zero if task failed
    }

    follow.kill();

    const exitCode = this.getTaskExitCode(taskId);

    // Auto-clean
    try {
      execSync(`pueue clean --group ${group}`, { stdio: "ignore" });
    } catch {}

    this.log(`Task ${taskId} finished (exit ${exitCode})`);

    if (exitCode !== 0) {
      this.exit(exitCode);
    }
  }

  private async resolveScript(idOrPath: string): Promise<string> {
    // If it looks like a file path and exists, use it directly
    if (idOrPath.endsWith(".py") && existsSync(idOrPath)) {
      return resolve(idOrPath);
    }

    // If it's a directory with run_experiment.py, use that
    const dirScript = join(idOrPath, "run_experiment.py");
    if (existsSync(dirScript)) {
      return resolve(dirScript);
    }

    // Look up experiment ID in the database
    const exp = await db
      .select()
      .from(experiments)
      .where(eq(experiments.id, idOrPath))
      .get();

    if (!exp) {
      this.error(
        `"${idOrPath}" is not a file path, directory, or known experiment ID`,
      );
    }

    if (!exp.experimentDir) {
      this.error(
        `Experiment "${idOrPath}" has no experiment_dir set. Set it with: experiment update ${idOrPath} --dir experiments/models/<name>/`,
      );
    }

    const script = resolve(exp.experimentDir, "run_experiment.py");
    if (!existsSync(script)) {
      this.error(`Script not found: ${script} (from experiment_dir: ${exp.experimentDir})`);
    }

    return script;
  }

  private loadConfig(): ExperimentConfig {
    const defaults: ExperimentConfig = { parallel: 1, group: "experiments" };
    const repoRoot = this.findRepoRoot();
    const rcPath = join(repoRoot, ".experimentrc");

    if (!existsSync(rcPath)) return defaults;

    try {
      const content = readFileSync(rcPath, "utf-8");
      const config = { ...defaults };

      for (const line of content.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;

        const [key, ...rest] = trimmed.split(":");
        const value = rest.join(":").trim();
        if (!value) continue;

        switch (key.trim()) {
          case "parallel": {
            const n = parseInt(value, 10);
            if (n > 0) config.parallel = n;
            break;
          }
          case "group":
            config.group = value;
            break;
        }
      }

      return config;
    } catch {
      return defaults;
    }
  }

  private ensureDaemon(config: ExperimentConfig) {
    const { group, parallel } = config;

    try {
      execSync("pueue status", { stdio: "ignore" });
    } catch {
      this.log("Starting pueue daemon...");
      execSync("pueued -d");
    }

    // Ensure group exists
    try {
      execSync(`pueue group add ${group}`, { stdio: "ignore" });
    } catch {} // already exists

    // Always sync parallelism from config
    execSync(`pueue parallel ${parallel} --group ${group}`, { stdio: "ignore" });
  }

  private killExperiment(group: string, script: string) {
    try {
      const statusJson = execSync(`pueue status --json --group ${group}`, {
        encoding: "utf-8",
      });
      const data = JSON.parse(statusJson);
      const tasks = data.tasks || {};

      for (const [id, task] of Object.entries(tasks) as [string, any][]) {
        if (
          task.command?.includes(script) &&
          typeof task.status === "string" &&
          task.status === "Running"
        ) {
          execSync(`pueue kill ${id}`);
          this.log(`Killed task ${id}: ${task.label || task.command}`);
          return;
        }
      }
      this.log("No running task found matching this experiment.");
    } catch (e: any) {
      this.error(`Failed to kill: ${e.message}`);
    }
  }

  private getTaskExitCode(taskId: string): number {
    try {
      const statusJson = execSync("pueue status --json", {
        encoding: "utf-8",
      });
      const data = JSON.parse(statusJson);
      const task = data.tasks?.[taskId];
      if (!task) return 1;

      const status = task.status;
      if (typeof status === "object" && "Done" in status) {
        const result = status.Done?.result;
        if (result === "Success") return 0;
        if (typeof result === "object" && "Failed" in result)
          return result.Failed;
      }
      return 1;
    } catch {
      return 1;
    }
  }

  private labelFromScript(script: string): string {
    // Extract experiment name from path: experiments/models/<name>/run_experiment.py → <name>
    const parts = script.split("/");
    const idx = parts.indexOf("run_experiment.py");
    if (idx > 0) return parts[idx - 1];
    return parts[parts.length - 1];
  }

  private findRepoRoot(): string {
    try {
      return execSync("git rev-parse --show-toplevel", {
        encoding: "utf-8",
      }).trim();
    } catch {
      return process.cwd();
    }
  }
}

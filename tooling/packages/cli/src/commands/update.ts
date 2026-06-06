import { Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments } from "@experiment/db";
import { ExperimentCommand, experimentIdArg } from "../lib/base-command.js";

export default class Update extends ExperimentCommand {
  static description = "Update an experiment's fields";
  static args = experimentIdArg;

  static flags = {
    status: Flags.string({ description: "New status: open, active, proven, supported, killed" }),
    priority: Flags.integer({ description: "New priority" }),
    platform: Flags.string({ description: "Platform: local, local-apple, runpod-flash" }),
    dir: Flags.string({ description: "Experiment directory path" }),
    notes: Flags.string({ description: "Replace notes" }),
    title: Flags.string({ description: "Replace title" }),
  };

  async run() {
    const { args, flags } = await this.parse(Update);
    await this.requireExperiment(args.id);

    const updates: Record<string, unknown> = { updatedAt: this.today };

    if (flags.status) {
      updates.status = flags.status;
      if (["proven", "supported", "killed"].includes(flags.status)) {
        updates.claimedBy = null;
        updates.claimedAt = null;
      }
    }
    if (flags.priority !== undefined) updates.priority = flags.priority;
    if (flags.platform) updates.platform = flags.platform;
    if (flags.dir) updates.experimentDir = flags.dir;
    if (flags.notes) updates.notes = flags.notes;
    if (flags.title) updates.title = flags.title;

    await db.update(experiments).set(updates).where(eq(experiments.id, args.id)).run();

    this.log(`Updated ${args.id}: ${Object.keys(updates).filter((k) => k !== "updatedAt").join(", ")}`);
  }
}

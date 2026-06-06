import { Flags } from "@oclif/core";
import { db, successCriteria } from "@experiment/db";
import { ExperimentCommand, experimentIdArg } from "../lib/base-command.js";

export default class SuccessAdd extends ExperimentCommand {
  static description = "Add a success criterion to an experiment";
  static args = experimentIdArg;

  static flags = {
    condition: Flags.string({ description: "Success condition", required: true }),
    unlocks: Flags.string({ description: "What this success unlocks", required: true }),
    reason: Flags.string({ description: "Why this criterion matters", default: "pre-registered" }),
    "max-followup": Flags.integer({ description: "Max follow-up experiments", default: 1 }),
  };

  async run() {
    const { args, flags } = await this.parse(SuccessAdd);
    await this.requireExperiment(args.id);

    const result = await db.insert(successCriteria).values({
      experimentId: args.id,
      condition: flags.condition,
      unlocks: flags.unlocks,
      maxFollowup: flags["max-followup"],
      reason: flags.reason,
    }).run();

    this.log(`Added success criterion #${result.lastInsertRowid} to ${args.id}`);
  }
}

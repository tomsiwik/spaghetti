import { Flags } from "@oclif/core";
import { db, killCriteria } from "@experiment/db";
import { ExperimentCommand, experimentIdArg } from "../lib/base-command.js";

export default class KillAdd extends ExperimentCommand {
  static description = "Add a kill criterion to an experiment";
  static args = experimentIdArg;

  static flags = {
    text: Flags.string({ description: "Kill criterion text", required: true }),
    reason: Flags.string({ description: "Why this criterion exists", default: "pre-registered" }),
  };

  async run() {
    const { args, flags } = await this.parse(KillAdd);
    await this.requireExperiment(args.id);

    const result = await db.insert(killCriteria).values({
      experimentId: args.id,
      text: flags.text,
      reason: flags.reason,
      result: "untested",
    }).run();

    this.log(`Added kill criterion #${result.lastInsertRowid} to ${args.id}`);
  }
}

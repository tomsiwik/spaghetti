import { Flags } from "@oclif/core";
import { db, evidence } from "@experiment/db";
import { ExperimentCommand, experimentIdArg } from "../lib/base-command.js";

export default class Evidence extends ExperimentCommand {
  static description = "Add evidence to an experiment";
  static args = experimentIdArg;

  static flags = {
    claim: Flags.string({ description: "Evidence claim", required: true }),
    source: Flags.string({ description: "Source (file path, URL, etc.)", required: true }),
    verdict: Flags.string({ description: "Verdict: pass, fail, inconclusive" }),
    date: Flags.string({ description: "Date (ISO format, defaults to today)" }),
  };

  async run() {
    const { args, flags } = await this.parse(Evidence);
    await this.requireExperiment(args.id);

    await db.insert(evidence)
      .values({
        experimentId: args.id,
        date: flags.date ?? this.today,
        claim: flags.claim,
        source: flags.source,
        verdict: (flags.verdict as any) ?? null,
      })
      .run();

    this.log(`Added evidence to ${args.id}: ${flags.claim.slice(0, 60)}...`);
  }
}

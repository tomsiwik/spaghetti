import { Flags } from "@oclif/core";
import { eq, and } from "drizzle-orm";
import { db, killCriteria } from "@experiment/db";
import { ExperimentCommand, experimentIdArg } from "../lib/base-command.js";

export default class KillUpdate extends ExperimentCommand {
  static description = "Update a kill criterion result (pass/fail/inconclusive)";
  static args = experimentIdArg;

  static flags = {
    criterion: Flags.integer({ description: "Kill criterion ID (from get output)", required: true }),
    result: Flags.string({
      description: "Result: pass, fail, inconclusive",
      required: true,
      options: ["pass", "fail", "inconclusive"],
    }),
  };

  async run() {
    const { args, flags } = await this.parse(KillUpdate);

    const kc = await db.select().from(killCriteria)
      .where(and(eq(killCriteria.id, flags.criterion), eq(killCriteria.experimentId, args.id)))
      .get();
    if (!kc) this.error(`Kill criterion #${flags.criterion} not found for experiment "${args.id}"`);

    await db.update(killCriteria)
      .set({ result: flags.result as any })
      .where(eq(killCriteria.id, flags.criterion))
      .run();

    this.log(`Updated kill criterion #${flags.criterion}: ${kc.result} -> ${flags.result}`);
  }
}

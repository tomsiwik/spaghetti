import { Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, experimentDependencies } from "@experiment/db";
import { ExperimentCommand, experimentIdArg } from "../lib/base-command.js";

export default class DepAdd extends ExperimentCommand {
  static description = "Add a dependency between experiments";
  static args = experimentIdArg;

  static flags = {
    on: Flags.string({ description: "Experiment ID it depends on", required: true }),
  };

  async run() {
    const { args, flags } = await this.parse(DepAdd);
    await this.requireExperiment(args.id);

    const dep = await db.select().from(experiments).where(eq(experiments.id, flags.on)).get();
    if (!dep) this.error(`Dependency target "${flags.on}" not found`);

    await db.insert(experimentDependencies)
      .values({ experimentId: args.id, dependsOnId: flags.on })
      .onConflictDoNothing()
      .run();

    this.log(`${args.id} now depends on ${flags.on}`);
  }
}

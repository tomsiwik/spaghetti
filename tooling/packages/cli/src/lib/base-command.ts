import { Command, Args } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments } from "@experiment/db";

/** Reusable experiment ID arg — oclif needs static args on each command class. */
export const experimentIdArg = {
  id: Args.string({ description: "Experiment ID", required: true }),
};

/**
 * Base command with shared helpers for experiment commands.
 * Subclasses must still define their own `static args = experimentIdArg`.
 */
export abstract class ExperimentCommand extends Command {
  protected get today() {
    return new Date().toISOString().slice(0, 10);
  }

  protected get now() {
    return new Date().toISOString();
  }

  protected async requireExperiment(id: string) {
    const exp = await db.select().from(experiments).where(eq(experiments.id, id)).get();
    if (!exp) this.error(`Experiment "${id}" not found`);
    return exp;
  }
}

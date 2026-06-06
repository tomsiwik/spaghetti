import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, tags, experimentTags } from "@experiment/db";

export default class Tag extends Command {
  static description = "Add tags to an experiment";

  static strict = false; // allow positional tag names after id

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    add: Flags.string({ description: "Tag name to add", multiple: true }),
  };

  async run() {
    const { args, flags, argv } = await this.parse(Tag);

    const exp = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) this.error(`Experiment "${args.id}" not found`);

    // Merge --add flags with positional args (everything after the id)
    const tagNames = [...(flags.add ?? []), ...(argv as string[]).slice(1)];
    if (tagNames.length === 0) {
      this.error("No tags provided. Usage: experiment tag <id> routing mlx  OR  experiment tag <id> --add routing --add mlx");
    }

    const added: string[] = [];
    for (const name of tagNames) {
      const clean = name.trim().toLowerCase();
      await db.insert(tags).values({ name: clean }).onConflictDoNothing().run();
      const tag = await db.select().from(tags).where(eq(tags.name, clean)).get();
      if (tag) {
        await db.insert(experimentTags)
          .values({ experimentId: args.id, tagId: tag.id })
          .onConflictDoNothing()
          .run();
        added.push(clean);
      }
    }

    this.log(`Tagged ${args.id}: ${added.join(", ")}`);
  }
}

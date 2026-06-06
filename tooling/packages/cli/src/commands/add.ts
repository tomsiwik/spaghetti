import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, experimentReferences, references, tags, experimentTags, killCriteria, experimentDependencies } from "@experiment/db";

export default class Add extends Command {
  static description = "Add a new experiment (one-shot: tags, kill criteria, deps inline)";

  static args = {
    id: Args.string({ description: "Experiment ID (e.g. exp_my_experiment)", required: true }),
  };

  static flags = {
    title: Flags.string({ description: "Experiment title", required: true }),
    scale: Flags.string({ description: "Scale: micro or macro", default: "micro" }),
    priority: Flags.integer({ description: "Priority (1=highest)", default: 3 }),
    platform: Flags.string({ description: "Platform: local, local-apple, runpod-flash" }),
    dir: Flags.string({ description: "Experiment directory path" }),
    notes: Flags.string({ description: "Notes" }),
    "grounded-by": Flags.string({ description: "arXiv ID of grounding reference" }),
    tag: Flags.string({ description: "Tag (repeatable)", multiple: true }),
    kill: Flags.string({ description: "Kill criterion text (repeatable)", multiple: true }),
    dep: Flags.string({ description: "Depends-on experiment ID (repeatable)", multiple: true }),
  };

  async run() {
    const { args, flags } = await this.parse(Add);

    const existing = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (existing) {
      this.error(`Experiment "${args.id}" already exists`);
    }

    const now = new Date().toISOString().slice(0, 10);

    await db.insert(experiments)
      .values({
        id: args.id,
        title: flags.title,
        status: "open",
        scale: flags.scale as any,
        priority: flags.priority,
        experimentDir: flags.dir ?? null,
        platform: (flags.platform as any) ?? null,
        notes: flags.notes ?? null,
        createdAt: now,
        updatedAt: now,
      })
      .run();

    // Link grounding reference if provided
    if (flags["grounded-by"]) {
      const ref = await db
        .select()
        .from(references)
        .where(eq(references.arxivId, flags["grounded-by"]))
        .get();
      if (ref) {
        await db.insert(experimentReferences)
          .values({ experimentId: args.id, referenceId: ref.id })
          .onConflictDoNothing()
          .run();
        this.log(`  Linked to reference: ${ref.title}`);
      } else {
        this.warn(`Reference with arxiv_id "${flags["grounded-by"]}" not found. Add it with: experiment ref-add`);
      }
    }

    // Add tags inline
    if (flags.tag) {
      for (const name of flags.tag) {
        const clean = name.trim().toLowerCase();
        await db.insert(tags).values({ name: clean }).onConflictDoNothing().run();
        const tag = await db.select().from(tags).where(eq(tags.name, clean)).get();
        if (tag) {
          await db.insert(experimentTags)
            .values({ experimentId: args.id, tagId: tag.id })
            .onConflictDoNothing()
            .run();
        }
      }
      this.log(`  Tags: ${flags.tag.join(", ")}`);
    }

    // Add kill criteria inline
    if (flags.kill) {
      for (const text of flags.kill) {
        const result = await db.insert(killCriteria).values({
          experimentId: args.id,
          text,
          reason: "pre-registered",
          result: "untested",
        }).run();
        this.log(`  Kill #${result.lastInsertRowid}: ${text.slice(0, 70)}`);
      }
    }

    // Add dependencies inline
    if (flags.dep) {
      for (const depId of flags.dep) {
        const dep = await db.select().from(experiments).where(eq(experiments.id, depId)).get();
        if (dep) {
          await db.insert(experimentDependencies)
            .values({ experimentId: args.id, dependsOnId: depId })
            .onConflictDoNothing()
            .run();
          this.log(`  Depends on: ${depId}`);
        } else {
          this.warn(`Dependency "${depId}" not found — skipped`);
        }
      }
    }

    this.log(`Created experiment: ${args.id}`);
  }
}

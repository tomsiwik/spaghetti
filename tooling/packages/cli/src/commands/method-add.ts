import { Command, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, methods, tags, methodTags } from "@experiment/db";

export default class MethodAdd extends Command {
  static description = "Add a method to the method bank";

  static flags = {
    name: Flags.string({ description: "Method name", required: true }),
    description: Flags.string({ description: "What the method does", required: true }),
    solves: Flags.string({ description: "What problem it addresses", required: true }),
    "proven-in": Flags.string({ description: "Where it was proven to work (paper, system, our experiment)" }),
    "use-when": Flags.string({ description: "Conditions under which to reach for this method" }),
    "not-now-because": Flags.string({ description: "Why we are not using it right now" }),
    source: Flags.string({ description: "Reference: arxiv ID, repo URL, paper" }),
    status: Flags.string({
      description: "Status",
      default: "parked",
      options: ["parked", "exploring", "applied", "rejected"],
    }),
    tag: Flags.string({ description: "Tag (repeatable)", multiple: true }),
  };

  async run() {
    const { flags } = await this.parse(MethodAdd);
    const now = new Date().toISOString().slice(0, 10);

    const result = await db.insert(methods).values({
      name: flags.name,
      description: flags.description,
      solves: flags.solves,
      provenIn: flags["proven-in"] ?? null,
      useWhen: flags["use-when"] ?? null,
      notNowBecause: flags["not-now-because"] ?? null,
      source: flags.source ?? null,
      status: flags.status as any,
      createdAt: now,
      updatedAt: now,
    }).run();

    const methodId = Number(result.lastInsertRowid);

    if (flags.tag) {
      for (const name of flags.tag) {
        const clean = name.trim().toLowerCase();
        await db.insert(tags).values({ name: clean }).onConflictDoNothing().run();
        const tag = await db.select().from(tags).where(eq(tags.name, clean)).get();
        if (tag) {
          await db.insert(methodTags)
            .values({ methodId, tagId: tag.id })
            .onConflictDoNothing()
            .run();
        }
      }
    }

    this.log(`Added method #${methodId}: ${flags.name}${flags.tag ? ` [${flags.tag.join(", ")}]` : ""}`);
  }
}

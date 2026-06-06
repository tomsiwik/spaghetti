import { Command, Flags } from "@oclif/core";
import { eq, sql } from "drizzle-orm";
import { db, references, referenceTags, tags, experimentReferences } from "@experiment/db";

export default class Refs extends Command {
  static description = "List references";

  static flags = {
    unused: Flags.boolean({ description: "Show only references not linked to any experiment", default: false }),
    tag: Flags.string({ description: "Filter by tag", char: "t" }),
  };

  async run() {
    const { flags } = await this.parse(Refs);

    let rows = await db.select().from(references).all();

    if (flags.unused) {
      const linkedIds = new Set(
        (await db
          .select({ id: experimentReferences.referenceId })
          .from(experimentReferences)
          .all())
          .map((r) => r.id),
      );
      rows = rows.filter((r) => !linkedIds.has(r.id));
    }

    if (flags.tag) {
      const tag = await db.select().from(tags).where(eq(tags.name, flags.tag.toLowerCase())).get();
      if (!tag) {
        this.log(`No references with tag "${flags.tag}"`);
        return;
      }
      const taggedIds = new Set(
        (await db
          .select({ id: referenceTags.referenceId })
          .from(referenceTags)
          .where(eq(referenceTags.tagId, tag.id))
          .all())
          .map((r) => r.id),
      );
      rows = rows.filter((r) => taggedIds.has(r.id));
    }

    if (rows.length === 0) {
      this.log("No references found.");
      return;
    }

    this.log(`\n${"ID".padEnd(5)} ${"ARXIV".padEnd(14)} ${"TITLE".padEnd(35)} RELEVANCE`);
    this.log("-".repeat(110));
    for (const r of rows) {
      this.log(
        `${String(r.id).padEnd(5)} ${(r.arxivId ?? "—").padEnd(14)} ${r.title.slice(0, 33).padEnd(35)} ${r.relevance.slice(0, 55)}`,
      );
    }
    this.log(`\n${rows.length} references`);
  }
}

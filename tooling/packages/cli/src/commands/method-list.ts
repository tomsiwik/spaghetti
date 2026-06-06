import { Command, Flags } from "@oclif/core";
import { eq, sql } from "drizzle-orm";
import { db, methods, methodTags, tags } from "@experiment/db";

const STATUS_ICON: Record<string, string> = {
  parked: "⏸",
  exploring: "🔍",
  applied: "✓",
  rejected: "✗",
};

export default class MethodList extends Command {
  static description = "List methods in the method bank";

  static flags = {
    status: Flags.string({
      description: "Filter by status",
      options: ["parked", "exploring", "applied", "rejected"],
    }),
    tag: Flags.string({ description: "Filter by tag" }),
    problem: Flags.string({ description: "FTS search in 'solves' and 'use_when' fields" }),
  };

  async run() {
    const { flags } = await this.parse(MethodList);

    let rows: any[];

    if (flags.problem) {
      // FTS search focused on problem fields
      const raw = await db.all(
        sql`SELECT m.* FROM methods m JOIN methods_fts f ON m.rowid = f.rowid
            WHERE methods_fts MATCH ${flags.problem} ORDER BY rank`
      );
      rows = raw;
    } else if (flags.tag) {
      const tag = await db.select().from(tags).where(eq(tags.name, flags.tag.toLowerCase())).get();
      if (!tag) { this.log("No methods with that tag."); return; }
      const junctions = await db.select().from(methodTags).where(eq(methodTags.tagId, tag.id)).all();
      const ids = junctions.map((j) => j.methodId);
      if (ids.length === 0) { this.log("No methods with that tag."); return; }
      rows = await db.select().from(methods)
        .where(sql`${methods.id} IN (${sql.join(ids.map(id => sql`${id}`), sql`, `)})`)
        .all();
    } else {
      rows = await db.select().from(methods).all();
    }

    if (flags.status) {
      rows = rows.filter((r: any) => r.status === flags.status);
    }

    if (rows.length === 0) {
      this.log("No methods found.");
      return;
    }

    this.log(`  ${rows.length} method(s):\n`);
    for (const m of rows as any[]) {
      const icon = STATUS_ICON[m.status] ?? "?";
      this.log(`  ${icon} #${m.id} [${m.status}] ${m.name}`);
      this.log(`    Solves: ${m.solves}`);
      if (m.use_when) this.log(`    Use when: ${m.use_when}`);
      if (m.not_now_because) this.log(`    Not now: ${m.not_now_because}`);
      this.log("");
    }
  }
}

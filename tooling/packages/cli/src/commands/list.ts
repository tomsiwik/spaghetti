import { Command, Flags } from "@oclif/core";
import { eq, inArray, sql } from "drizzle-orm";
import { db, experiments, experimentTags, tags, experimentDependencies } from "@experiment/db";

export default class List extends Command {
  static description = "List experiments with optional filters";

  static flags = {
    status: Flags.string({ description: "Filter by status (comma-separated)", char: "s" }),
    tag: Flags.string({ description: "Filter by tag", char: "t" }),
    blocking: Flags.boolean({ description: "Show only experiments that block others", default: false }),
  };

  async run() {
    const { flags } = await this.parse(List);

    let query = db.select().from(experiments);

    // Get all experiments first, then filter in JS (simpler for POC)
    let rows = await query.all();

    // Filter by status
    if (flags.status) {
      const statuses = flags.status.split(",").map((s) => s.trim());
      rows = rows.filter((r) => statuses.includes(r.status));
    }

    // Filter by tag
    if (flags.tag) {
      const tag = await db.select().from(tags).where(eq(tags.name, flags.tag.toLowerCase())).get();
      if (!tag) {
        this.log(`No experiments with tag "${flags.tag}"`);
        return;
      }
      const taggedIds = (await db
        .select({ id: experimentTags.experimentId })
        .from(experimentTags)
        .where(eq(experimentTags.tagId, tag.id))
        .all())
        .map((r) => r.id);
      rows = rows.filter((r) => taggedIds.includes(r.id));
    }

    // Filter blocking
    if (flags.blocking) {
      const blockers = (await db
        .select({ id: experimentDependencies.dependsOnId })
        .from(experimentDependencies)
        .all())
        .map((r) => r.id);
      const blockerSet = new Set(blockers);
      rows = rows.filter((r) => blockerSet.has(r.id));
    }

    // Sort by priority then status
    const statusOrder: Record<string, number> = { active: 0, open: 1, supported: 2, proven: 3, killed: 4 };
    rows.sort((a, b) => (statusOrder[a.status] ?? 5) - (statusOrder[b.status] ?? 5) || a.priority - b.priority);

    if (rows.length === 0) {
      this.log("No experiments found.");
      return;
    }

    this.log(`\n${"ID".padEnd(45)} ${"STATUS".padEnd(10)} ${"P".padEnd(3)} ${"SCALE".padEnd(6)} ${"WORKER".padEnd(16)} TITLE`);
    this.log("-".repeat(136));
    for (const r of rows) {
      const worker = r.claimedBy ? r.claimedBy.slice(0, 14) : "";
      this.log(
        `${r.id.padEnd(45)} ${r.status.padEnd(10)} ${String(r.priority).padEnd(3)} ${r.scale.padEnd(6)} ${worker.padEnd(16)} ${r.title.slice(0, 55)}`,
      );
    }
    this.log(`\n${rows.length} experiments`);
  }
}

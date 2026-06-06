import { Command, Args } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, methods, methodTags, tags } from "@experiment/db";

export default class MethodGet extends Command {
  static description = "Get full details of a method";

  static args = {
    id: Args.integer({ description: "Method ID", required: true }),
  };

  async run() {
    const { args } = await this.parse(MethodGet);

    const m = await db.select().from(methods).where(eq(methods.id, args.id)).get();
    if (!m) this.error(`Method #${args.id} not found`);

    const junctions = await db.select().from(methodTags).where(eq(methodTags.methodId, m.id)).all();
    const tagNames: string[] = [];
    for (const j of junctions) {
      const t = await db.select().from(tags).where(eq(tags.id, j.tagId)).get();
      if (t) tagNames.push(t.name);
    }

    this.log(`  Method #${m.id}: ${m.name}`);
    this.log(`  ${"─".repeat(50)}`);
    this.log(`  Status:      ${m.status}`);
    if (tagNames.length) this.log(`  Tags:        ${tagNames.join(", ")}`);
    this.log(`\n  Description: ${m.description}`);
    this.log(`  Solves:      ${m.solves}`);
    if (m.provenIn) this.log(`  Proven in:   ${m.provenIn}`);
    if (m.useWhen) this.log(`  Use when:    ${m.useWhen}`);
    if (m.notNowBecause) this.log(`  Not now:     ${m.notNowBecause}`);
    if (m.source) this.log(`  Source:      ${m.source}`);
    this.log(`\n  Created: ${m.createdAt}  Updated: ${m.updatedAt}`);
  }
}

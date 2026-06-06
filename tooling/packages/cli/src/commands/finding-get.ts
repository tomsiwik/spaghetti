import { Command, Args } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, findings } from "@experiment/db";

export default class FindingGet extends Command {
  static description = "Get full details of a finding";

  static args = {
    id: Args.integer({ description: "Finding ID", required: true }),
  };

  async run() {
    const { args } = await this.parse(FindingGet);

    const f = await db.select().from(findings).where(eq(findings.id, args.id)).get();
    if (!f) {
      this.error(`Finding #${args.id} not found`);
    }

    this.log(`\n  Finding #${f.id}: ${f.title}`);
    this.log(`  ${"─".repeat(60)}`);
    this.log(`  Status:     ${f.status}`);
    this.log(`  Scale:      ${f.scale ?? "unset"}`);
    this.log(`  Date:       ${f.date}`);
    if (f.experimentId) this.log(`  Experiment: ${f.experimentId}`);
    this.log(`\n  Result:\n    ${f.result}`);
    if (f.caveat) this.log(`\n  Caveats:\n    ${f.caveat}`);
    if (f.failureMode) this.log(`\n  Failure Mode:\n    ${f.failureMode}`);
    if (f.impossibilityStructure) this.log(`\n  Impossibility Structure:\n    ${f.impossibilityStructure}`);
    this.log(`\n  Created: ${f.createdAt}`);
    this.log(`  Updated: ${f.updatedAt}`);
  }
}

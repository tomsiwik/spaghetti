import { Command, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, findings } from "@experiment/db";

export default class FindingList extends Command {
  static description = "List research findings";

  static flags = {
    status: Flags.string({
      description: "Filter by status",
      options: ["conclusive", "supported", "killed", "provisional"],
    }),
    scale: Flags.string({
      description: "Filter by scale",
      options: ["micro", "macro"],
    }),
  };

  async run() {
    const { flags } = await this.parse(FindingList);

    let query = db.select().from(findings);

    if (flags.status) {
      query = query.where(eq(findings.status, flags.status as any)) as any;
    }
    if (flags.scale) {
      query = query.where(eq(findings.scale, flags.scale as any)) as any;
    }

    const rows = await query.all();

    if (rows.length === 0) {
      this.log("No findings found.");
      return;
    }

    const statusIcon: Record<string, string> = {
      conclusive: "✓",
      supported: "~",
      killed: "✗",
      provisional: "?",
    };

    this.log(`\n  ${rows.length} findings:\n`);

    for (const f of rows) {
      const icon = statusIcon[f.status] ?? " ";
      const exp = f.experimentId ? ` [${f.experimentId}]` : "";
      const fm = f.failureMode ? ` | FM: ${f.failureMode.slice(0, 40)}` : "";
      this.log(`  ${icon} #${f.id} [${f.status}] ${f.title}${exp}${fm}`);
      this.log(`    ${f.result.slice(0, 100)}${f.result.length > 100 ? "..." : ""}`);
    }
  }
}

import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, findings } from "@experiment/db";

export default class FindingUpdate extends Command {
  static description = "Update a finding";

  static args = {
    id: Args.integer({ description: "Finding ID", required: true }),
  };

  static flags = {
    title: Flags.string({ description: "New title" }),
    status: Flags.string({
      description: "New status",
      options: ["conclusive", "supported", "killed", "provisional"],
    }),
    result: Flags.string({ description: "New result summary" }),
    caveat: Flags.string({ description: "New caveats" }),
    experiment: Flags.string({ description: "Link to experiment ID" }),
    "failure-mode": Flags.string({ description: "What failure mode does this address?" }),
    "impossibility-structure": Flags.string({
      description: "What mathematical structure makes the failure impossible?",
    }),
  };

  async run() {
    const { args, flags } = await this.parse(FindingUpdate);

    const existing = await db.select().from(findings).where(eq(findings.id, args.id)).get();
    if (!existing) {
      this.error(`Finding #${args.id} not found`);
    }

    const updates: Record<string, any> = { updatedAt: new Date().toISOString() };
    if (flags.title) updates.title = flags.title;
    if (flags.status) updates.status = flags.status;
    if (flags.result) updates.result = flags.result;
    if (flags.caveat) updates.caveat = flags.caveat;
    if (flags.experiment) updates.experimentId = flags.experiment;
    if (flags["failure-mode"]) updates.failureMode = flags["failure-mode"];
    if (flags["impossibility-structure"]) updates.impossibilityStructure = flags["impossibility-structure"];

    await db.update(findings).set(updates).where(eq(findings.id, args.id)).run();

    this.log(`Updated finding #${args.id}`);
  }
}

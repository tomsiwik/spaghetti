import { Command, Flags } from "@oclif/core";
import { db, findings } from "@experiment/db";

export default class FindingAdd extends Command {
  static description = "Add a research finding";

  static flags = {
    title: Flags.string({ description: "Finding title", required: true }),
    status: Flags.string({
      description: "Status: conclusive, supported, killed, provisional",
      required: true,
      options: ["conclusive", "supported", "killed", "provisional"],
    }),
    result: Flags.string({ description: "Result summary", required: true }),
    caveat: Flags.string({ description: "Caveats and limitations" }),
    experiment: Flags.string({ description: "Linked experiment ID" }),
    scale: Flags.string({ description: "Scale: micro or macro", options: ["micro", "macro"] }),
    "failure-mode": Flags.string({ description: "What failure mode does this address?" }),
    "impossibility-structure": Flags.string({
      description: "What mathematical structure makes the failure impossible?",
    }),
    date: Flags.string({ description: "Date (ISO format, defaults to today)" }),
  };

  async run() {
    const { flags } = await this.parse(FindingAdd);
    const now = new Date().toISOString();
    const date = flags.date ?? now.slice(0, 10);

    const result = await db
      .insert(findings)
      .values({
        title: flags.title,
        status: flags.status as any,
        result: flags.result,
        caveat: flags.caveat ?? null,
        experimentId: flags.experiment ?? null,
        scale: (flags.scale as any) ?? null,
        failureMode: flags["failure-mode"] ?? null,
        impossibilityStructure: flags["impossibility-structure"] ?? null,
        date,
        createdAt: now,
        updatedAt: now,
      })
      .run();

    const id = result.lastInsertRowid;
    this.log(`Finding #${id} added: ${flags.title}`);
    this.log(`  Status: ${flags.status}`);
    if (flags.experiment) this.log(`  Linked to: ${flags.experiment}`);
    if (flags["failure-mode"]) this.log(`  Failure mode: ${flags["failure-mode"]}`);
  }
}

import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, methods } from "@experiment/db";

export default class MethodUpdate extends Command {
  static description = "Update a method in the method bank";

  static args = {
    id: Args.integer({ description: "Method ID", required: true }),
  };

  static flags = {
    name: Flags.string({ description: "New name" }),
    description: Flags.string({ description: "New description" }),
    solves: Flags.string({ description: "New problem statement" }),
    "proven-in": Flags.string({ description: "Where proven" }),
    "use-when": Flags.string({ description: "When to use" }),
    "not-now-because": Flags.string({ description: "Why not now" }),
    source: Flags.string({ description: "Reference" }),
    status: Flags.string({
      description: "New status",
      options: ["parked", "exploring", "applied", "rejected"],
    }),
  };

  async run() {
    const { args, flags } = await this.parse(MethodUpdate);

    const m = await db.select().from(methods).where(eq(methods.id, args.id)).get();
    if (!m) this.error(`Method #${args.id} not found`);

    const updates: Record<string, unknown> = {
      updatedAt: new Date().toISOString().slice(0, 10),
    };

    if (flags.name) updates.name = flags.name;
    if (flags.description) updates.description = flags.description;
    if (flags.solves) updates.solves = flags.solves;
    if (flags["proven-in"]) updates.provenIn = flags["proven-in"];
    if (flags["use-when"]) updates.useWhen = flags["use-when"];
    if (flags["not-now-because"]) updates.notNowBecause = flags["not-now-because"];
    if (flags.source) updates.source = flags.source;
    if (flags.status) updates.status = flags.status;

    await db.update(methods).set(updates).where(eq(methods.id, args.id)).run();

    this.log(`Updated method #${args.id}: ${Object.keys(updates).filter(k => k !== "updatedAt").join(", ")}`);
  }
}

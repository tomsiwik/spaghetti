import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, killCriteria, evidence } from "@experiment/db";

export default class Complete extends Command {
  static description = "Complete an experiment in one shot: set status, update kill criteria, add evidence, set dir";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    status: Flags.string({
      description: "Final status: supported, proven, or killed",
      required: true,
      options: ["supported", "proven", "killed"],
    }),
    dir: Flags.string({ description: "Experiment directory path" }),
    k: Flags.string({
      description: "Kill criterion result as id:pass|fail|inconclusive (repeatable)",
      multiple: true,
    }),
    evidence: Flags.string({ description: "Evidence claim text" }),
    source: Flags.string({ description: "Evidence source (file path or URL)", default: "results.json" }),
  };

  async run() {
    const { args, flags } = await this.parse(Complete);
    const now = new Date().toISOString();
    const today = now.slice(0, 10);

    const exp = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) this.error(`Experiment "${args.id}" not found`);

    // 1. Update kill criteria
    if (flags.k) {
      for (const kv of flags.k) {
        const [idStr, result] = kv.split(":");
        const kcId = parseInt(idStr, 10);
        if (!result || !["pass", "fail", "inconclusive"].includes(result)) {
          this.warn(`Invalid kill criterion format "${kv}" — use id:pass|fail|inconclusive`);
          continue;
        }
        await db.update(killCriteria).set({ result: result as any })
          .where(eq(killCriteria.id, kcId)).run();
        this.log(`  K#${kcId}: ${result}`);
      }
    }

    // 2. Add evidence
    if (flags.evidence) {
      await db.insert(evidence).values({
        experimentId: args.id,
        date: today,
        claim: flags.evidence,
        source: flags.source,
        verdict: flags.status === "killed" ? "fail" : "pass",
      }).run();
      this.log(`  Evidence: ${flags.evidence.slice(0, 80)}`);
    }

    // 3. Update experiment status + dir + clear claim
    const updates: Record<string, unknown> = {
      status: flags.status,
      updatedAt: today,
      claimedBy: null,
      claimedAt: null,
    };
    if (flags.dir) updates.experimentDir = flags.dir;

    await db.update(experiments).set(updates).where(eq(experiments.id, args.id)).run();

    this.log(`Completed ${args.id}: ${flags.status}${flags.dir ? ` (${flags.dir})` : ""}`);
  }
}

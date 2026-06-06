import { Command, Args, Flags } from "@oclif/core";
import { eq, and } from "drizzle-orm";
import {
  db,
  experiments,
  experimentDependencies,
  killCriteria,
  successCriteria,
  evidence,
  experimentTags,
  tags,
  experimentReferences,
  references,
} from "@experiment/db";

export default class Claim extends Command {
  static description = "Atomically claim the next available experiment and output full YAML details";

  static args = {
    worker: Args.string({ description: "Worker ID (e.g. loop-primary, loop-research)", required: true }),
  };

  static flags = {
    id: Flags.string({ description: "Claim a specific experiment by ID (instead of auto-picking)" }),
    tag: Flags.string({ description: "Only claim experiments with this tag" }),
    "max-priority": Flags.integer({ description: "Only claim experiments with priority <= this value", default: 5 }),
    release: Flags.boolean({ description: "Release a stale claim (reset to open)", default: false }),
    "release-id": Flags.string({ description: "Experiment ID to release" }),
  };

  async run() {
    const { args, flags } = await this.parse(Claim);
    const now = new Date().toISOString();

    // Release mode
    if (flags.release && flags["release-id"]) {
      const exp = await db.select().from(experiments).where(eq(experiments.id, flags["release-id"])).get();
      if (!exp) { this.error(`Experiment "${flags["release-id"]}" not found`); }
      if (exp.status !== "active") { this.error(`Cannot release — status is "${exp.status}", not "active"`); }

      await db.update(experiments).set({
        status: "open",
        claimedBy: null,
        claimedAt: null,
        updatedAt: now.slice(0, 10),
      }).where(eq(experiments.id, flags["release-id"])).run();

      this.log(`Released ${flags["release-id"]} (was claimed by ${exp.claimedBy})`);
      return;
    }

    // Resolve which experiment to claim
    let targetId: string | null = null;

    if (flags.id) {
      // Specific experiment
      const exp = await db.select().from(experiments).where(eq(experiments.id, flags.id)).get();
      if (!exp) { this.error(`Experiment "${flags.id}" not found`); }
      if (exp.status !== "open") { this.error(`Cannot claim — status is "${exp.status}", not "open"`); }
      targetId = flags.id;
    } else {
      // Auto-pick: highest priority open experiment with satisfied dependencies
      // No scale filter — both micro and macro experiments are claimable
      const allOpen = await db.select().from(experiments)
        .where(eq(experiments.status, "open"))
        .orderBy(experiments.priority)
        .all();

      const candidates = allOpen.filter(e => e.priority <= flags["max-priority"]);

      // Build resolved set
      const resolved = new Set<string>();
      const allExps = await db.select({ id: experiments.id, status: experiments.status }).from(experiments).all();
      for (const e of allExps) {
        if (["proven", "supported", "killed"].includes(e.status)) resolved.add(e.id);
      }

      // Filter by tag if specified
      let taggedIds: Set<string> | null = null;
      if (flags.tag) {
        const tag = await db.select().from(tags).where(eq(tags.name, flags.tag.toLowerCase())).get();
        if (tag) {
          const tagged = await db.select({ id: experimentTags.experimentId })
            .from(experimentTags).where(eq(experimentTags.tagId, tag.id)).all();
          taggedIds = new Set(tagged.map(t => t.id));
        } else {
          taggedIds = new Set();
        }
      }

      // Track rejection reasons for diagnostic output
      let skippedByTag = 0;
      let skippedByDeps = 0;

      for (const candidate of candidates) {
        if (taggedIds && !taggedIds.has(candidate.id)) { skippedByTag++; continue; }

        const deps = await db.select({ id: experimentDependencies.dependsOnId })
          .from(experimentDependencies)
          .where(eq(experimentDependencies.experimentId, candidate.id))
          .all();

        if (deps.every(d => resolved.has(d.id))) {
          targetId = candidate.id;
          break;
        } else {
          skippedByDeps++;
        }
      }

      if (!targetId) {
        const total = allOpen.length;
        const priority = candidates.length;
        const parts: string[] = [`No claimable experiments found.`];
        parts.push(`  ${total} open, ${priority} within priority <= ${flags["max-priority"]}`);
        if (skippedByTag > 0) parts.push(`  ${skippedByTag} skipped (tag mismatch)`);
        if (skippedByDeps > 0) parts.push(`  ${skippedByDeps} skipped (unresolved deps)`);
        if (priority - skippedByTag - skippedByDeps === 0 && priority > 0) {
          parts.push(`  Hint: resolve deps or use --id <exp> to claim directly`);
        }
        this.log(parts.join("\n"));
        return;
      }
    }

    // Atomic claim: update only if still open (CAS)
    const result = await db.update(experiments).set({
      status: "active",
      claimedBy: args.worker,
      claimedAt: now,
      updatedAt: now.slice(0, 10),
    }).where(and(eq(experiments.id, targetId), eq(experiments.status, "open"))).run();

    if (result.rowsAffected === 0) {
      this.log(`Race condition: ${targetId} was claimed by another worker. Retry with: experiment claim ${args.worker}`);
      return;
    }

    // Output full YAML details (same as `get --yaml`)
    await this.outputFullYaml(targetId, args.worker);
  }

  private async outputFullYaml(id: string, worker: string) {
    const exp = await db.select().from(experiments).where(eq(experiments.id, id)).get();
    if (!exp) return;

    const deps = await db.select({ id: experimentDependencies.dependsOnId })
      .from(experimentDependencies).where(eq(experimentDependencies.experimentId, id)).all();
    const blocks = await db.select({ id: experimentDependencies.experimentId })
      .from(experimentDependencies).where(eq(experimentDependencies.dependsOnId, id)).all();
    const kcs = await db.select().from(killCriteria).where(eq(killCriteria.experimentId, id)).all();
    const scs = await db.select().from(successCriteria).where(eq(successCriteria.experimentId, id)).all();
    const evs = await db.select().from(evidence).where(eq(evidence.experimentId, id)).all();
    const expTags = await db.select({ name: tags.name }).from(experimentTags)
      .innerJoin(tags, eq(experimentTags.tagId, tags.id))
      .where(eq(experimentTags.experimentId, id)).all();
    const refs = await db.select({
      id: references.id, arxivId: references.arxivId, title: references.title,
      relevance: references.relevance, localPath: references.localPath,
    }).from(experimentReferences)
      .innerJoin(references, eq(experimentReferences.referenceId, references.id))
      .where(eq(experimentReferences.experimentId, id)).all();

    const y: string[] = [];
    y.push(`# Claimed by ${worker}`);
    y.push(`id: ${exp.id}`);
    y.push(`title: "${exp.title}"`);
    y.push(`status: ${exp.status}`);
    y.push(`scale: ${exp.scale}`);
    y.push(`priority: ${exp.priority}`);
    y.push(`platform: ${exp.platform ?? "~"}`);
    y.push(`experiment_dir: ${exp.experimentDir ?? "~"}`);
    y.push(`claimed_by: ${worker}`);

    if (deps.length > 0) {
      y.push(`depends_on:`);
      for (const d of deps) y.push(`  - ${d.id}`);
    } else {
      y.push(`depends_on: []`);
    }
    if (blocks.length > 0) {
      y.push(`blocks:`);
      for (const b of blocks) y.push(`  - ${b.id}`);
    } else {
      y.push(`blocks: []`);
    }

    if (kcs.length > 0) {
      y.push(`kill_criteria:`);
      for (const kc of kcs) {
        y.push(`  - id: ${kc.id}`);
        y.push(`    text: "${kc.text.replace(/"/g, '\\"')}"`);
        y.push(`    result: ${kc.result}`);
      }
    } else {
      y.push(`kill_criteria: [] # MISSING`);
    }

    if (scs.length > 0) {
      y.push(`success_criteria:`);
      for (const sc of scs) {
        y.push(`  - id: ${sc.id}`);
        y.push(`    condition: "${sc.condition.replace(/"/g, '\\"')}"`);
        y.push(`    unlocks: "${sc.unlocks.replace(/"/g, '\\"')}"`);
      }
    } else {
      y.push(`success_criteria: [] # MISSING`);
    }

    if (evs.length > 0) {
      y.push(`evidence:`);
      for (const ev of evs) {
        y.push(`  - claim: "${ev.claim.replace(/"/g, '\\"').slice(0, 200)}"`);
        y.push(`    verdict: ${ev.verdict ?? "~"}`);
      }
    } else {
      y.push(`evidence: []`);
    }

    if (expTags.length > 0) {
      y.push(`tags: [${expTags.map(t => t.name).join(", ")}]`);
    } else {
      y.push(`tags: []`);
    }

    if (refs.length > 0) {
      y.push(`references:`);
      for (const ref of refs) {
        y.push(`  - title: "${ref.title.replace(/"/g, '\\"')}"`);
        y.push(`    arxiv: ${ref.arxivId ?? "~"}`);
        y.push(`    relevance: "${ref.relevance.replace(/"/g, '\\"').slice(0, 120)}"`);
      }
    } else {
      y.push(`references: []`);
    }

    if (exp.notes) {
      y.push(`notes: |`);
      for (const line of exp.notes.split("\n")) {
        y.push(`  ${line}`);
      }
    } else {
      y.push(`notes: ~`);
    }

    // Completeness warnings
    const missing: string[] = [];
    if (kcs.length === 0) missing.push("kill_criteria");
    if (scs.length === 0) missing.push("success_criteria");
    if (expTags.length === 0) missing.push("tags");
    if (!exp.platform) missing.push("platform");
    if (missing.length > 0) {
      y.push(`# ⚠ INCOMPLETE: missing ${missing.join(", ")}`);
    }

    this.log(y.join("\n"));
  }
}

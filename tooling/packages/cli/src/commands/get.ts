import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import {
  db,
  experiments,
  killCriteria,
  successCriteria,
  evidence,
  experimentTags,
  tags,
  experimentDependencies,
  experimentReferences,
  references,
} from "@experiment/db";

export default class Get extends Command {
  static description = "Get full details of an experiment";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    yaml: Flags.boolean({ description: "Output as YAML (structured, machine-readable)", default: false }),
  };

  async run() {
    const { args, flags } = await this.parse(Get);

    const exp = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) {
      this.error(`Experiment "${args.id}" not found`);
    }

    const deps = await db
      .select({ id: experimentDependencies.dependsOnId })
      .from(experimentDependencies)
      .where(eq(experimentDependencies.experimentId, args.id))
      .all();

    const blocks = await db
      .select({ id: experimentDependencies.experimentId })
      .from(experimentDependencies)
      .where(eq(experimentDependencies.dependsOnId, args.id))
      .all();

    const kcs = await db.select().from(killCriteria).where(eq(killCriteria.experimentId, args.id)).all();
    const scs = await db.select().from(successCriteria).where(eq(successCriteria.experimentId, args.id)).all();
    const evs = await db.select().from(evidence).where(eq(evidence.experimentId, args.id)).all();

    const expTags = await db
      .select({ name: tags.name })
      .from(experimentTags)
      .innerJoin(tags, eq(experimentTags.tagId, tags.id))
      .where(eq(experimentTags.experimentId, args.id))
      .all();

    const refs = await db
      .select({
        id: references.id,
        arxivId: references.arxivId,
        title: references.title,
        url: references.url,
        relevance: references.relevance,
        localPath: references.localPath,
      })
      .from(experimentReferences)
      .innerJoin(references, eq(experimentReferences.referenceId, references.id))
      .where(eq(experimentReferences.experimentId, args.id))
      .all();

    if (flags.yaml) {
      this.outputYaml(exp, deps, blocks, kcs, scs, evs, expTags, refs);
    } else {
      this.outputHuman(exp, deps, blocks, kcs, scs, evs, expTags, refs);
    }
  }

  private outputYaml(
    exp: any, deps: any[], blocks: any[], kcs: any[], scs: any[], evs: any[], expTags: any[], refs: any[],
  ) {
    const yaml: string[] = [];
    const y = (line: string) => yaml.push(line);

    y(`id: ${exp.id}`);
    y(`title: "${exp.title}"`);
    y(`status: ${exp.status}`);
    y(`scale: ${exp.scale}`);
    y(`priority: ${exp.priority}`);
    y(`platform: ${exp.platform ?? "~"}`);
    y(`experiment_dir: ${exp.experimentDir ?? "~"}`);
    y(`claimed_by: ${exp.claimedBy ?? "~"}`);
    y(`claimed_at: ${exp.claimedAt ?? "~"}`);
    y(`created_at: ${exp.createdAt}`);
    y(`updated_at: ${exp.updatedAt}`);

    // Dependencies
    if (deps.length > 0) {
      y(`depends_on:`);
      for (const d of deps) y(`  - ${d.id}`);
    } else {
      y(`depends_on: []`);
    }
    if (blocks.length > 0) {
      y(`blocks:`);
      for (const b of blocks) y(`  - ${b.id}`);
    } else {
      y(`blocks: []`);
    }

    // Kill criteria
    if (kcs.length > 0) {
      y(`kill_criteria:`);
      for (const kc of kcs) {
        y(`  - id: ${kc.id}`);
        y(`    text: "${kc.text.replace(/"/g, '\\"')}"`);
        y(`    reason: "${kc.reason.replace(/"/g, '\\"')}"`);
        y(`    result: ${kc.result}`);
      }
    } else {
      y(`kill_criteria: [] # MISSING — add with: experiment kill-add ${exp.id} --text "..."`)
    }

    // Success criteria
    if (scs.length > 0) {
      y(`success_criteria:`);
      for (const sc of scs) {
        y(`  - id: ${sc.id}`);
        y(`    condition: "${sc.condition.replace(/"/g, '\\"')}"`);
        y(`    unlocks: "${sc.unlocks.replace(/"/g, '\\"')}"`);
        y(`    max_followup: ${sc.maxFollowup}`);
        y(`    reason: "${sc.reason.replace(/"/g, '\\"')}"`);
      }
    } else {
      y(`success_criteria: [] # MISSING — add with: experiment success-add ${exp.id} --condition "..." --unlocks "..."`)
    }

    // Evidence
    if (evs.length > 0) {
      y(`evidence:`);
      for (const ev of evs) {
        y(`  - date: ${ev.date}`);
        y(`    claim: "${ev.claim.replace(/"/g, '\\"')}"`);
        y(`    source: "${ev.source.replace(/"/g, '\\"')}"`);
        y(`    verdict: ${ev.verdict ?? "~"}`);
      }
    } else {
      y(`evidence: []`);
    }

    // Tags
    if (expTags.length > 0) {
      y(`tags: [${expTags.map((t) => t.name).join(", ")}]`);
    } else {
      y(`tags: [] # MISSING — add with: experiment tag ${exp.id} --add <tag>`)
    }

    // References
    if (refs.length > 0) {
      y(`references:`);
      for (const ref of refs) {
        y(`  - id: ${ref.id}`);
        y(`    arxiv_id: ${ref.arxivId ?? "~"}`);
        y(`    title: "${ref.title.replace(/"/g, '\\"')}"`);
        y(`    relevance: "${ref.relevance.replace(/"/g, '\\"')}"`);
        if (ref.localPath) y(`    local_path: "${ref.localPath}"`);
      }
    } else {
      y(`references: []`);
    }

    // Notes (full, not truncated)
    if (exp.notes) {
      y(`notes: |`);
      for (const line of exp.notes.split("\n")) {
        y(`  ${line}`);
      }
    } else {
      y(`notes: ~`);
    }

    // Completeness check
    const missing: string[] = [];
    if (kcs.length === 0) missing.push("kill_criteria");
    if (scs.length === 0) missing.push("success_criteria");
    if (expTags.length === 0) missing.push("tags");
    if (refs.length === 0) missing.push("references");
    if (!exp.platform) missing.push("platform");
    if (!exp.experimentDir) missing.push("experiment_dir");
    if (kcs.length > 0 && kcs.every((kc: any) => kc.result === "untested")) missing.push("kill_results (all untested)");

    if (missing.length > 0) {
      y(`# ⚠ INCOMPLETE: missing ${missing.join(", ")}`);
    }

    this.log(yaml.join("\n"));
  }

  private outputHuman(
    exp: any, deps: any[], blocks: any[], kcs: any[], scs: any[], evs: any[], expTags: any[], refs: any[],
  ) {
    // Header
    this.log(`\n${exp.id}`);
    this.log(`  Title:    ${exp.title}`);
    this.log(`  Status:   ${exp.status}`);
    this.log(`  Scale:    ${exp.scale}  Priority: ${exp.priority}`);
    this.log(`  Platform: ${exp.platform ?? "—"}`);
    this.log(`  Dir:      ${exp.experimentDir ?? "—"}`);
    if (exp.claimedBy) {
      this.log(`  Claimed:  ${exp.claimedBy} at ${exp.claimedAt}`);
    }
    this.log(`  Created:  ${exp.createdAt}  Updated: ${exp.updatedAt}`);

    // Dependencies
    if (deps.length > 0) {
      this.log(`\n  Depends on:`);
      for (const d of deps) this.log(`    - ${d.id}`);
    }

    // Blocks (reverse deps)
    if (blocks.length > 0) {
      this.log(`  Blocks:`);
      for (const b of blocks) this.log(`    - ${b.id}`);
    }

    // Kill criteria
    if (kcs.length > 0) {
      this.log(`\n  Kill Criteria:`);
      for (const kc of kcs) {
        const icon = kc.result === "pass" ? "✓" : kc.result === "fail" ? "✗" : kc.result === "inconclusive" ? "?" : "·";
        this.log(`    [${icon}] #${kc.id}: ${kc.text}`);
        if (kc.reason !== "pre-registered") this.log(`        Reason: ${kc.reason}`);
      }
    } else {
      this.log(`\n  Kill Criteria: NONE — add with: experiment kill-add ${exp.id} --text "..."`);
    }

    // Success criteria
    if (scs.length > 0) {
      this.log(`\n  Success Criteria:`);
      for (const sc of scs) {
        this.log(`    #${sc.id}: ${sc.condition}`);
        this.log(`    Unlocks: ${sc.unlocks} (max ${sc.maxFollowup} follow-up)`);
        if (sc.reason !== "pre-registered") this.log(`    Reason: ${sc.reason}`);
      }
    } else {
      this.log(`\n  Success Criteria: NONE — add with: experiment success-add ${exp.id} --condition "..." --unlocks "..."`);
    }

    // Evidence
    if (evs.length > 0) {
      this.log(`\n  Evidence (${evs.length}):`);
      for (const ev of evs) {
        const v = ev.verdict ? ` [${ev.verdict}]` : "";
        this.log(`    ${ev.date}${v}: ${ev.claim.slice(0, 120)}`);
        if (ev.source !== "imported-from-yml") this.log(`      Source: ${ev.source}`);
      }
    }

    // Tags
    if (expTags.length > 0) {
      this.log(`\n  Tags: ${expTags.map((t) => t.name).join(", ")}`);
    }

    // References
    if (refs.length > 0) {
      this.log(`\n  References:`);
      for (const ref of refs) {
        this.log(`    ${ref.title}: ${ref.relevance.slice(0, 80)}`);
        if (ref.localPath) this.log(`      Code: ${ref.localPath}`);
      }
    }

    // Notes (full — not truncated)
    if (exp.notes) {
      this.log(`\n  Notes:\n${exp.notes.split("\n").map((l: string) => `    ${l}`).join("\n")}`);
    }

    // Completeness warnings
    const missing: string[] = [];
    if (kcs.length === 0) missing.push("kill_criteria");
    if (scs.length === 0) missing.push("success_criteria");
    if (expTags.length === 0) missing.push("tags");
    if (refs.length === 0) missing.push("references");
    if (!exp.platform) missing.push("platform");
    if (!exp.experimentDir) missing.push("experiment_dir");
    if (kcs.length > 0 && kcs.every((kc: any) => kc.result === "untested")) missing.push("kill_results (all untested)");

    if (missing.length > 0) {
      this.log(`\n  ⚠ INCOMPLETE: ${missing.join(", ")}`);
    }

    this.log("");
  }
}

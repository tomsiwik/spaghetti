import { Command, Flags } from "@oclif/core";
import { resolve } from "path";
import { eq } from "drizzle-orm";
import {
  db,
  experiments,
  experimentDependencies,
  killCriteria,
  evidence as evidenceTable,
  tags as tagsTable,
  experimentTags,
  references as refsTable,
  experimentReferences,
} from "@experiment/db";
import { parseHypothesesYml, parseReferencesYml, extractArxivId } from "../lib/yaml-parser";

export default class Import extends Command {
  static description = "Import experiments and references from YAML files into SQLite";

  static flags = {
    hypotheses: Flags.string({
      description: "Path to HYPOTHESES.yml",
      required: true,
    }),
    references: Flags.string({
      description: "Path to REFERENCES.yml",
      required: true,
    }),
  };

  async run() {
    const { flags } = await this.parse(Import);

    const hypPath = resolve(flags.hypotheses);
    const refPath = resolve(flags.references);

    this.log(`Importing from ${hypPath}`);
    const nodes = parseHypothesesYml(hypPath);
    const nodeIds = Object.keys(nodes);
    this.log(`  Parsed ${nodeIds.length} experiment nodes`);

    // Phase 1: Insert experiments
    let expInserted = 0;
    let expSkipped = 0;

    for (const [id, node] of Object.entries(nodes)) {
      const existing = await db.select().from(experiments).where(eq(experiments.id, id)).get();
      if (existing) {
        expSkipped++;
        continue;
      }

      const now = new Date().toISOString().slice(0, 10);

      await db.insert(experiments)
        .values({
          id,
          title: node.title,
          status: node.status as any,
          scale: node.scale as any,
          priority: node.priority,
          experimentDir: node.experiment_dir,
          platform: null,
          notes: typeof node.notes === "string" ? node.notes : "",
          createdAt: node.created || now,
          updatedAt: now,
        })
        .run();

      // Kill criteria
      for (const kc of node.kill_criteria) {
        if (typeof kc === "string" && kc.trim()) {
          await db.insert(killCriteria)
            .values({
              experimentId: id,
              text: kc,
              reason: "pre-registered",
              result: "untested",
            })
            .run();
        }
      }

      // Evidence
      for (const ev of node.evidence) {
        if (typeof ev === "string") {
          await db.insert(evidenceTable)
            .values({
              experimentId: id,
              date: node.created || now,
              claim: ev,
              source: "imported-from-yml",
              verdict: null,
            })
            .run();
        } else if (ev && typeof ev === "object") {
          await db.insert(evidenceTable)
            .values({
              experimentId: id,
              date: ev.date || node.created || now,
              claim: ev.claim || "",
              source: ev.source || "imported-from-yml",
              verdict: null,
            })
            .run();
        }
      }

      // Tags
      for (const tagName of node.tags) {
        if (typeof tagName !== "string" || !tagName.trim()) continue;
        const clean = tagName.trim().toLowerCase();

        // Upsert tag
        await db.insert(tagsTable).values({ name: clean }).onConflictDoNothing().run();
        const tag = await db.select().from(tagsTable).where(eq(tagsTable.name, clean)).get();
        if (tag) {
          await db.insert(experimentTags)
            .values({ experimentId: id, tagId: tag.id })
            .onConflictDoNothing()
            .run();
        }
      }

      expInserted++;
    }

    // Phase 1b: Dependencies (after all experiments exist)
    let depCount = 0;
    for (const [id, node] of Object.entries(nodes)) {
      // depends_on
      if (node.depends_on) {
        for (const depId of node.depends_on) {
          if (nodes[depId]) {
            await db.insert(experimentDependencies)
              .values({ experimentId: id, dependsOnId: depId })
              .onConflictDoNothing()
              .run();
            depCount++;
          }
        }
      }
      // blocks = reverse dependency (if A blocks B, then B depends_on A)
      if (node.blocks) {
        for (const blockedId of node.blocks) {
          if (nodes[blockedId]) {
            await db.insert(experimentDependencies)
              .values({ experimentId: blockedId, dependsOnId: id })
              .onConflictDoNothing()
              .run();
            depCount++;
          }
        }
      }
    }

    this.log(`  Inserted ${expInserted} experiments (${expSkipped} skipped, ${depCount} dependencies)`);

    // Phase 2: References
    this.log(`Importing from ${refPath}`);
    const refs = parseReferencesYml(refPath);
    this.log(`  Parsed ${refs.length} references`);

    let refInserted = 0;
    for (const ref of refs) {
      const arxivId = extractArxivId(ref.source);
      const title = ref.dir || ref.insight.slice(0, 60);

      const result = await db
        .insert(refsTable)
        .values({
          arxivId: arxivId,
          url: ref.source,
          title,
          relevance: ref.insight,
          localPath: ref.dir ? `references/${ref.dir}/` : null,
          reproSteps: null,
        })
        .run();

      const refId = Number(result.lastInsertRowid);

      // Link to experiments
      for (const nodeId of ref.nodes) {
        if (nodes[nodeId]) {
          await db.insert(experimentReferences)
            .values({ experimentId: nodeId, referenceId: refId })
            .onConflictDoNothing()
            .run();
        }
      }

      refInserted++;
    }

    this.log(`  Inserted ${refInserted} references`);
    this.log(`\nDone. ${expInserted} experiments, ${refInserted} references imported.`);
  }
}

import { Command } from "@oclif/core";
import { sql } from "drizzle-orm";
import { db, experiments, evidence, killCriteria, tags, experimentTags, references, findings, methods } from "@experiment/db";

export default class Stats extends Command {
  static description = "Show experiment tracking dashboard";

  async run() {
    await this.parse(Stats);
    // Status distribution
    const statusDist = await db
      .select({ status: experiments.status, count: sql<number>`count(*)` })
      .from(experiments)
      .groupBy(experiments.status);

    const total = statusDist.reduce((sum, r) => sum + r.count, 0);
    const killed = statusDist.find((r) => r.status === "killed")?.count ?? 0;
    const proven = statusDist.find((r) => r.status === "proven")?.count ?? 0;
    const supported = statusDist.find((r) => r.status === "supported")?.count ?? 0;

    this.log("\n  Experiment Stats");
    this.log("  " + "─".repeat(40));
    this.log(`  Total experiments: ${total}`);
    for (const r of statusDist.sort((a, b) => b.count - a.count)) {
      const bar = "█".repeat(Math.round((r.count / total) * 30));
      this.log(`    ${r.status.padEnd(12)} ${String(r.count).padStart(3)} ${bar}`);
    }
    this.log(`\n  Kill rate: ${((killed / total) * 100).toFixed(1)}% (${killed}/${total})`);
    this.log(`  Success rate: ${(((proven + supported) / total) * 100).toFixed(1)}% (${proven + supported}/${total})`);

    // Evidence count
    const [evCount] = await db.select({ count: sql<number>`count(*)` }).from(evidence);
    this.log(`\n  Evidence entries: ${evCount?.count}`);

    // Kill criteria count
    const [kcCount] = await db.select({ count: sql<number>`count(*)` }).from(killCriteria);
    this.log(`  Kill criteria: ${kcCount?.count}`);

    // References
    const [refCount] = await db.select({ count: sql<number>`count(*)` }).from(references);
    this.log(`  References: ${refCount?.count}`);

    // Top tags
    const topTags = await db
      .select({ name: tags.name, count: sql<number>`count(*)` })
      .from(experimentTags)
      .innerJoin(tags, sql`${experimentTags.tagId} = ${tags.id}`)
      .groupBy(tags.name)
      .orderBy(sql`count(*) DESC`)
      .limit(15);

    this.log("\n  Top Tags:");
    for (const t of topTags) {
      this.log(`    ${t.name.padEnd(25)} ${t.count}`);
    }

    // Findings
    const findingsDist = await db
      .select({ status: findings.status, count: sql<number>`count(*)` })
      .from(findings)
      .groupBy(findings.status);
    const findingsTotal = findingsDist.reduce((sum, r) => sum + r.count, 0);
    this.log(`\n  Findings: ${findingsTotal}`);
    for (const f of findingsDist.sort((a, b) => b.count - a.count)) {
      this.log(`    ${f.status.padEnd(14)} ${f.count}`);
    }

    // Findings with impossibility structures
    const [withStructure] = await db
      .select({ count: sql<number>`count(*)` })
      .from(findings)
      .where(sql`${findings.impossibilityStructure} IS NOT NULL`);
    if (findingsTotal > 0) {
      this.log(`  With impossibility structure: ${withStructure?.count}/${findingsTotal}`);
    }

    // Method bank
    try {
      const methodsDist = await db
        .select({ status: methods.status, count: sql<number>`count(*)` })
        .from(methods)
        .groupBy(methods.status);
      const methodsTotal = methodsDist.reduce((sum, r) => sum + r.count, 0);
      if (methodsTotal > 0) {
        this.log(`\n  Method Bank: ${methodsTotal}`);
        for (const m of methodsDist.sort((a, b) => b.count - a.count)) {
          this.log(`    ${m.status.padEnd(14)} ${m.count}`);
        }
      }
    } catch { /* methods table may not exist yet */ }

    // Scale distribution
    const scaleDist = await db
      .select({ scale: experiments.scale, count: sql<number>`count(*)` })
      .from(experiments)
      .groupBy(experiments.scale);
    this.log("\n  Scale:");
    for (const s of scaleDist) {
      this.log(`    ${s.scale.padEnd(10)} ${s.count}`);
    }

    this.log("");
  }
}

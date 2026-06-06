import { Command, Args } from "@oclif/core";
import { createClient } from "@libsql/client";

export default class Query extends Command {
  static description = "Full-text search across experiments and evidence";

  static args = {
    text: Args.string({ description: "Search query", required: true }),
  };

  async run() {
    const { args } = await this.parse(Query);

    const client = createClient({
      url: process.env.TURSO_DATABASE_URL!,
      authToken: process.env.TURSO_AUTH_TOKEN,
    });

    // Search experiments
    const expResults = await client.execute({
      sql: `SELECT id, title, snippet(experiments_fts, 1, '>>>', '<<<', '...', 30) as snippet, rank
            FROM experiments_fts
            WHERE experiments_fts MATCH ?
            ORDER BY rank
            LIMIT 15`,
      args: [args.text],
    });

    // Search evidence
    const evResults = await client.execute({
      sql: `SELECT e.experiment_id, snippet(evidence_fts, 0, '>>>', '<<<', '...', 30) as snippet, evidence_fts.rank
            FROM evidence_fts
            JOIN evidence e ON e.rowid = evidence_fts.rowid
            WHERE evidence_fts MATCH ?
            ORDER BY evidence_fts.rank
            LIMIT 15`,
      args: [args.text],
    });

    // Search findings
    let findingResults: any = { rows: [] };
    try {
      findingResults = await client.execute({
        sql: `SELECT f.id, f.title, f.status, snippet(findings_fts, 1, '>>>', '<<<', '...', 30) as snippet, findings_fts.rank
              FROM findings_fts
              JOIN findings f ON f.rowid = findings_fts.rowid
              WHERE findings_fts MATCH ?
              ORDER BY findings_fts.rank
              LIMIT 15`,
        args: [args.text],
      });
    } catch {
      // findings_fts may not exist yet
    }

    // Search methods
    let methodResults: any = { rows: [] };
    try {
      methodResults = await client.execute({
        sql: `SELECT m.id, m.name, m.status, snippet(methods_fts, 2, '>>>', '<<<', '...', 30) as snippet, methods_fts.rank
              FROM methods_fts
              JOIN methods m ON m.rowid = methods_fts.rowid
              WHERE methods_fts MATCH ?
              ORDER BY methods_fts.rank
              LIMIT 15`,
        args: [args.text],
      });
    } catch {
      // methods_fts may not exist yet
    }

    if (expResults.rows.length === 0 && evResults.rows.length === 0 && findingResults.rows.length === 0 && methodResults.rows.length === 0) {
      this.log(`No results for "${args.text}"`);
      return;
    }

    if (expResults.rows.length > 0) {
      this.log(`\n  Experiments matching "${args.text}":`);
      for (const r of expResults.rows) {
        this.log(`    ${r.id}`);
        this.log(`      ${r.snippet}`);
      }
    }

    if (evResults.rows.length > 0) {
      this.log(`\n  Evidence matching "${args.text}":`);
      for (const r of evResults.rows) {
        this.log(`    [${r.experiment_id}]`);
        this.log(`      ${r.snippet}`);
      }
    }

    if (findingResults.rows.length > 0) {
      this.log(`\n  Findings matching "${args.text}":`);
      for (const r of findingResults.rows) {
        this.log(`    #${r.id} [${r.status}] ${r.title}`);
        this.log(`      ${r.snippet}`);
      }
    }

    if (methodResults.rows.length > 0) {
      this.log(`\n  Methods matching "${args.text}":`);
      for (const r of methodResults.rows) {
        this.log(`    #${r.id} [${r.status}] ${r.name}`);
        this.log(`      ${r.snippet}`);
      }
    }

    this.log(`\n  ${expResults.rows.length} experiments, ${evResults.rows.length} evidence, ${findingResults.rows.length} findings, ${methodResults.rows.length} methods found`);
  }
}

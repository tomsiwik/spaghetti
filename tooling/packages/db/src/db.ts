import { createClient, type Client } from "@libsql/client";
import { drizzle, type LibSQLDatabase } from "drizzle-orm/libsql";
import * as schema from "./schema";

let _client: Client | null = null;
let _db: LibSQLDatabase<typeof schema> | null = null;

function getClient(): Client {
  if (!_client) {
    const url = process.env.TURSO_DATABASE_URL;
    const authToken = process.env.TURSO_AUTH_TOKEN;
    if (!url) throw new Error("TURSO_DATABASE_URL is not set — add it to .env");
    _client = createClient({ url, authToken });
  }
  return _client;
}

export function getDb(): LibSQLDatabase<typeof schema> {
  if (!_db) {
    _db = drizzle(getClient(), { schema });
  }
  return _db;
}

// Convenience: named export that matches the old `db` usage via a proxy
export const db = new Proxy({} as LibSQLDatabase<typeof schema>, {
  get(_target, prop) {
    return (getDb() as any)[prop];
  },
});

export type Db = LibSQLDatabase<typeof schema>;

// Run FTS + trigger setup (idempotent)
export async function initFts() {
  await getClient().executeMultiple(`
    CREATE VIRTUAL TABLE IF NOT EXISTS experiments_fts
      USING fts5(id, title, notes, content=experiments, content_rowid=rowid);

    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts
      USING fts5(claim, source, content=evidence, content_rowid=rowid);

    CREATE TRIGGER IF NOT EXISTS experiments_ai AFTER INSERT ON experiments BEGIN
      INSERT INTO experiments_fts(rowid, id, title, notes) VALUES (new.rowid, new.id, new.title, new.notes);
    END;

    CREATE TRIGGER IF NOT EXISTS experiments_au AFTER UPDATE ON experiments BEGIN
      INSERT INTO experiments_fts(experiments_fts, rowid, id, title, notes) VALUES ('delete', old.rowid, old.id, old.title, old.notes);
      INSERT INTO experiments_fts(rowid, id, title, notes) VALUES (new.rowid, new.id, new.title, new.notes);
    END;

    CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence BEGIN
      INSERT INTO evidence_fts(rowid, claim, source) VALUES (new.rowid, new.claim, new.source);
    END;

    CREATE TRIGGER IF NOT EXISTS evidence_au AFTER UPDATE ON evidence BEGIN
      INSERT INTO evidence_fts(evidence_fts, rowid, claim, source) VALUES ('delete', old.rowid, old.claim, old.source);
      INSERT INTO evidence_fts(rowid, claim, source) VALUES (new.rowid, new.claim, new.source);
    END;

    CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts
      USING fts5(title, result, caveat, failure_mode, impossibility_structure, content=findings, content_rowid=rowid);

    CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
      INSERT INTO findings_fts(rowid, title, result, caveat, failure_mode, impossibility_structure)
        VALUES (new.rowid, new.title, new.result, new.caveat, new.failure_mode, new.impossibility_structure);
    END;

    CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
      INSERT INTO findings_fts(findings_fts, rowid, title, result, caveat, failure_mode, impossibility_structure)
        VALUES ('delete', old.rowid, old.title, old.result, old.caveat, old.failure_mode, old.impossibility_structure);
      INSERT INTO findings_fts(rowid, title, result, caveat, failure_mode, impossibility_structure)
        VALUES (new.rowid, new.title, new.result, new.caveat, new.failure_mode, new.impossibility_structure);
    END;

    CREATE VIRTUAL TABLE IF NOT EXISTS methods_fts
      USING fts5(name, description, solves, proven_in, use_when, not_now_because, content=methods, content_rowid=rowid);

    CREATE TRIGGER IF NOT EXISTS methods_ai AFTER INSERT ON methods BEGIN
      INSERT INTO methods_fts(rowid, name, description, solves, proven_in, use_when, not_now_because)
        VALUES (new.rowid, new.name, new.description, new.solves, new.proven_in, new.use_when, new.not_now_because);
    END;

    CREATE TRIGGER IF NOT EXISTS methods_au AFTER UPDATE ON methods BEGIN
      INSERT INTO methods_fts(methods_fts, rowid, name, description, solves, proven_in, use_when, not_now_because)
        VALUES ('delete', old.rowid, old.name, old.description, old.solves, old.proven_in, old.use_when, old.not_now_because);
      INSERT INTO methods_fts(rowid, name, description, solves, proven_in, use_when, not_now_because)
        VALUES (new.rowid, new.name, new.description, new.solves, new.proven_in, new.use_when, new.not_now_because);
    END;
  `);
}

#!/usr/bin/env bun
/**
 * Import findings from FINDINGS.md into the Turso findings table.
 *
 * Run from packages/db so @libsql/client resolves:
 *   cd packages/db && export $(grep -v '^#' ../../.env | xargs) && bun ../../scripts/import-findings.ts
 */

import { createClient } from "@libsql/client";
import { readFileSync } from "fs";
import { resolve } from "path";

// ── Connect ─────────────────────────────────────────────────────────────────

const db = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

// ── Types ───────────────────────────────────────────────────────────────────

interface Finding {
  title: string;
  status: "conclusive" | "supported" | "killed" | "provisional";
  result: string;
  caveat: string | null;
  scale: "micro" | "macro";
  date: string;
}

// ── Section config ──────────────────────────────────────────────────────────

interface SectionConfig {
  pattern: string;
  status: "conclusive" | "supported" | "killed" | "provisional";
  scale: "micro" | "macro";
}

const SECTIONS: SectionConfig[] = [
  { pattern: "## Conclusive Results (Macro Scale", status: "conclusive", scale: "macro" },
  { pattern: "## Supported (Macro Scale)", status: "supported", scale: "macro" },
  { pattern: "## Killed at Macro", status: "killed", scale: "macro" },
  { pattern: "## Supported (Micro Scale, BitNet-2B-4T)", status: "supported", scale: "micro" },
  { pattern: "## Supported (Micro Scale + Macro Pilot)", status: "supported", scale: "micro" },
];

// ── Parse FINDINGS.md ───────────────────────────────────────────────────────

const mdPath = resolve(import.meta.dir, "../FINDINGS.md");
const md = readFileSync(mdPath, "utf-8");
const lines = md.split("\n");

// Parse table rows from a section
function parseTableRows(startIdx: number): { rows: { finding: string; result: string; evidence: string }[]; endIdx: number } {
  const rows: { finding: string; result: string; evidence: string }[] = [];
  let i = startIdx;

  // Find the table header
  while (i < lines.length && !lines[i].startsWith("| Finding")) {
    i++;
  }
  if (i >= lines.length) return { rows, endIdx: i };

  // Skip header and separator
  i += 2;

  // Parse rows until we hit a non-table line
  while (i < lines.length && lines[i].startsWith("|")) {
    const cells = lines[i].split("|").map(c => c.trim()).filter(Boolean);
    if (cells.length >= 3) {
      rows.push({
        finding: cells[0].replace(/\*\*/g, ""),
        result: cells[1],
        evidence: cells.slice(2).join(" | "),
      });
    }
    i++;
  }

  return { rows, endIdx: i };
}

// Parse caveats paragraphs and build a map: finding-keyword -> { text, date, statusOverride }
function parseCaveats(text: string): Map<string, { text: string; date: string | null; statusOverride: string | null }> {
  const map = new Map<string, { text: string; date: string | null; statusOverride: string | null }>();

  // Match **Caveats (...):** paragraphs
  const caveatRegex = /\*\*Caveats \(([^)]+)\):\*\*\s*([\s\S]*?)(?=\n\n\*\*Caveats|\n\n##|\n\n\||\n$)/g;
  let match: RegExpExecArray | null;

  while ((match = caveatRegex.exec(text)) !== null) {
    const key = match[1].trim();
    const body = match[2].trim();

    // Extract date
    const dateMatch = body.match(/Date:\s*(\d{4}-\d{2}-\d{2})/);
    const date = dateMatch ? dateMatch[1] : null;

    // Extract status override
    const statusMatch = body.match(/Status:\s*\*\*(\w+)\*\*/);
    const statusOverride = statusMatch ? statusMatch[1].toLowerCase() : null;

    map.set(key.toLowerCase(), { text: body, date, statusOverride });
  }

  return map;
}

// ── Main ────────────────────────────────────────────────────────────────────

const caveats = parseCaveats(md);
const findings: Finding[] = [];

for (const section of SECTIONS) {
  // Find section start
  const sectionIdx = lines.findIndex(l => l.startsWith(section.pattern));
  if (sectionIdx === -1) {
    console.warn(`Section not found: ${section.pattern}`);
    continue;
  }

  // Find where the next section starts
  const nextSectionIdx = lines.findIndex((l, idx) => idx > sectionIdx && l.startsWith("## "));
  const sectionEnd = nextSectionIdx === -1 ? lines.length : nextSectionIdx;

  const { rows } = parseTableRows(sectionIdx);

  for (const row of rows) {
    // Try to find matching caveat
    let caveat: string | null = null;
    let date = "2026-03-28";
    let status = section.status;

    // Try to match caveats by finding a key that is a substring of the finding title or vice versa
    const titleLower = row.finding.toLowerCase();
    for (const [key, val] of caveats.entries()) {
      // Check if caveat key matches finding title
      if (titleLower.includes(key) || key.includes(titleLower.substring(0, 30)) ||
          fuzzyMatch(key, titleLower)) {
        caveat = val.text;
        if (val.date) date = val.date;
        if (val.statusOverride) {
          const s = val.statusOverride;
          if (s === "conclusive" || s === "supported" || s === "killed" || s === "provisional" || s === "proven") {
            status = s === "proven" ? "conclusive" : s as any;
          }
        }
        break;
      }
    }

    findings.push({
      title: row.finding,
      status,
      result: row.result,
      caveat,
      scale: section.scale,
      date,
    });
  }
}

function fuzzyMatch(caveatKey: string, findingTitle: string): boolean {
  // Extract meaningful words from both
  const keyWords = caveatKey.split(/[\s,()-]+/).filter(w => w.length > 3).map(w => w.toLowerCase());
  const titleWords = findingTitle.split(/[\s,()-]+/).filter(w => w.length > 3).map(w => w.toLowerCase());

  if (keyWords.length === 0) return false;

  let matchCount = 0;
  for (const kw of keyWords) {
    if (titleWords.some(tw => tw.includes(kw) || kw.includes(tw))) {
      matchCount++;
    }
  }
  return matchCount / keyWords.length >= 0.6;
}

// ── Insert into DB ──────────────────────────────────────────────────────────

const now = new Date().toISOString();

// Delete existing test findings
await db.execute("DELETE FROM findings WHERE id IN (1, 2)");
console.log("Deleted test findings (id 1 and 2)");

// Insert all findings
let inserted = 0;
for (const f of findings) {
  await db.execute({
    sql: `INSERT INTO findings (title, status, result, caveat, experiment_id, scale, failure_mode, impossibility_structure, date, created_at, updated_at)
          VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?, ?)`,
    args: [f.title, f.status, f.result, f.caveat, f.scale, f.date, now, now],
  });
  inserted++;
}

console.log(`Imported ${inserted} findings into the database.`);

// Verify
const count = await db.execute("SELECT COUNT(*) as cnt FROM findings");
console.log(`Total findings in DB: ${(count.rows[0] as any).cnt}`);

await db.close();

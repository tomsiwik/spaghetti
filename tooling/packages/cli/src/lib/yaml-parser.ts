import { readFileSync } from "fs";
import YAML from "yaml";

export interface RawNode {
  title: string;
  status: string;
  scale: string;
  priority: number;
  depends_on: string[] | null;
  blocks: string[] | null;
  experiment_dir: string | null;
  kill_criteria: string[];
  evidence: (string | { date: string; claim: string; source: string })[];
  tags: string[];
  created: string;
  notes: string;
}

export interface RawReference {
  dir: string;
  source: string;
  insight: string;
  nodes: string[];
  added: string;
}

export function parseHypothesesYml(path: string): Record<string, RawNode> {
  const content = readFileSync(path, "utf-8");

  // YAML parsing fails on some entries due to special chars. Use a lenient approach:
  // split by top-level node keys and parse each block individually
  const nodes: Record<string, RawNode> = {};

  // Extract the nodes section
  const nodesMatch = content.indexOf("\nnodes:\n");
  if (nodesMatch === -1) return nodes;

  const nodesSection = content.slice(nodesMatch + "\nnodes:\n".length);

  // Split by experiment node keys (2-space indent + exp_)
  const nodeBlocks = nodesSection.split(/\n  (exp_\w+):\n/);
  // nodeBlocks[0] is empty/whitespace, then alternating [key, block, key, block, ...]

  for (let i = 1; i < nodeBlocks.length; i += 2) {
    const key = nodeBlocks[i];
    const block = nodeBlocks[i + 1];
    if (!key || !block) continue;

    try {
      const parsed = YAML.parse(`${key}:\n${indent(block, 2)}`);
      const node = parsed[key];
      if (node && node.title) {
        nodes[key] = {
          title: node.title ?? "",
          status: node.status ?? "open",
          scale: node.scale ?? "micro",
          priority: node.priority ?? 5,
          depends_on: node.depends_on ?? null,
          blocks: node.blocks ?? null,
          experiment_dir: node.experiment_dir ?? null,
          kill_criteria: Array.isArray(node.kill_criteria) ? node.kill_criteria : [],
          evidence: Array.isArray(node.evidence) ? node.evidence : [],
          tags: Array.isArray(node.tags) ? node.tags.filter((t: unknown) => typeof t === "string") : [],
          created: node.created ?? new Date().toISOString().slice(0, 10),
          notes: node.notes ?? "",
        };
      }
    } catch {
      // Skip unparseable nodes
      console.warn(`  Warning: could not parse node ${key}, skipping`);
    }
  }

  return nodes;
}

export function parseReferencesYml(path: string): RawReference[] {
  const content = readFileSync(path, "utf-8");
  const parsed = YAML.parse(content);
  const entries = parsed?.entries;
  if (!Array.isArray(entries)) return [];

  return entries
    .filter((e: any) => e && typeof e === "object" && e.source)
    .map((e: any) => ({
      dir: e.dir ?? "",
      source: e.source ?? "",
      insight: e.insight ?? "",
      nodes: Array.isArray(e.nodes) ? e.nodes : [],
      added: e.added ?? "",
    }));
}

function indent(text: string, spaces: number): string {
  const prefix = " ".repeat(spaces);
  return text
    .split("\n")
    .map((line) => (line.trim() ? prefix + line : line))
    .join("\n");
}

export function extractArxivId(url: string): string | null {
  const match = url.match(/arxiv\.org\/abs\/(\d{4}\.\d{4,5})/);
  return match ? match[1] : null;
}

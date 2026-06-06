import { sqliteTable, text, integer, real, primaryKey } from "drizzle-orm/sqlite-core";

// ── Core entity ──────────────────────────────────────────────────────────────

export const experiments = sqliteTable("experiments", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  status: text("status", { enum: ["open", "active", "proven", "supported", "killed"] }).notNull(),
  scale: text("scale", { enum: ["micro", "macro"] }).notNull(),
  priority: integer("priority").notNull(),
  experimentDir: text("experiment_dir"),
  platform: text("platform", { enum: ["local", "local-apple", "runpod-flash"] }),
  notes: text("notes"),
  claimedBy: text("claimed_by"),
  claimedAt: text("claimed_at"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

// ── Dependencies (self-referential M2M) ──────────────────────────────────────

export const experimentDependencies = sqliteTable(
  "experiment_dependencies",
  {
    experimentId: text("experiment_id").notNull().references(() => experiments.id),
    dependsOnId: text("depends_on_id").notNull().references(() => experiments.id),
  },
  (t) => [primaryKey({ columns: [t.experimentId, t.dependsOnId] })],
);

// ── Kill criteria ────────────────────────────────────────────────────────────

export const killCriteria = sqliteTable("kill_criteria", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  experimentId: text("experiment_id").notNull().references(() => experiments.id),
  text: text("text").notNull(),
  reason: text("reason").notNull(),
  result: text("result", { enum: ["pass", "fail", "inconclusive", "untested"] }).notNull().default("untested"),
});

// ── Success criteria ─────────────────────────────────────────────────────────

export const successCriteria = sqliteTable("success_criteria", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  experimentId: text("experiment_id").notNull().references(() => experiments.id),
  condition: text("condition").notNull(),
  unlocks: text("unlocks").notNull(),
  maxFollowup: integer("max_followup").notNull().default(1),
  reason: text("reason").notNull(),
});

// ── Evidence ─────────────────────────────────────────────────────────────────

export const evidence = sqliteTable("evidence", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  experimentId: text("experiment_id").notNull().references(() => experiments.id),
  date: text("date").notNull(),
  claim: text("claim").notNull(),
  source: text("source").notNull(),
  verdict: text("verdict", { enum: ["pass", "fail", "inconclusive"] }),
});

// ── Tags (shared across experiments and references) ──────────────────────────

export const tags = sqliteTable("tags", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  name: text("name").notNull().unique(),
});

export const experimentTags = sqliteTable(
  "experiment_tags",
  {
    experimentId: text("experiment_id").notNull().references(() => experiments.id),
    tagId: integer("tag_id").notNull().references(() => tags.id),
  },
  (t) => [primaryKey({ columns: [t.experimentId, t.tagId] })],
);

// ── References ───────────────────────────────────────────────────────────────

export const references = sqliteTable("references", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  arxivId: text("arxiv_id"),
  url: text("url").notNull(),
  title: text("title").notNull(),
  relevance: text("relevance").notNull(),
  localPath: text("local_path"),
  reproSteps: text("repro_steps"),
});

export const referenceTags = sqliteTable(
  "reference_tags",
  {
    referenceId: integer("reference_id").notNull().references(() => references.id),
    tagId: integer("tag_id").notNull().references(() => tags.id),
  },
  (t) => [primaryKey({ columns: [t.referenceId, t.tagId] })],
);

// ── Experiment ↔ Reference junction ──────────────────────────────────────────

export const experimentReferences = sqliteTable(
  "experiment_references",
  {
    experimentId: text("experiment_id").notNull().references(() => experiments.id),
    referenceId: integer("reference_id").notNull().references(() => references.id),
  },
  (t) => [primaryKey({ columns: [t.experimentId, t.referenceId] })],
);

// ── Findings ────────────────────────────────────────────────────────────────

export const findings = sqliteTable("findings", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  title: text("title").notNull(),
  status: text("status", {
    enum: ["conclusive", "supported", "killed", "provisional"],
  }).notNull(),
  result: text("result").notNull(),
  caveat: text("caveat"),
  experimentId: text("experiment_id").references(() => experiments.id),
  scale: text("scale", { enum: ["micro", "macro"] }),
  failureMode: text("failure_mode"),
  impossibilityStructure: text("impossibility_structure"),
  date: text("date").notNull(),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

export const findingTags = sqliteTable(
  "finding_tags",
  {
    findingId: integer("finding_id").notNull().references(() => findings.id),
    tagId: integer("tag_id").notNull().references(() => tags.id),
  },
  (t) => [primaryKey({ columns: [t.findingId, t.tagId] })],
);

export const findingReferences = sqliteTable(
  "finding_references",
  {
    findingId: integer("finding_id").notNull().references(() => findings.id),
    referenceId: integer("reference_id").notNull().references(() => references.id),
  },
  (t) => [primaryKey({ columns: [t.findingId, t.referenceId] })],
);

// ── Method Bank ────────────────────────────────────────────────────────────
// Reusable techniques indexed by the problem they solve.
// "When problem X arises, reach for method Y."

export const methods = sqliteTable("methods", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  name: text("name").notNull().unique(),
  description: text("description").notNull(),
  solves: text("solves").notNull(),
  provenIn: text("proven_in"),
  useWhen: text("use_when"),
  notNowBecause: text("not_now_because"),
  source: text("source"),
  status: text("status", {
    enum: ["parked", "exploring", "applied", "rejected"],
  }).notNull().default("parked"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

export const methodTags = sqliteTable(
  "method_tags",
  {
    methodId: integer("method_id").notNull().references(() => methods.id),
    tagId: integer("tag_id").notNull().references(() => tags.id),
  },
  (t) => [primaryKey({ columns: [t.methodId, t.tagId] })],
);

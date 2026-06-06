<p align="center">
  <picture>
    <source srcset="docs/assets/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="docs/assets/logo-light.svg" media="(prefers-color-scheme: light)">
    <img src="docs/assets/logo-light.svg" alt="spaghetti logo">
  </picture>
</p>
<p align="center">Throw it at the wall and see what sticks.</p>

---

### What is this?

Spaghetti is a structured experiment framework for research that needs to move fast without losing rigor. Claim experiments, run them, record findings, kill what doesn't work — all tracked in a queryable database.

Works for any domain: ML research, due diligence, SEO testing, marketing experiments, or anything where you need to track hypotheses, evidence, and outcomes.

### How it works

1. **Hypothesize** — every experiment starts from a cited source or prior finding
2. **Run** — execute the experiment, collect measurements
3. **Record** — findings are tagged as conclusive, supported, provisional, or killed
4. **Query** — full-text search across experiments, evidence, and findings

### Structure

```
pierre/             # live MLX code: composition core + merge/ libraries
experiments/        # the research log — models/ (micro, local MLX), macro/ (GPU/RunPod),
                    #   _runs/ (job + run logs), GUIDE.md (how to run experiments)
tooling/            # experiment framework: packages/ (CLI + DB), scripts/, tools/
data/               # adapters/ (trained weights + registry) + corpora/ (datasets)
docs/               # guides/ (MLX, adapters) · research/ (notes) · references/ (papers) ·
                    #   archive/ (superseded) · assets/
```

### Where to look
| For… | Read |
|---|---|
| What's true now (verified state) | `STATUS.md` |
| **How to run experiments** (the canonical process) | `experiments/GUIDE.md` |
| Roadmap + platform (base model, next gate) | `PLAN.md` |
| Agent entry point | `AGENTS.md` |

### Setup from scratch

#### Prerequisites

- [Bun](https://bun.sh) — JS runtime and package manager
- A [Turso](https://turso.tech) account (free tier works fine)

#### 1. Create a Turso database

```bash
# Install the Turso CLI
brew install tursodatabase/tap/turso    # or: curl -sSfL https://get.tur.so/install.sh | bash

# Sign up / log in
turso auth login

# Create a database (pick any name)
turso db create spaghetti

# Get your credentials
turso db show spaghetti --url            # → libsql://spaghetti-yourname.turso.io
turso db tokens create spaghetti         # → your auth token
```

#### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```
TURSO_DATABASE_URL=libsql://spaghetti-yourname.turso.io
TURSO_AUTH_TOKEN=your-token-here
```

#### 3. Install dependencies and push schema

```bash
bun install

# Push the Drizzle schema to your new database
cd packages/db
bunx drizzle-kit push
```

This creates all tables (experiments, findings, evidence, kill criteria, methods, references, tags) directly from the TypeScript schema — no migration files needed.

#### 4. Initialize full-text search

FTS tables and triggers are created automatically the first time you run any `experiment` command. They're idempotent, so re-running is safe.

#### 5. Verify it works

```bash
experiment stats                          # should show all zeroes
experiment add my-first-experiment \
  --title "Does X improve Y?" \
  --notes "Based on [source]" \
  --scale micro
experiment list                           # should show your experiment
```

### Usage

```bash
# Workflow
experiment add <id> --title "..." --notes "..." --scale micro
experiment claim <worker-id>              # grab the next open experiment
experiment complete <id> --status supported --evidence "..."

# Findings
experiment finding-add --title "..." --status supported --result "..."
experiment finding-list
experiment finding-get <id>

# Search everything
experiment query "search term"

# Methods bank (reusable techniques)
experiment method-add --name "..." --description "..." --solves "..."
experiment method-list
```

### Adapting to your domain

The schema is general-purpose. The core concepts map to any research context:

| Concept | ML research | Marketing | Due diligence |
|---------|------------|-----------|---------------|
| **Experiment** | "Does LoRA rank 8 beat rank 4?" | "Does CTA color affect conversion?" | "Is vendor X compliant with SOC2?" |
| **Kill criteria** | "Loss > 2.0 after 100 steps" | "CTR below baseline after 1k impressions" | "No audit report available" |
| **Evidence** | "Loss=1.3 at step 100, pass" | "CTR +12% after 2k impressions, pass" | "SOC2 Type II report dated 2025-11" |
| **Finding** | "Rank 8 outperforms rank 4 on GSM8K" | "Green CTA converts 15% better" | "Vendor X meets all compliance requirements" |
| **Method** | "Cosine annealing schedule" | "A/B test with 95% confidence" | "Evidence triangulation framework" |

To customize the schema (add columns, tables), edit `packages/db/src/schema.ts` and run `bunx drizzle-kit push` again.

### Tech stack

- **Runtime**: [Bun](https://bun.sh)
- **CLI framework**: [oclif](https://oclif.io)
- **Database**: [Turso](https://turso.tech) (remote SQLite via libsql)
- **ORM**: [Drizzle](https://orm.drizzle.team)
- **Search**: SQLite FTS5 (full-text search with triggers)

import { Command, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, references, referenceTags, tags } from "@experiment/db";

export default class RefAdd extends Command {
  static description = "Add a literature reference";

  static flags = {
    arxiv: Flags.string({ description: "arXiv ID (e.g. 2505.22934)" }),
    url: Flags.string({ description: "Paper/repo URL (defaults to HF paper URL if arxiv provided)" }),
    title: Flags.string({ description: "Reference title", required: true }),
    relevance: Flags.string({ description: "One-line relevance to our work", required: true }),
    "local-path": Flags.string({ description: "Local path to code/repo" }),
    "repro-steps": Flags.string({ description: "How to reproduce/clone" }),
    tag: Flags.string({ description: "Tag (repeatable)", multiple: true }),
  };

  async run() {
    const { flags } = await this.parse(RefAdd);

    const url = flags.url ?? (flags.arxiv ? `https://huggingface.co/papers/${flags.arxiv}` : null);
    if (!url) {
      this.error("Either --url or --arxiv must be provided");
    }

    const result = await db
      .insert(references)
      .values({
        arxivId: flags.arxiv ?? null,
        url,
        title: flags.title,
        relevance: flags.relevance,
        localPath: flags["local-path"] ?? null,
        reproSteps: flags["repro-steps"] ?? null,
      })
      .run();

    const refId = Number(result.lastInsertRowid);

    // Add tags
    if (flags.tag) {
      for (const tagName of flags.tag) {
        const clean = tagName.trim().toLowerCase();
        await db.insert(tags).values({ name: clean }).onConflictDoNothing().run();
        const tag = await db.select().from(tags).where(eq(tags.name, clean)).get();
        if (tag) {
          await db.insert(referenceTags)
            .values({ referenceId: refId, tagId: tag.id })
            .onConflictDoNothing()
            .run();
        }
      }
    }

    this.log(`Added reference #${refId}: ${flags.title} (${flags.arxiv ?? url})`);
  }
}

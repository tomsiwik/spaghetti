# Analyst hat

## Purpose
Read the completed paper and review, then write compact `LEARNINGS.md` notes with implications for the next experiment.

## Rules
- **Never wait for user input.**
- Quick pass only. Max 10 tool calls. Max 10 minutes.
- Do **not** spawn sub-agents.
- Do **not** create or modify entries in `.ralph/agent/memories.md`. The memories file is frozen — only a human can add to it.
- Do **not** write taxonomies, sub-classifications, or antipattern catalogs.
- **Single-pass.** This hat reads two files (PAPER.md, REVIEW-adversarial.md) and writes one (LEARNINGS.md). If either input is missing, exit immediately — don't retry, don't poll. Wait for the next activation.

## Workflow

1. Use the triggering event payload to identify which experiment to analyze.

2. Read: `PAPER.md`, `REVIEW-adversarial.md`.

3. Write `LEARNINGS.md` in at most 30 lines:
   - Core Finding (1-2 sentences)
   - Why (1-2 sentences)
   - Implication for Next Experiment (1-2 sentences)

4. If the experiment was killed and a paper explains why: `experiment ref-add`.

5. Emit `learning.complete` with experiment id + one-line summary.

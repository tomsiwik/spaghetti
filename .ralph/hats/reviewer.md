# Reviewer hat

## Purpose
Review `MATH.md`, `PAPER.md`, and `results.json` directly. Write `REVIEW-adversarial.md` and route the loop with a compact event.

## Context discipline
- **Never wait for user input.** Make the call and move on.
- Review directly. Do **not** spawn sub-agents.
- Max 20 tool calls. Max 15 minutes.
- **Doom-loop self-check.** Run `python .ralph/tools/doom_loop.py` first. If it exits non-zero, break the cycle.
- **Don't trigger before results land.** If `results.json` doesn't exist or the experiment's pueue task is still running, this hat should not have been activated. Emit `review.deferred` and exit — don't spam reads on incomplete artifacts.

## Workflow

1. Use the triggering `experiment.done` payload to identify which experiment to review.

2. Read: `MATH.md`, `PAPER.md`, `results.json`.

3. **Adversarial checklist** — any failure in (a)-(m) blocks PROCEED:

   **Consistency:**
   - (a) `results.json["verdict"]` matches proposed DB status
   - (b) `results.json["all_pass"]` matches claim
   - (c) PAPER.md verdict line matches DB status
   - (d) `is_smoke: true` → status must be `provisional`, never `supported`

   **KC integrity:**
   - (e) KC was not modified after first run (check git diff)
   - (f) No tautological KC (algebraic identity, single-adapter "composition", unused args)
   - (g) Code measures same quantity as MATH.md describes

   **Code bugs:**
   - (h) Composition math: grep for independent lora_A/lora_B summation → buggy
   - (i) LORA_SCALE ≥ 12 hardcoded → unsafe
   - (j) Routing on single sample applied to all → tautological
   - (k) `shutil.copy` of sibling adapter as new domain → fake
   - (l) Hardcoded `{"pass": True}` → fabricated
   - (m) Model in MATH.md ≠ model loaded in code → proxy substitution

   **Non-blocking flags:**
   - (n) Base accuracy 0% with no thinking chars → truncated eval
   - (o) n < 15 → stats warning
   - (p) Every experiment needs at least one target-metric KC (not just proxy)

4. Write `REVIEW-adversarial.md` — max 1 page. Verdict: `PROCEED`, `REVISE`, `KILL`, or `PROVISIONAL`.

5. Route:
   - **PROCEED**: `experiment finding-add`, emit `review.proceed`
   - **REVISE**: emit `review.revise` with ≤3 numbered fixes
   - **KILL**: `experiment complete --status killed`, `experiment finding-add`, emit `review.killed`
   - **PROVISIONAL**: `experiment update <id> --status provisional`, `experiment finding-add --status provisional`, emit `review.proceed` with `PROVISIONAL:` prefix

## REVISE discipline
- Max 3 blocking fixes per cycle. Max 2 rounds. On round 3, proceed with caveats.

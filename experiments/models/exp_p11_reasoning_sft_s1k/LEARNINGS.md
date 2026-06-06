# LEARNINGS: exp_p11_reasoning_sft_s1k

## Audit-2026-04-17 re-classification (2026-04-18)

**Status**: KILLED — structural antipattern (tags: `audit-2026-04-17-rerun`, `code-bug`).
**Re-run**: not executed. The code-bug fix (strip_thinking channel-aware, now applied)
only corrects the base-eval corruption; the adapter's 36.1% measurement is unchanged
and the catastrophic-forgetting finding is preserved. See REVIEW-adversarial.md §Round 2
for the full audit decision tree.

**Additional antipatterns surfaced by the audit (beyond the Round-1 docs-errors)**:
1. Training format mismatch: `<think>...</think>` literal text fed to mlx_lm.lora
   rather than Gemma 4 thinking channel tokens. Same class as
   `exp_p11_grpo_reasoning_adapter` evidence. The adapter learned to imitate
   scaffolding text, not to use the channel.
2. K1492 ("thinking NOT suppressed, >0 chars") is a **false pass** — the 1641
   avg_thinking_chars were literal `<think>` text, not real channel usage. The
   KC measures the wrong object (antipattern #6).
3. K1491 GSM8K is INVALID (HTTP 422), not FAIL.

**v2 requirements** (if anyone reopens this line): (a) train on data with real
Gemma 4 thinking channel tokens via `apply_chat_template(..., enable_thinking=True)`
round-tripped through tokenizer before writing JSONL, not literal `<think>` text;
(b) diverse traces spanning all MMLU-Pro categories, NOT math-only (this is the
original structural finding, still binding); (c) GSM8K fetch via `datasets`
library, not datasets-server rows API; (d) K1492 must measure channel engagement
(token-id presence), not character count of surface text.

**Cross-ref**: Finding #538 (s1K catastrophic forgetting), Finding #536 (62.1%
real base), Finding #587 (strip_thinking brittleness, same fix cluster),
downstream LIMO preemptively killed citing this experiment's impossibility
structure.

---

## Core Finding
Competition math SFT (s1K, 1000 steps, 27 examples) caused catastrophic forgetting on MMLU-Pro:
adapter dropped from 62.1% base to 36.1% (−26pp). Thinking was preserved (1641 chars/q).

## Why
s1K traces are near-orthogonal to MMLU-Pro's token distribution in embedding space. Gradient
descent on math-only traces pushes the model away from general reasoning breadth. This is not
a hyperparameter problem — it's a distributional mismatch at the trace level (arXiv:2502.03387
shows LIMO quality > quantity, but subject diversity is not covered). The math category itself
only scored 20%, ruling out "domain gain + forgetting" — it's pure degradation.

## Data Correction (from adversarial review)
Per-category table in PAPER.md has two wrong rows (base eval bug contaminated Phase 4a):
- biology: actual adapted = ~50%, not 10%
- computer science: actual adapted = ~40%, not 5%
K1491 should be labeled INVALID (HTTP 422 = untestable), not FAIL.

## Implications for LIMO (P11.A1, pueue task 5)
GAIR/LIMO is harder olympiad competition math — expect similar or worse degradation.
If math category scores <25% with LIMO adapter, kill immediately without full eval.
LIMO's training format uses `<think>...</think>` which may not match Gemma 4's actual
`<|channel>thought...<channel|>` tokens — check avg_thinking_chars in Phase 3 eval.

## Impossibility Structure
SFT preserves MMLU-Pro accuracy ONLY IF training traces span MMLU-Pro's 14-category
token distribution. Single-domain traces (math only) are structurally insufficient.
Required fix: diverse traces across all MMLU-Pro categories, not math-only datasets.

# LEARNINGS — `exp_model_knowledge_gap_26b_base`

**Verdict.** PROVISIONAL (BLOCKED-on-resource, macro-scope design-only). F#768 registered.

## Core Finding

A proof-first kill prior (F#478 monotonic extension via Kaplan 2020 / Hoffmann 2022) is *strongly* predictive of failure on Gemma 4 26B-A4B but is **not** a strict proof, because the model is MoE and the §3.2 expert-routing niche mechanism (Fedus 2022, Zhou 2022) creates a non-monotonic per-domain effective capacity `M_eff(d)`. The right action is PROVISIONAL with an explicit unblock path, not KILLED.

## Why

1. **Cache absent.** `mlx-community/gemma-4-26b-a4b-it-4bit` is not in `~/.cache/huggingface/hub/`. ~14 GB download + ~2.5 h training × 3 domains exceeds single-iteration researcher budget (guardrail 1009).
2. **Refused silent proxy.** Scaffold raises `NotImplementedError` rather than substitute 4B/E4B (researcher antipattern 'm'). Correct.
3. **F#666 preserved pre-run.** K1702 (proxy: ≥5pp MMLU-Pro) paired with K1816 (target: win-rate ≥60% N=30) — added pre-run as numeric supplier for the vague K1703, not a post-failure swap.
4. **Doom-loop broken.** Prior researcher iter RELEASED-TO-OPEN; this iter escalated to PROVISIONAL via reviewer = structurally different action.

## Implications for Next Experiment

1. **Block on routing-distribution measurement first.** Before re-attempting any 26B-A4B adapter sweep, run a separate experiment that measures `|⋃ E_d|` (expert union size, top-k=2 routing) on N=100 in-domain tokens for each candidate domain `d ∈ {code, math, medical, legal, finance}`. Cheap (forward-only, no training).
2. **Decision rule (pre-registered):** if any `d` has narrow routing (`|⋃ E_d| ≤ 2` of 16 experts), re-open with a *single-domain* focused run on that `d` only — not a 5-domain sweep. If no `d` routes narrowly, upgrade this experiment to KILLED on §3.1 grounds; no compute needed.
3. **Macro-scope guardrail.** PLAN.md "M5 Pro 48 GB IS macro scale" applies — do NOT plan a 5-domain × 500-step training sweep on a 14 GB MoE base from a single drain iteration. Macro-scope work needs an explicit `_impl` companion at P=3 with checkpointing across iterations.
4. **F#768 anchors the pattern.** Future "BLOCKED-on-resource + proof-first kill prior + non-trivial paper-grounded escape mechanism" filings should cite F#768 and follow the same PROVISIONAL → unblock-experiment → re-open-or-upgrade-to-KILLED path. Saves repeated reviewer deliberation.

## No new antipattern memory

REVIEW-adversarial flagged no recurring process bug. The filing is clean: §0 skills cited, graceful-failure `main()`, prediction-vs-measurement table all-untested, KCs unmodified, no scope-changing fix. Nothing to add to `.ralph/agent/memories.md`.

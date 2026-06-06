# LEARNINGS — exp_g4_quantization_aware_adapter_training

**Verdict:** KILLED (preempt-structural)
**Form:** F#666-pure-standalone — triple-fire (F#666 + F#502/F#646 schema + predicate-not-met)
**Finding-add:** SKIPPED per F#769 ledger-explosion closing-note

## Core Finding

QAT-LoRA on Gemma-4 is a legitimate scientific question that was filed with a structurally undecidable KC pair: K1920 (PPL gap < 0.05) is a pure proxy and K1921 (wall-clock ratio > 2×) is an engineering budget gate. With no target-metric KC, neither KILL nor SUPPORT is honestly reachable per guardrail 1007. The prior researcher's release notes (frozen 2026-04-25) further conditioned re-attempt on two unresolved predicates: locked citation (LoftQ-vs-arxiv:2310.08659 still suggested, not selected) and locked STE composition mechanism for `mlx.QuantizedLinear` (no native grad in MLX 0.31).

## Why

Three independent blockers, any one sufficient (F#666-pure + F#502/F#646 schema + predicate-not-met). Per F#769 closing-note, when every blocker is an established closed cohort, no new per-instance finding is filed; reviewer cites cohort evidence. Doom-loop guard satisfied: 2 consecutive KILLs but on structurally distinct mechanisms (PROD-cascade-category-error vs. KC-pairing-violation), no A→B→A→B alternation.

## Implications for Next Experiment

1. **Re-spec required, not retry**: any QAT-adapter follow-up must arrive with (a) paired target-metric KC (e.g., MMLU-Pro / HumanEval / GSM8K accuracy gap < 1pp), (b) locked citation in DB `references` array, (c) one-page STE-MLX mechanism spec from `/mlx-dev`. New experiment ID — not reopen.
2. **Drain accounting**: −1 from open queue (this was P=4, outside drain scope; converts a prior deferral to a definitive KILL).
3. **Avoid for next claim**: 3rd-consec preempt-KILL on resource/spec-blocked P≥3 (this is 2nd consec); 13th F#502/F#646 hygiene; 6th PROD super-family (closed at 4); cohort-deferred 3 hedgehog `_impl` siblings (JS/Python/Rust @ P=3) until `cache:26b-gemma4-teacher` lands; 8th Hedgehog ablation; 14th g4-ablation; 6th MEMENTO; 2nd hash-primitive; 5th cos-sim; 2nd argmax-divergence; 3rd ap-017(s).
4. **No new antipattern memory**: F#666-pure-standalone, F#502/F#646 schema, F#769 ledger-explosion are all already-injected memories. No NEW recurring process bug surfaced.

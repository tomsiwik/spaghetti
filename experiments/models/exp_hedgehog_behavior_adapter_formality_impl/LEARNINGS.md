# LEARNINGS — exp_hedgehog_behavior_adapter_formality_impl

## Core Finding (F#786 ratified)
Hedgehog per-layer cos-sim distillation generalizes to a 2nd behavior axis (formality) on Gemma 4 E4B 4-bit. Phase B loss converged 0.155→0.034 (5.6× reduction in 30 steps); proxy cos-sim 0.9614 across 42 layers. Heuristic Δ=+6.42 pp (base 45.16 → student 51.58, n=8) — under +10 pp threshold, capped by 3rd-instance K2 thinking-mode truncation antipattern, not adapter-null.

## Why
- Phase B/C training signal landed end-to-end → adapter is real, not scaffolding.
- K#1963 heuristic Δ scored on `<|channel>thought` preamble (256-tok cap), not the answer text. Sample snippets show both base and student stuck at 256 with thinking-process emit.
- F#786 = cluster-extension PROVISIONAL parallel to F#783 (politeness) + F#784 (refactor); same training+heuristic dynamics, same K2 ceiling.

## 3rd-instance antipattern PROMOTED
`mem-antipattern-thinking-mode-truncates-judge-budget` written this iter (politeness F#783 + refactor F#784 + formality F#786). Mitigations for `_full`: (1) `enable_thinking=False`, (2) stop_token=`<|channel>final` resume, (3) `max_tokens=800`, (4) Claude API judge with full 800-tok completion. Pick (1) OR (2) AND (4) per F#786 PAPER.md routing.

## Implications for Next Experiment
- HALT D-cascade D.3 partially exhausted (formality_impl ✓, conciseness_impl pending). The 3rd K2-collapse confirmed; further `_impl` runs add no antipattern signal.
- Top researcher pick: `exp_hedgehog_behavior_adapter_conciseness_impl` (P=1 macro, 4th HALT D.3 entry — would yield 4th-instance K2-collapse confirmation but antipattern already promoted, marginal). Alternative: claim a `_full` follow-up if pueue ANTHROPIC_API_KEY env-var pattern verifies (mem-1777104328-c05a documents pueue env-var workaround for SMOKE_TEST; same single-string form should propagate ANTHROPIC_API_KEY).
- 2nd researcher pick: `exp_hedgehog_pair_composition_class_composition_impl` (P=2 macro, novel composition axis, would test whether Hedgehog adapters compose).
- AVOID: 5th-iter `_impl` Hedgehog axis (saturating).

## Drain accounting
- P≤2 open: 9 (formality_full added by reviewer iter ~68). Active: 0.
- Finding-ledger: 47 (F#786). 5th consecutive non-preempt iter pending researcher claim.

## Routing
Emit `learning.complete` with payload: id + result summary + next-claim hint (conciseness_impl OR class_composition_impl).

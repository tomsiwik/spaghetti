# LEARNINGS — exp_hedgehog_behavior_adapter_conciseness_full (smoke iter)

## Core Finding (F#790, provisional)
Smoke gate (MATH.md §9) caught a NEW-code MMLU harness bug in 122s before a 3-5h pueue full-run produced degenerate output. K#1965 deterministic length-reduction PASS at 57.76% lower-bound (base 8/8 still capped at max_tokens=512). K#1966 MMLU drop unmeasured — base_acc=0.15 (below chance) because Gemma 4 IT thinking-mode prefix `<|channel>thought\n…` causes naive first-letter scan to extract the **C** in **CHANNEL** with `MMLU_GEN_MAX_TOKENS=4`. Verdict PROVISIONAL (smoke + degenerate-flag A4); no F#666 bilateral fail → no KILL.

## Why
- K#1965 reads token counts (deterministic, no judge) — escapes the K2-heuristic-collapse antipattern that capped politeness/refactor/formality at heuristic_only. Real positive signal even with base censoring.
- K#1966 is a multiple-choice harness; thinking-mode prefix + tiny gen budget + greedy first-letter parser interact pathologically. Both base and adapter return "C" → drop=0.0pp meaninglessly.
- Smoke gate is the structural defense: small-N MMLU run flagged base_acc<0.5 (pre-reg A4) before any compute waste.

## Implications for Next Experiment
1. **v2 substrate**: current `_full` task remains; do NOT file new task. Researcher next iter applies PAPER.md §7 fix list to run_experiment.py:
   - A9 fix: `enable_thinking=False` in `apply_chat_template`, OR `MMLU_GEN_MAX_TOKENS≥256` + parse for `<|channel>final` marker, OR explicit "single letter only" system prompt.
   - A10 fix: `GEN_MAX_TOKENS=2048` (smoke 512 still censored 8/8 base).
   - Re-run smoke first; verify base_acc∈[0.40,0.70] AND `base_capped_count<0.5×n` BEFORE submitting full pueue.
2. **Drain alternative**: 6 P≤2 open (memento_replication, class_composition_full_impl, triple_composition, politeness_full, refactor_full, formality_full). Top picks for variety: `exp_hedgehog_triple_composition_3domain` (P=2 micro, only non-Hedgehog-_full P=2 micro available); `exp_memento_gemma4_replication_impl` (P=1 macro, distinct from Hedgehog cluster). Avoid 4th identical Hedgehog _full smoke (politeness/refactor/formality) until v2 harness is fixed in conciseness_full and ported — same MMLU bug will recur in all three.
3. **Antipattern promotion (filed this iter)**: `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` — distinct mechanism from existing `mem-antipattern-thinking-mode-truncates-judge-budget` (judge-budget is heuristic-on-register; this is multiple-choice first-letter-scan). Will auto-inject into all hat activations.
4. **Positive structural pattern flagged**: smoke-gate-validation-works memory NOT created — single instance, would be premature. Wait for 2nd structurally-distinct smoke-catch (different bug class) before promoting to antipattern memory.
5. **Linear_to_lora_layers shim**: 5th recurrence; existing memory `mem-antipattern-linear-to-lora-layers-shim-recurrence` updated by reinforcement, no new memory needed.

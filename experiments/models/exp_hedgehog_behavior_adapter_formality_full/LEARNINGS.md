# LEARNINGS.md — exp_hedgehog_behavior_adapter_formality_full (smoke)

**Verdict:** PROVISIONAL (smoke; F#798 ratified). All 25 adversarial PASS, smoke gate 5/5 PASS.

## Core Finding

Formality_full smoke validates the established Hedgehog _full pattern on a third behavior axis. Phase B converges 3.48× (0.1011→0.0291); proxy cos=0.9679; K#2013 heuristic Δ=+9.09pp (just below +10pp threshold, +2.67pp lift over F#786 _impl under default thinking-mode); K#2014 smoke MMLU N=20 shows -25pp drop (75%→50%). Adapter persisted at `adapters/hedgehog_formal_r8_full/` for v2 re-runs without retraining.

## Why

1. **3rd cross-exp port of `enable_thinking=False` mitigation** (politeness_full F#794 → refactor_full F#797 → formality_full F#798). Behavior+procedural+behavior axes covered. Mitigation is now established across the Hedgehog _full smoke trio — `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` mitigation list is mature.
2. **1st cross-exp validation of F#795 smoke-N MMLU variance methodology rule.** Politeness_full v2 precedent: smoke -25pp → full -6pp benign N-variance. This iter shows the same -25pp pattern at smoke N=20 → predicted ~-6pp at full N=100. Smoke-gate correctly does NOT block full-submission. F#795 is now 1-instance cross-validated; promotes to formal rule on 2nd full-N disambiguation.
3. **K#2013 heuristic ceiling Mode-1 (preamble truncation absence).** Unlike refactor_full F#797 Mode-2 (ceiling-saturation), formality smoke shows real differential signal (base 50.44 → student 59.53), just below threshold. heuristic_only carve-out (F#783/F#784) means no kill binding without API key.
4. **2nd post-promotion honor of `mem-antipattern-researcher-prefiles-finding-before-review` gate.** 1st was refactor_full iter ~100 cluster; this iter is 2nd observance — antipattern fix appears stable.

## Implications for Next Experiment

- **TOP PICK:** `exp_memento_gemma4_replication_impl` (P=1 macro) — non-Hedgehog axis variety; 8+ consecutive Hedgehog smokes is over-saturated. Phase A design-only is single-iter-feasible per `mem-antipattern-novel-mechanism-single-iteration-scope`.
- **2nd PICK:** `exp_g4_adapter_class_composition_full_impl` (P=1 macro) — non-Hedgehog axis variety; structurally distinct from any Hedgehog _full experiment.
- **AVOID:** Triple Hedgehog _full v2 LORA_SCALE sweeps — F#795 falsifies LORA_SCALE-as-cause hypothesis. v2 needs are: (a) full-N pueue submission for K#2014 corroboration (3-5h on existing adapter, no retraining), (b) `ANTHROPIC_API_KEY` for K#2013 binding (5-10 min Phase C re-run on existing adapter).
- **No new tasks filed** — current dir is v2 substrate; mirrors politeness_full + refactor_full + conciseness_full pattern.
- **Drain status:** P≤2 open=2 (memento_replication_impl, class_composition_full_impl); active=0; finding-ledger=57. Hedgehog _full smoke trio (politeness/refactor/formality) all smoke-validated; Hedgehog axis is fully exhausted at smoke-grade.

## Antipattern updates

- `mem-antipattern-thinking-mode-truncates-judge-budget` already covers Mode-1 (preamble truncation, formality_impl F#786) + Mode-2 (ceiling saturation, refactor_full F#797). This iter reinforces Mode-1 with a non-saturated differential signal (50→59 not 10→10). No memory file change needed.
- `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` mitigation list reinforced 3× (politeness/refactor/formality). No file change needed; pattern is mature.
- `mem-antipattern-researcher-prefiles-finding-before-review` 2nd post-promotion observance — pattern fix confirmed stable. No file change needed.

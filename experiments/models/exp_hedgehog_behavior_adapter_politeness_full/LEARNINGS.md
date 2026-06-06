# LEARNINGS.md — exp_hedgehog_behavior_adapter_politeness_full

**Status:** PROVISIONAL · F#796 (reviewer canonical, exp-scope, full-N) · F#795 (researcher, supported, methodology-scope) · supersedes smoke F#793/F#794

## Core Finding
At full N=100 on the SAME adapter, K#2000 cos=0.9943 PASS and K#2002a MMLU drop=−6pp PASS (adapter IMPROVES MMLU 61→67). This clears the F#666 verdict-matrix tautological-proxy gate (PASS, PASS quadrant). The smoke iter ~98 K#2002a 25pp drop signal that drove F#793/F#794 was a small-N variance artifact: 1/100=1pp granularity at full vs 5/20=25pp at smoke. K#2001 stays `heuristic_only` at Δ=+15.72pp (n=50; smoke +18.5pp at n=8 was a 3pp small-N over-count). Mapped to `provisional` per researcher.md §6 #3 because PAPER.md verdict reads "PARTIALLY_SUPPORTED".

## Why
1. **Full-N disambiguates smoke F#666 candidates.** The −25pp→+6pp sign reversal on the same adapter+harness is consistent with Wilson-interval N=20 single-flip bound (±10pp at p≈0.7); F#795 codifies "no F#666 KILL on smoke-N MMLU drop without N≥50 corroboration" as methodology rule. 1st structurally-distinct full-N disambiguation in this repo.
2. **F#666 verdict matrix pre-registered for all 4 (K#2000 × K#2002a) quadrants** in MATH.md §4. Outcome (PASS, PASS) → "non-degenerate-but-bounded" (clears KILL clause). 1st quadrant-explicit pre-reg matrix in repo — positive structural pattern.
3. **`enable_thinking=False` mitigation HOLDS at full N.** A3 base_acc=0.61 (vs 0.75 at smoke) is mean-regression to MMLU population rate, not regression to degenerate channel-prefix output (4/4 distinct base letters preserved). 1st full-N cross-experiment validation of `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` mitigation; 2nd validation overall (1st was conciseness_full smoke).
4. **F4' falsifies the `LORA_SCALE=6.0` aggression hypothesis** (smoke F2 candidate). Same scale at full N IMPROVES MMLU. The "cross-task attention leakage" hypothesis is also falsified. v2 LORA_SCALE sweep is no longer load-bearing for K#2002a.

## Literature anchor
Hedgehog distillation methodology — Wang et al. 2024 "The Hedgehog & The Porcupine: Expressive Linear Attentions with Softmax Mimicry" (cf MATH.md §1 lemma derivation). F#666 target-gating discipline is specific to this repo (no upstream lit equivalent). Wilson-interval N-variance bound is textbook (Agresti & Coull 1998); the application here — flagging single-question MMLU flips as KILL signal at small N — is the antipattern F#795 mitigates.

## Implications for Next Experiment
- **AVOID guidance LIFTED:** F#795 unblocks `refactor_full` and `formality_full` _full smokes. Smoke-N MMLU drops are now known to be N-variance and should NOT trigger F#666 KILL preemptively.
- **TOP PICK:** `exp_memento_gemma4_replication_impl` (P=1 macro, non-Hedgehog axis variety) — breaks 8-iter Hedgehog cluster monoculture; analyst's prior 2nd-pick.
- **SECOND PICK:** `exp_hedgehog_behavior_adapter_refactor_full` (port full-N rerun pattern; reuse adapter checkpoint methodology). Smoke-gate now validated at full-N for one Hedgehog _full member; precedent supports porting.
- **THIRD PICK:** `exp_hedgehog_behavior_adapter_formality_full` (same port pattern). Defer until refactor_full validates the port outside politeness.
- **NOT NEEDED:** v2 LORA_SCALE sweep (F4' falsifies); v2 only requires `ANTHROPIC_API_KEY` for K#2001 PASS/FAIL binding (5-10 min Phase C re-run on existing adapter).

## Antipattern signals (memory implications)
1. **`mem-antipattern-researcher-prefiles-finding-before-review` reaches 2nd instance** (1st: F#793 smoke; 2nd: F#795 full-N). Per memory's own "Promote if 2nd instance observed" trigger, the recommended fix should be elevated: researcher.md template should add an explicit "DO NOT call finding-add before emitting experiment.done" gate. Memory updated this iter with 2nd-instance line + escalation.
2. **Positive pattern:** 8th `linear_to_lora_layers` shim PRE-EMPTED — 1st prophylactic at full mode (smoke iter ~98 was 1st prophylactic overall). Existing memory unchanged.
3. **Positive pattern:** F#666 verdict-matrix pre-registration in MATH.md §4 (quadrant-explicit) is novel — worth tracking as a positive structural pattern; no memory yet (1 instance, await structurally-distinct 2nd before promoting).

## Drain accounting
P≤2 open: 4 (memento_replication_impl, class_composition_full_impl, refactor_full, formality_full). Active: 0. Finding-ledger: 55 (F#796 latest).

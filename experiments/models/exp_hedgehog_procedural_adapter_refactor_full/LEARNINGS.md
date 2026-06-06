# LEARNINGS.md — exp_hedgehog_procedural_adapter_refactor_full

**Verdict: PROVISIONAL** (smoke + structural-KC PASS + target-KC heuristic_only + 3 KCs deferred to v2)

Drain-window iter ~102 analyst pass. Ratifies F#797 (canonical reviewer-attributed). Researcher honored finding-add gate (1st explicit observance of `mem-antipattern-researcher-prefiles-finding-before-review` post-promotion).

---

## Core Finding

K#2004 mean per-layer cos=0.9776 PASS (n=8, worst-layer 0.9341). K#2005 heuristic Δ=0.0 ceiling-saturated (Mode-2 NEW). 3 KCs (K#2006/2007/2008) deferred to v2 per pre-reg MATH.md §3 (HumanEval, non-refactor curated, NEUTRAL ablation). Smoke gate ALL 5 PASS — full submission unblocked. 2nd cross-exp port of `enable_thinking=False` mitigation VALIDATED (politeness_full → refactor_full); spans behavior + procedural axes.

## Why

Hedgehog cos-sim distillation lands cleanly at smoke (Phase B 2.55× loss reduction in 30 steps; A1 gate PASS). K#2005 heuristic regex (refactor markers `def`/`class`/code-block boundaries) saturates at ceiling=10.0 because `enable_thinking=False` lets BOTH base AND student emit verbose multi-section refactor explanations under N=6 — not preamble truncation (Mode-1 in F#783/F#784/F#786) but ceiling saturation where heuristic cannot distinguish the two outputs because both contain the lexical markers it counts.

This is a **NEW K2-collapse mode** distinct from Mode-1: the same antipattern memory now has two failure surfaces. Both still require Claude API binding for true K2 verdict; smoke heuristic remains informative-only.

## Implications for Next Experiment

1. **Antipattern memory upgraded** — `mem-antipattern-thinking-mode-truncates-judge-budget` now documents two collapse modes (Mode-1 preamble truncation, Mode-2 ceiling saturation). Both modes cap verdict at PROVISIONAL absent API; mitigation list still valid; carve-out still applies.

2. **Adapter checkpoint preserved** at `adapters/hedgehog_refactor_r8/` for v2 reuse without retraining. v2 unblock list (PAPER.md §7) gives priority order: API key → token-LoRA baseline → HumanEval → curated non-refactor → NEUTRAL → 26B teacher → full-N.

3. **Drain status (verified):** P≤2 open=3 (memento_replication_impl, class_composition_full_impl, formality_full); active=0; ledger=56. F#795 lifted analyst's prior AVOID guidance on formality_full smoke (smoke-N MMLU drops disambiguated as N-variance); refactor_full smoke ports the pattern cleanly.

4. **Next-pick recommendation (researcher):**
   - **TOP PICK:** `exp_hedgehog_behavior_adapter_formality_full` — port the now-twice-validated `enable_thinking=False` + smoke-gate pattern to the last open Hedgehog `_full` axis (formality). Single-iter feasible (smoke-only, ~60-90s pueue). Closes the Hedgehog `_full` smoke trio (politeness/refactor/formality).
   - **2nd PICK:** `exp_g4_adapter_class_composition_full_impl` — non-Hedgehog axis variety (P=1 macro). Single-iter feasibility uncertain (composition multi-adapter scope); apply `mem-antipattern-novel-mechanism-single-iteration-scope` carve-out (Phase A design-only, file PROVISIONAL with Phase B/C deferred).
   - **3rd PICK / AVOID for now:** `exp_memento_gemma4_replication_impl` — P=1 macro novel mechanism, 6-10h budget; multi-iter; defer until Hedgehog _full smoke trio is closed.
   - **Researcher gate:** do NOT pre-file the F#-finding; reviewer files canonical post-adversarial-pass per `mem-antipattern-researcher-prefiles-finding-before-review` (now a hard rule, 2 instances + promotion observed in iter ~99-100).

## References

- F#797 (provisional, canonical reviewer-attributed) — refactor_full smoke + Mode-2 ceiling-saturation observation
- F#783 / F#784 / F#786 — Mode-1 preamble-truncation precedents
- F#794 / F#796 — politeness_full 1st cross-exp port
- F#795 — smoke-N MMLU variance disambiguation methodology
- `mem-antipattern-thinking-mode-truncates-judge-budget` (now Mode-1+Mode-2 dual-mode)
- `mem-antipattern-researcher-prefiles-finding-before-review` (1st gate observance this iter)
- `mem-antipattern-linear-to-lora-layers-shim-recurrence` (9th pre-emption)
- `mem-pattern-deterministic-proxy-escapes-k2-collapse` (still applies — refactor heuristic was NOT deterministic, hence Mode-2)

# LEARNINGS — exp_hedgehog_procedural_adapter_refactor_impl

## Core Finding (F#784, ratified)

PROVISIONAL smoke. K1 per-layer cos-sim PASS at 0.9706 (n=8 held-out, all 42 layers > 0.92, threshold > 0.80; predicted 0.83). K2 heuristic_only (max_tokens=192 truncated thinking-mode preamble; both arms scored length-floor 10.0, no signal). K3 (HumanEval) and K4 (non-refactor) explicitly deferred to `_full` with blockers logged. Cluster-extension to F#783 (politeness_impl).

## Why

K1 = 0.97 reflects the *smoke shortcut*: same-arch teacher/student with scale-toggle (E4B catalog-prompt scale=0 vs E4B neutral scale=6.0). Easier regime than the canonical 26B-teacher comparison from MATH.md §0 (0.83 prediction was for 26B teacher). The high cos-sim is consistent with the LoRA absorbing the catalog-prompt routing perturbation but does NOT yet validate the procedural-knowledge transfer claim against a stronger teacher. K2 collapsed because thinking-mode preamble exceeds 192-token budget; this is the **2nd consecutive Hedgehog smoke** to hit the truncation trap (politeness_impl was the 1st, F#783) — one more recurrence promotes a `mem-antipattern-thinking-mode-truncates-judge-budget` memory.

## Implications for Next Experiment

1. **HALT-override pattern validated** (2nd consecutive: politeness_impl + refactor_impl both yielded real PROVISIONAL via pueue+smoke). Researcher should not HALT on the next P=1/P=2 Hedgehog _impl; coordinator-override is reliable.
2. **Next-claim recommendation**: `exp_rdt_loop_kv_cache_impl` (P=1 micro, parallel-independent, NO F#683-cluster dependency — provides variety + tests HALT-override on a non-Hedgehog axis). After that, `exp_hedgehog_behavior_adapter_formality_impl` (P=1 macro, same proven pueue+smoke template; expect K1 PASS, K2 likely-collapse → 3rd-instance promotes thinking-mode-truncation antipattern memory).
3. **For the `_full` follow-on**: (a) export `ANTHROPIC_API_KEY` into pueue env for paired Claude judge; (b) raise `max_tokens` to ~512 OR set `enable_thinking=False` for K2-judge generations only (preserve thinking for K1 attn capture); (c) wire 26B teacher with sequential-phase residency on 48GB M5 Pro (same memory-discipline pattern as `exp_g4_zs_base_transfer_4bit_fp16_full`); (d) add token-space LoRA baseline at matched rank for K2 head-to-head per MATH.md §4.
4. **Sizing-bug operational lesson** (NOT a finding, NOT antipattern-recurrence yet): when an IMPL embeds a small smoke list, assert `N_TRAIN + N_HELDOUT ≤ len(SMOKE_LIST)` at top of `run_experiment.py`. 1st observation only; threshold for finding is 2-3 recurrences.

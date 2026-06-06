# PAPER.md — exp_hedgehog_domain_adapter_js

## Verdict: PROVISIONAL — design-only, no empirical claim filed

Pre-registered KCs (K1790, K1791, K1792, K1793) all `untested`. Scaffold runs cleanly, writes structured blockers, no silent mechanism swap. Sibling precedent: F#683 (politeness) and F#684 (procedural-refactor) — both PROVISIONAL, same blocker structure, same `_impl` follow-up pattern.

## Predictions vs Measurements

| KC | Predicted | Measured | Status |
|----|-----------|----------|--------|
| K1790 (#1790): per-layer cos(teacher, student) > 0.80 | cos ≈ 0.83 (range [0.80, 0.88]) | not measured | untested (Phase B blocker) |
| K1791 (#1791): JS-bench vs token-space LoRA baseline ≥ 0 | Δ ∈ [0, +1.5] | not measured | untested (Phase B + Baseline blocker) |
| K1792 (#1792): HumanEval (Python) pass@1 drop < 3 pp | ≤ 2 pp | not measured | untested (Phase D blocker) |
| K1793 (#1793): MMLU subset drop < 2 pp | ≤ 1 pp | not measured | untested (Phase D blocker) |

## What actually ran

`run_experiment.py` executed in 1.6 s, wrote `results.json` with `verdict="PROVISIONAL"`, all four KCs recorded as `"untested"`, and 5 structured blockers enumerating which phases are `NotImplementedError` (Phase 0 corpus curation, Phase B Hedgehog training loop, Phase Baseline token-space LoRA for K1791 head-to-head, Phase C eval, Phase D eval). No silent substitution of cross-entropy SFT for cos-sim distillation (scope-preservation per antipattern-t).

## Why PROVISIONAL (honest scope call)

Per `mem-antipattern-claim-time-tag-saturation` + `mem-antipattern-novel-mechanism-single-iteration-scope`:

- The claim picker returned this experiment (tag: `hedgehog`, `domain-adapter`, `novel`).
- Hedgehog per-layer cos-sim distillation is NOT available through the `mlx_lm.lora` CLI. It requires a custom MLX training loop that:
  1. runs a 26B teacher forward pass with MDN + Eloquent-JS excerpts in context (peak memory load-bearing on 48 GB M5 Pro — sequential-phase eviction between teacher/student likely required),
  2. captures per-layer attention-output tensors from all 42 Gemma 4 E4B blocks (student side),
  3. computes per-layer cos-sim loss with stop-gradient on teacher side,
  4. steps via `nn.value_and_grad(student, loss_fn)` + `mlx.optimizers.AdamW` with `mx.eval` discipline and `mx.clear_cache()` between batches (F#673).
- Sibling `exp_hedgehog_behavior_adapter_politeness` (F#683, 2026-04-23) and `exp_hedgehog_procedural_adapter_refactor` (F#684, 2026-04-23) hit the same budget wall and filed PROVISIONAL with `_impl` follow-ups. JEPA sibling did the same. This is the 3rd Hedgehog-axis instance of the pattern — design-locked, implementation deferred.

Full pipeline realistic budget: ~4–6 h (two training jobs at 800 steps, 26B teacher residency, judge per held-out pair, HumanEval + MMLU subset). Exceeds researcher-hat cap (30 min / 40 tool calls, guardrail 1009).

## Measurement blockers (detail)

1. **Phase 0 (dataset):** MDN fair-use summaries + Eloquent-JS (CC-BY-NC 3.0) chapter summaries for 6 focus topics + LLM-generated (Q, A) pairs. Stratified split so held-out contains distinct *instances* of same concept categories (tests generalization, not memorization). Validation by spot-check (n=20) + LLM-judge.
2. **Phase A + B (training loop):** custom MLX loop as above. Cannot be done via `mlx_lm.lora` CLI (which runs cross-entropy on target tokens, not cos-sim on per-layer attention outputs).
3. **Phase Baseline:** token-space LoRA on same (Q, A) pairs — available via `mlx_lm.lora` CLI — but K1791 must be head-to-head paired with Phase B adapter, so deferred to `_impl`.
4. **Phase C (K1790 + K1791):** depends on Phase A/B and Baseline.
5. **Phase D (K1792 + K1793):** depends on Phase B adapter.

## Why not silently downscale (scope-preservation)

Tempting to: (a) proxy teacher to E4B → violates MATH.md §0 "No proxy substitution"; (b) swap cos-sim to SFT-CE → changes what K1790 measures (from attention-routing distillation to output-token imitation); (c) skip K1791 baseline → K1791 becomes unpaired, F#666 violation; (d) reduce N_STEPS to fit in iteration → changes KC predictions (Zhang 2024 convergence requires 800+ steps for stable cos-sim ≥ 0.80).

Each would be a silent scope reduction (reviewer antipattern (t)). PROVISIONAL is the honest classification.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue ✓
5. No KC was modified between MATH.md and now. K1790–K1793 match DB-registered text exactly ✓
6. Antipattern scan:
   - composition math bug — N/A (no composition at this layer)
   - tautological routing — N/A (no routing)
   - LORA_SCALE — 6.0 ≤ 8 per F#328/F#330 ✓
   - shutil.copy as new adapter — N/A (no adapter produced) ✓
   - hardcoded `pass: True` — no KCs marked PASS ✓
   - proxy-model substitution — scaffold explicitly refuses to proxy teacher to E4B; BLOCKED path emits PROVISIONAL ✓
   - eval-template truncation — N/A (no eval run) ✓
   - claim-time-tag-saturation — PROVISIONAL classification is precisely the documented remedy ✓
   - novel-mechanism-single-iteration-scope — design-locked, deferred to `_impl` (matches precedent) ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

Enumerated in MATH.md §8 (A1–A6). The load-bearing ones for this PROVISIONAL filing:
- **A5.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; a 4–6 h pipeline is out of scope.
- **A1.** 26B teacher not yet cached on this M5 Pro (per `~/.cache/huggingface/hub` inspection 2026-04-24). Teacher availability is an additional resource blocker the `_impl` follow-up must address.

## Follow-up path

Three independent paths that would unblock:

1. **Primary:** file a matching `_impl` experiment (sibling precedent: `exp_hedgehog_procedural_adapter_refactor_impl`, `exp_hedgehog_behavior_adapter_politeness_impl`) scheduled for a dedicated 4–6 h session. Invokes `/mlx-dev` + `/fast-mlx` before writing the training loop.
2. **Resource-readiness check:** pre-cache `mlx-community/gemma-4-26b-a4b-it-4bit` before the `_impl` session to avoid a 14 GB download mid-run.
3. **Teacher-alternatives study (optional, lower priority):** compare 26B teacher vs 8B teacher (if available) on the same (π_JS, Q) pairs — would inform whether cheaper teachers can seed the distillation with acceptable quality ceiling.

Composition with sibling adapters is blocked by F#688 (`exp_hedgehog_composition_polite_refactor_js` PREEMPT-KILLED) until all three parents (politeness, refactor, domain-js) are target-SUPPORTED simultaneously. This experiment reaching SUPPORTED is one of three independent unblock paths for the composition child.

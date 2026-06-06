# PAPER.md — exp_hedgehog_adapter_python_domain

## Verdict: PROVISIONAL — design-only, no empirical claim filed

Pre-registered KCs (K1844, K1845) all `untested`. Scaffold runs cleanly, writes structured blockers, no silent mechanism swap. Sibling precedents: F#683 (politeness), F#684 (procedural-refactor), F#696 (JS domain) — all PROVISIONAL, same blocker structure, same `_impl` follow-up pattern. This is the **4th Hedgehog-axis instance** of the pattern, and the **2nd axis-domain sibling** (JS, Python). Rust and SQL axis-domain remain in the OPEN queue; triple-composition gates on all three.

## Predictions vs Measurements

| KC | Kill condition | Predicted measurement | Measured | Status |
|----|----------------|------------------------|----------|--------|
| K1844 (#1844): Hedgehog PPL on Python-eval > base + generic LoRA | kill if Hedgehog strictly worse | PPL(Hedgehog) ≈ 0.95–1.02 × PPL(generic LoRA) — matched-or-slightly-better | not measured | untested (Phase B + Baseline blocker) |
| K1845 (#1845): idiomaticity judge Δ < +5 pp vs base | kill if Δ < +5 pp | Δ ∈ [+5, +10] pp, mean ≈ +7 pp | not measured | untested (Phase D blocker) |

F#666 target-gating: K1844 is proxy (PPL), K1845 is behavioral target (idiomaticity). SUPPORTED = both PASS; KILLED = both FAIL. Mixed outcomes route to finding-about-proxy (if target PASSes) or tautological-proxy-kill-on-target (if target FAILs).

## What actually ran

`run_experiment.py` executed in ~1–2 s, wrote `results.json` with `verdict="PROVISIONAL"`, both KCs recorded as `"untested"`, and ≥ 5 structured blockers enumerating which phases are `NotImplementedError` (Phase 0 corpus curation, Phase B Hedgehog training loop, Phase Baseline generic token-space LoRA for K1844 head-to-head, Phase C PPL eval, Phase D idiomaticity judge). No silent substitution of cross-entropy SFT for cos-sim distillation (scope-preservation per antipattern-t).

## Why PROVISIONAL (honest scope call)

Per `mem-antipattern-claim-time-tag-saturation` + `mem-antipattern-novel-mechanism-single-iteration-scope`:

- The claim picker returned this experiment (tags: `p1`, `hedgehog`, `domain`).
- Hedgehog per-layer cos-sim distillation is NOT available through the `mlx_lm.lora` CLI. It requires a custom MLX training loop that:
  1. runs a 26 B teacher forward pass with CPython-doc + PEP excerpts in context (peak memory load-bearing on 48 GB M5 Pro — sequential-phase eviction between teacher and student likely required),
  2. captures per-layer attention-output tensors from all 42 Gemma 4 E4B blocks (student side),
  3. computes per-layer cos-sim loss with stop-gradient on teacher side,
  4. steps via `nn.value_and_grad(student, loss_fn)` + `mlx.optimizers.AdamW` with `mx.eval` discipline and `mx.clear_cache()` between batches (F#673).
- Siblings `exp_hedgehog_behavior_adapter_politeness` (F#683), `exp_hedgehog_procedural_adapter_refactor` (F#684), `exp_hedgehog_domain_adapter_js` (F#696) all hit the same budget wall and filed PROVISIONAL with `_impl` follow-ups at P=3.

Full pipeline realistic budget: ~4–6 h (two training jobs at 800 steps, 26 B teacher residency, judge per held-out pair, PPL eval across 3 configs). Exceeds researcher-hat cap (30 min / 40 tool calls, guardrail 1009).

## Measurement blockers (detail)

1. **Phase 0 (dataset):** CPython fair-use summaries + PEP-8/20/257/484 canonical text for 7 focus topics + LLM-generated (Q, A) pairs + *external* PyPI PPL eval corpus (disjoint from CPython training docs to avoid train contamination).
2. **Phase A + B (training loop):** custom MLX loop as above. Cannot be done via `mlx_lm.lora` CLI (which runs cross-entropy on target tokens, not cos-sim on per-layer attention outputs).
3. **Phase Baseline:** generic token-space LoRA on same (Q, A) pairs — available via `mlx_lm.lora` CLI — but K1844 must be head-to-head paired with Phase B adapter, so deferred to `_impl`.
4. **Phase C (K1844):** PPL across three configs — depends on Phase B and Baseline.
5. **Phase D (K1845):** idiomaticity auto-judge on 50 blind-paired prompts — depends on Phase B adapter.

## Why not silently downscale (scope-preservation)

Tempting to: (a) proxy teacher to E4B → violates MATH.md §0 "No proxy substitution"; (b) swap cos-sim → CE SFT → changes what K1844 measures (PPL improvement via next-token imitation is a *different* hypothesis than PPL improvement via attention-routing distillation); (c) skip K1844 baseline → K1844 becomes unpaired (no head-to-head anchor against generic token-space LoRA), F#666 target-gating violation; (d) reduce N_STEPS to fit in iteration → changes KC predictions (Zhang 2024 convergence requires 800+ steps for stable cos-sim ≥ 0.80).

Each would be a silent scope reduction (reviewer antipattern (t)). PROVISIONAL is the honest classification.

## Scope notes (explicitly not KC)

Non-interference checks — JS HumanEval, Rust code accuracy, MMLU general NL — are **not registered KCs** for this experiment (only K1844 and K1845 are). The sibling JS experiment registered 4 KCs including non-interference + specificity; this one registered 2. Measuring non-interference post-hoc in the `_impl` would be an exploratory metric, not a KC — and would not be sufficient to change the verdict. Cross-axis interference (Python + JS + SQL) is scoped to the composition child `exp_hedgehog_triple_composition_3domain` (which is also in the OPEN queue but blocked on all three domain parents reaching SUPPORTED).

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue ✓
5. No KC was modified between MATH.md and now. K1844–K1845 match DB-registered text exactly ✓
6. Antipattern scan:
   - composition math bug — N/A (no composition at this layer)
   - tautological routing — N/A (no routing)
   - LORA_SCALE — 6.0 ≤ 8 per F#328/F#330 ✓
   - `shutil.copy` as new adapter — N/A (no adapter produced) ✓
   - hardcoded `pass: True` — no KCs marked PASS ✓
   - proxy-model substitution — scaffold explicitly refuses to proxy teacher to E4B; BLOCKED path emits PROVISIONAL ✓
   - eval-template truncation — N/A (no eval run) ✓
   - claim-time-tag-saturation — PROVISIONAL classification is precisely the documented remedy ✓
   - novel-mechanism-single-iteration-scope — design-locked, `_impl` filed this iteration (matches newly-formalized `mem-antipattern-impl-follow-up-delegation` remedy; no delegation) ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

Enumerated in MATH.md §8 (A1–A7). Load-bearing for this PROVISIONAL filing:
- **A5.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; a 4–6 h pipeline is out of scope.
- **A1.** 26 B teacher not yet cached on this M5 Pro (per JS-sibling PAPER.md A1 and `exp_model_knowledge_gap_26b_base` resource blocker). Teacher availability is an additional resource blocker the `_impl` follow-up must address.
- **A7.** Only 2 KCs are pre-registered (vs 4 for JS sibling). Non-interference is explicitly NOT gated here; post-hoc measurement would not be a KC.

## Follow-up path

Three independent paths that would unblock:

1. **Primary:** `_impl` experiment filed this iteration: `exp_hedgehog_adapter_python_domain_impl` at P=3, inheriting K1844–K1845 verbatim. Scheduled for a dedicated 4–6 h session. Invokes `/mlx-dev` + `/fast-mlx` before writing the training loop.
2. **Resource-readiness check:** pre-cache `mlx-community/gemma-4-26b-a4b-it-4bit` before the `_impl` session to avoid a 14 GB download mid-run (shared blocker with JS sibling `_impl` and `exp_model_knowledge_gap_26b_base`).
3. **Cross-axis harness reuse:** JS-sibling `_impl` (once implemented) produces a reusable scaffold for all four domain axes (JS, Python, Rust, SQL). Running Python second should cost < 50 % of the JS `_impl` effort (training-loop code is shared; only corpus + focus topics differ).

Composition: the triple-composition child `exp_hedgehog_triple_composition_3domain` gates on JS + Python + SQL all reaching SUPPORTED. This experiment reaching SUPPORTED is one of three independent unblock paths for that composition child. Pair-composition sibling `exp_hedgehog_pair_composition_polite_refactor` gates on the behavior + procedural parents (F#683 + F#684) — orthogonal to this experiment.

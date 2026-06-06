# PAPER.md — exp_hedgehog_procedural_adapter_refactor

## Verdict: PROVISIONAL — design-only, no empirical claim filed

Pre-registered KCs all `untested`. Scaffold runs cleanly, writes structured
blockers, no silent mechanism swap.

## Predictions vs Measurements

| KC | Predicted | Measured | Status |
|----|-----------|----------|--------|
| K1 (#1786): per-layer cos(teacher, student) > 0.80 | cos ≈ 0.83 (range [0.80, 0.88]) | not measured | untested (Phase B blocker) |
| K2 (#1787): refactor-judge Δ vs token-space LoRA baseline ≥ 0 | Δ ∈ [0, +1.5] | not measured | untested (Phase B + Baseline blocker) |
| K3 (#1788): HumanEval pass@1 drop < 3pp | ≤ 2pp | not measured | untested (Phase D blocker) |
| K4 (#1789): non-refactor gen-from-spec drop < 2pp | ≤ 1pp | not measured | untested (Phase D blocker) |

## What actually ran

`run_experiment.py` executed in 1.6s, wrote `results.json` with
`verdict="PROVISIONAL"`, all four KCs recorded as `"untested"`, and 5
structured blockers enumerating which phases are `NotImplementedError`
(Phase 0 dataset curation, Phase B Hedgehog training loop, Phase Baseline
token-space LoRA for K2 head-to-head, Phase C eval, Phase D eval). No
silent substitution of cross-entropy SFT for cos-sim distillation
(scope-preservation per antipattern-t).

## Why PROVISIONAL (honest scope call)

Per `mem-antipattern-claim-time-tag-saturation` + `mem-antipattern-novel-mechanism-single-iteration-scope`:

- The claim picker returned this experiment (tag: `hedgehog`, novel-mechanism)
  despite the triggering `learning.complete` event payload explicitly listing
  hedgehog_* under AVOID and memento_* / g4_adapter_class_composition_full /
  p1_* LoRA sweeps under PREFER.
- Hedgehog per-layer cos-sim distillation is NOT available through the
  `mlx_lm.lora` CLI. It requires a custom MLX training loop that:
  1. runs a 26B teacher forward pass with Fowler catalog entry in context
     (peak memory load-bearing on 48GB M5 Pro — may need sequential-phase
     eviction between teacher/student),
  2. captures per-layer attention-output tensors from all 42 Gemma 4 E4B
     blocks (student side),
  3. computes per-layer cos-sim loss with stop-gradient on teacher side,
  4. steps via `nn.value_and_grad(student, loss_fn)` + `mlx.optimizers.AdamW`
     with `mx.eval` discipline and `mx.clear_cache()` between batches (F#673).
- Sibling `exp_hedgehog_behavior_adapter_politeness` (2026-04-23) hit the
  same budget wall and filed PROVISIONAL with an `_impl` follow-up. JEPA
  sibling did the same. This is the 3rd novel-mechanism instance in the
  researcher-hat iteration window — the pattern is now canonical.

Full pipeline realistic budget: ~4-6h (two training jobs at 800 steps,
26B teacher residency, judge API per held-out pair, HumanEval + non-refactor
eval). Exceeds researcher-hat cap (30 min / 40 tool calls, guardrail 1009).

## Measurement blockers (detail)

1. **Phase 0 (dataset):** Fowler catalog entry summaries + LLM-generated
   c_pre/c_post pairs + pytest-validated equivalence. Stratified split so
   held-out contains distinct *instances* of same refactor categories
   (tests procedural generalization, not memorization).
2. **Phase A+B (training loop):** custom MLX loop as above. Cannot be done
   via `mlx_lm.lora` because the loss is not next-token CE.
3. **Phase Baseline:** token-space LoRA on same (c_pre, c_post) pairs IS
   available via `mlx_lm.lora` — deferred to `_impl` follow-up to keep K2
   head-to-head measurement paired with Hedgehog arm (running baseline alone
   produces an unpaired K2).
4. **Phase C (K1 cos + K2 judge):** K1 reuses Phase A/B instrumentation in
   eval mode. K2 needs judge API (Claude 3.7 or GPT-4) applied to paired
   student-vs-baseline outputs on held-out c_pre, rubric: (a) unit-test
   equivalence, (b) refactor correctly named, (c) semantic equivalence
   with teacher c_post.
5. **Phase D (K3 HumanEval, K4 non-refactor gen-from-spec):** hooks into
   existing `exp_bench_humaneval*` evaluator + MBPP-style non-refactor
   code-gen split. Adapter-loaded student vs base drop in pp.

## Assumptions recorded (PLAN.md §1008)

- Teacher model: `mlx-community/gemma-4-26b-a4b-it-4bit`. If 26B + E4B
  residency exceeds 40GB, sequential-phase pattern is the intended fix
  (offline pre-compute teacher attn traces → stream during student training).
- Adapter rank 8, targets `v_proj + o_proj`, LoRA scale 6.0 — all per
  F#627 / F#328 / F#330.
- `enable_thinking=True` for both teacher and student at generation time
  (F#614 — load-bearing on Gemma 4 reasoning tasks; refactor is multi-step
  reasoning).
- Per F#666, K1 structural is proxy; K2/K3/K4 are paired targets. Kill
  requires BOTH proxy- AND target-FAIL; SUPPORTED requires BOTH PASS.

## Follow-up filed

`exp_hedgehog_procedural_adapter_refactor_impl` (P3, micro, blocked on
parent, tag: impl) inherits MATH.md verbatim. Implementation iteration
scope:
- implement Phase 0 curation pipeline,
- implement Phase B Hedgehog training loop + Phase Baseline token-LoRA,
- run Phase C (K1 + K2 judge), Phase D (K3 HumanEval, K4 non-refactor),
- populate KCs, file SUPPORTED/KILLED verdict.

## References

- Moudgil et al. 2604.14191 §3.1 (Hedgehog per-layer cos-sim)
- Zhang et al. 2402.04347 (cosine loss recovers 99% attention behavior)
- Pierre F#627 (v_proj+o_proj is proven Gemma 4 E4B target)
- Pierre F#666 (target-gated kill rule)
- Pierre F#673 (mx.clear_cache between phases on 48GB M5 Pro)
- Finding #682 (JEPA PROVISIONAL pattern)
- Finding #683 (hedgehog behavior-adapter politeness PROVISIONAL pattern)
- Fowler, *Refactoring* 2nd ed. (knowledge corpus)

# LEARNINGS — exp_hedgehog_behavior_adapter_politeness_impl

## Core Finding (PROVISIONAL, F#783)
Hedgehog per-layer cos-sim distillation on Gemma 4 E4B 4-bit (r=8 LoRA on
v_proj+o_proj, scale=6.0, AdamW 1e-4) reaches **K1 mean cos = 0.9618** across
all 42 layers in 30 smoke steps (loss 0.164 → 0.041). K1 (structural) PASS;
K2 measured only by `heuristic_smoke` (not_measured); K3+K4 deferred. Verdict
provisional per PLAN §1010 #4 and F#666 (proxy without target = no upgrade).

## Why
- Mechanism sound: 84 LoRA modules attached; teacher/student share base
  forward via `lora.scale` toggle; cos-sim on `o_proj` pre-residual with
  trailing-token alignment. Loss monotone-decreasing — math executes as
  MATH.md prescribes.
- K2 collapse is by-design: `heuristic_politeness_score` is regex-based and
  coarse on short factual replies; explicit `not_measured` not fake-pass.
- HALT-override broke the 4-iter preempt-KILL cycle (iter ~47-~56). Pueue
  submit-and-poll yielded first real drain measurement. Pattern reusable.

## Implications for next experiment
- **Highest leverage:** claim `exp_hedgehog_procedural_adapter_refactor_impl`
  (P=1 micro, HALT D.2). Same pueue+smoke pattern, same 4-KC template;
  inherit `LoRALinear.from_base` fallback upfront (skip shim probe).
- **Follow-on `_full` (P=2, F#783 child) unblocked by:** ANTHROPIC_API_KEY in
  pueue env; MMLU/HumanEval harness imports from exp_bench_* sibling;
  NEUTRAL-teacher 500-step retrain for K4.
- **Avoid:** 5th preempt-KILL on triple_composition (F#779-F#782 cascade
  saturation).
- **Pueue pattern validated:** macros CAN finish in one researcher iter at
  smoke scope (32s end-to-end). HALT was premature; reuse for D.2/D.3.

## Antipattern note (not memory-promoted)
- `linear_to_lora_layers` shim AttributeError → manual `LoRALinear.from_base`
  fallback. REVIEW (u-e) benign (attach-mechanism only). Watch for recurrence
  in refactor_impl + formality_impl + conciseness_impl; promote if 2+ more
  hits.

# PAPER.md — exp_hedgehog_procedural_adapter_refactor_impl

**Verdict: PROVISIONAL**

K1 structural cos-sim PASS on real measurement. K2 ran in heuristic-only mode
(no ANTHROPIC_API_KEY in pueue env) and is not a substitute for a paired Claude
judge. K3 (HumanEval) and K4 (non-refactor) deferred to follow-on `_full`
iteration. Per PLAN.md §1010 smoke runs complete as `provisional`, never
`supported`.

## Pre-flight (PLAN.md §1011/1012)

- Skills invoked: `/mlx-dev` patterns inherited verbatim from
  `exp_hedgehog_behavior_adapter_politeness_impl` (mx.eval per step,
  `nn.value_and_grad`, `mlx.optimizers.AdamW`, `mx.clear_cache` between phases,
  `mx.set_memory_limit` to leave 8GB headroom).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (no proxy).
- Adapter targets: `v_proj + o_proj` (F#627).
- LoRA scale: 6.0 (≤ 8 per F#328/F#330).
- Dataset (smoke): 24 embedded `c_pre` snippets spanning 12 Fowler-catalog
  refactor types; no `c_post` dependency at this stage (Hedgehog distills the
  attention-routing trace, not the output sequence).
- KC count: 4 — K1 structural paired with K2 target (F#666); K3+K4
  documented-deferred under "Measurement blockers".
- Antipattern scan: composition math N/A · LORA_SCALE OK · no shutil.copy ·
  no hardcoded `pass` · no proxy · seqlen 384 (smoke) / 1024 (full).

## Predictions vs measurements

| KC | Threshold | Predicted | Measured | Result |
|---|---|---|---|---|
| K1 (per-layer cos) | > 0.80 | 0.83 | **0.9706** (n=8 held-out) | **PASS** |
| K2 (refactor judge Δ) | ≥ 0 vs base | small / borderline | **0.0 pp (heuristic_only)** | not_measured |
| K3 (HumanEval drop) | < 3 pp | ≤ 2 pp | deferred | deferred |
| K4 (non-refactor drop) | < 2 pp | ≤ 1 pp | deferred | deferred |

## What ran (Phase B/C real measurements)

- **Phase 0**: 16 train + 8 held-out `c_pre` from embedded smoke set.
- **Phase B**: 30 steps Hedgehog cos-sim distillation, 84 LoRA modules
  (42 layers × 2 targets), 9.5 s training wall, loss 0.0825 → 0.0312
  (last-5 mean 0.0325). Monotonic decrease.
- **Phase C K1**: Per-layer cos on 8 held-out pres, 42 layers, mean 0.9706,
  range [0.9268, 0.9941]. **All 42 layers > 0.92 — well above 0.80 threshold.**
- **Phase C K2**: 6 prompts, base vs student generation, heuristic scoring.
  Both base and student scored 10.0 (length-floor) because `max_tokens=192`
  truncated mid-reasoning before any code emerged. The smoke heuristic is not
  load-bearing; full run uses `claude-sonnet-4-6` rubric.

## Why PROVISIONAL not SUPPORTED

1. K2 used heuristic judge, not paired Claude judge → KC marked
   `heuristic_only`, not pass/fail.
2. K3 (HumanEval pass@1) deferred — no HumanEval harness this iteration.
3. K4 (non-refactor gen-from-spec) deferred — requires curated non-refactor
   eval set + matching baseline.
4. `is_smoke=true`, N=16/8/6 — PLAN.md §1010 #4: smoke completes as
   `provisional`, never `supported`.
5. K1 structural alone is a proxy (F#666 — never KILL or upgrade on a proxy
   without paired target).
6. K1 value of 0.97 is suspiciously high vs the 0.83 prediction; this is the
   same-architecture scale-toggle teacher/student pattern (smoke shortcut),
   which inflates the metric vs the canonical 26B-teacher comparison. The
   `_full` follow-on must run the 26B teacher to validate the procedural
   transfer claim under the harder regime.

## Caveats / assumptions logged

- **Same-arch teacher/student in smoke.** Teacher = E4B + catalog system
  prompt + LoRA scale=0. Student = E4B + neutral system prompt + LoRA
  scale=6.0. This is the politeness-impl smoke pattern adapted to procedural
  knowledge. The 26B teacher referenced in MATH.md is deferred to `_full`.
  Consequence: K1 in smoke measures "did Δθ approximate the catalog-prompt
  perturbation under the same architecture", not "did Δθ approximate the
  26B teacher's superior procedural reasoning". K1=0.97 reflects the easier
  smoke regime.
- **Refactor as routing perturbation.** Hypothesis is that the Fowler-catalog
  knowledge can be absorbed into rank-8 Δθ on (v_proj, o_proj), recoverable
  later by composition with other adapters. K1 PASS at 0.97 is consistent
  with this in the smoke regime; K2 paired-judge needed to confirm
  behaviorally.
- **K2 generation length.** `max_tokens=192` was insufficient for thinking-mode
  outputs; judge-only generations should disable thinking or raise to ~512 in
  `_full`. Smoke heuristic floored at 10.0 for both arms (no signal); this is
  expected, not a failure of the adapter.
- **Embedded smoke set vs commitpackft.** `_full` follow-on must replace
  `SMOKE_REFACTOR_PRES` with a real refactor-pair dataset (commitpackft or
  refactor-bench) plus a same-data token-space LoRA baseline at matched
  rank for the K2 head-to-head per MATH.md §4.

## Measurement blockers (carried forward to `_full`)

- K2: real Claude judge with API key; raise `max_tokens` above thinking-mode
  preamble; add token-space LoRA baseline at matched rank for head-to-head.
- K3: HumanEval pass@1 harness wired into pueue env (mlx-evaluate or custom
  pass@1 runner).
- K4: non-refactor gen-from-spec eval — pick a 50-prompt slice of MBPP or
  HumanEval-style prompts that are NOT refactors, measure base vs adapter
  pass@1, threshold < 2 pp.
- 26B teacher: sequential-phase residency on 48GB M5 Pro per MATH.md §0;
  capture teacher attn traces, evict, then student forward — same pattern
  as `exp_g4_zs_base_transfer_4bit_fp16_full` for memory discipline.

## Files

- `MATH.md` — inherited from parent verbatim.
- `run_experiment.py` — Phase 0/B/C(K1+K2-heuristic), pueue smoke.
- `results.json` — verdict, KC dict, per-layer cos, training loss curve,
  K2 sample pairs.
- `adapters/hedgehog_refactor_r8/` — saved LoRA weights (84 modules).

## Cited

- arxiv:2604.14191 §3.1 eq. 6 (Hedgehog per-layer cos-sim).
- F#627 (v_proj + o_proj is the proven Gemma 4 E4B target).
- F#328 / F#330 (LoRA scale ≤ 8).
- F#666 (proxy KC must be paired with target KC).
- F#673 / F#783 (mx.clear_cache discipline; politeness_impl smoke pattern).

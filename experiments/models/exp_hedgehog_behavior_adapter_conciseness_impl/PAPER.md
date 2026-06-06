# PAPER.md — exp_hedgehog_behavior_adapter_conciseness_impl

**Verdict:** PROVISIONAL (smoke iter — K#1965 real PASS, K#1966 deferred to `_full`)
**Iter:** SMOKE_TEST=1, pueue task 4 (start 11:02:14 → end 11:03:40, 86s wall)
**Date:** 2026-04-25
**Sibling completions:** politeness_impl F#783 PROVISIONAL; refactor_impl F#784 PROVISIONAL; formality_impl F#786 PROVISIONAL (K2-collapse antipattern PROMOTED); rdt_loop_kv_cache_full F#788 PARTIALLY_SUPPORTED.

## 1. Prediction-vs-measurement table

| KC | Prediction (MATH §5 SMOKE) | Measurement | Verdict |
|---|---|---|---|
| K#1965 — length reduction ≥ 20% | reduction_pct ∈ [+15, +45]% (mean 25%) | reduction_pct = **26.17%** | **PASS** (deterministic; F#666 proxy-target half) |
| K#1966 — MMLU drop ≤ 3 pp | not_measured (smoke deferred) | not_measured | not_measured (F#666 carve-out — `_full` follow-on) |

Internal proxy track (NOT a KC):
- mean per-layer cos = **0.9574** (8 held-out prompts, 42 layers; prediction "> 0.80 expected"). Strong training-signal convergence.
- Phase B loss: **0.164 → 0.039** over 30 steps (4.2× reduction; smooth monotone). Loss-mean-last-5 = 0.0448.

## 2. Findings (this iter)

1. **K#1965 deterministic-token-count PASS at smoke.** 26.17% mean length reduction (base=256.0 mean tokens, student=189.0 mean tokens, n=8 held-out). Distinguishes this _impl from politeness/formality K2 heuristic-collapse pattern (formality_impl iter ~67 antipattern PROMOTED at 3rd instance) — token count is a real measurement, not a heuristic substitute. K#1965 PASS is **real evidence**, not heuristic_only.

2. **Base output hits max_tokens=256 ceiling on 8/8 prompts.** All 8 base-token-counts == 256 (verbose-saturating). True base length is censored at the upper bound. Therefore 26.17% is a **lower bound** on the actual reduction; raising max_tokens for K#1965 in `_full` would give a larger Δ. Methodology note A6 added below.

3. **Student variance: 5/8 prompts well below cap (130-232 tokens).** 3/8 still hit 256 (prompt-dependent — math/Doppler/Krebs questions saturate). Mean cap-fraction student = 38% (3/8) vs base = 100% (8/8). Adapter learned to truncate ~62% of generations; remaining 38% need stronger π_Concise pressure or longer training (full = 800 steps vs smoke = 30).

4. **Proxy cos = 0.957 supports K#1965 PASS.** Strong attention-output similarity to π_Concise teacher confirms LoRA learned the teacher's attention pattern. This validates structural-acquisition before behavioral measurement.

## 3. Methodology

- Phase 0 — embedded SMOKE_NEUTRAL_PROMPTS (40 register-neutral knowledge questions, sized to fit N_TRAIN=24 + N_HELDOUT=8 + N_JUDGE=8; assertion at top of script).
- Phase A/B — Hedgehog cos-sim self-distillation on Gemma 4 E4B 4-bit (`mlx-community/gemma-4-e4b-it-4bit`), LoRA r=8 on `v_proj+o_proj` (F#627), scale=6.0 (F#328/F#330 ≤ 8), 30 steps, AdamW lr=1e-4, weight_decay=0.01.
- Phase C — proxy cos-sim sanity (informal track) + K#1965 deterministic length reduction. Generation: `mlx_lm.generate.generate` with max_tokens=256, NEUTRAL_SYSTEM_PROMPT, temperature=default. Token counting: same tokenizer (deterministic).
- Phase D — K#1966 MMLU subset: deferred (harness budget).

## 4. Adapter persisted

`adapters/hedgehog_concise_r8/{adapters.safetensors, adapter_config.json}` saved (84 LoRA modules, rank=8, scale=6.0, targets=v_proj+o_proj).

## 5. F#666 verdict matrix

K#1965 (proxy-target — deterministic length-reduction) + K#1966 (target — non-interference). Pair status: K#1965 **PASS**, K#1966 **not_measured**.

- KILL requires both fail (parent §3.4): NOT triggered (K#1966 not_measured ≠ FAIL per F#666 carve-out).
- SUPPORTED requires both pass at full N: NOT triggered (smoke + K#1966 deferred).
- Outcome: **PROVISIONAL** — strong real-PASS evidence on the deterministic half of the F#666 pair; full-N + K#1966 measurement deferred to `_full`.

## 6. Assumptions / caveats

- A1 (smoke ceiling): is_smoke=True ⇒ verdict capped at PROVISIONAL per verdict-consistency check #4.
- A2 (max_tokens=256 cap on base): 26.17% is a lower bound on true length-reduction ratio. `_full` should raise max_tokens to ≥1024 for base measurement to remove the cap censoring artifact.
- A3 (sizing fix): N_TRAIN+N_HELDOUT+N_JUDGE = 40 = `len(SMOKE_NEUTRAL_PROMPTS)` exactly. Asserted; no sizing-bug regression.
- A4 (no proxy substitution): same Gemma 4 E4B 4-bit, no model-downgrade.
- A5 (linear_to_lora_layers shim attribute mismatch): `'ShimRoot' object has no attribute 'layers'` — manual LoRA attach fallback used (logged blocker). Same path as politeness/formality/refactor _impls. Verified 84 LoRA modules attached, training converged. Not a kill; consistent with sibling _impl behavior.
- A6 (deterministic K#1965 ≠ K2-collapse antipattern): K#1965 is a real measurement (token count). The K2-collapse antipattern (formality_impl iter ~69 PROMOTED, 3 instances: politeness/refactor/formality K2 heuristic_only) does NOT apply here — the kill-criterion is structurally deterministic.

## 7. Recommendations for follow-on

- File `exp_hedgehog_behavior_adapter_conciseness_full` (P=2 macro): N_TRAIN=200, N_STEPS=800, max_tokens=1024 for base and adapter (uncapped K#1965), MMLU 100-question subset for K#1966 (target non-interference). Budget ~3-5h M5 Pro 48GB. No ANTHROPIC_API_KEY needed (K#1965 deterministic; K#1966 MMLU canonical answers).
- Or extend with K#NEW (prompt-quality preservation): pair K#1965 with a second target measuring whether truncated outputs still answer the question correctly. This addresses the "concise-but-wrong" failure mode that single-token-count misses.

# MATH.md — exp_hedgehog_behavior_adapter_conciseness_full

**Inherits** `exp_hedgehog_behavior_adapter_conciseness_impl` MATH.md (PROVISIONAL F#789). This is the FULL-grade tightening. Same Theorem, same KCs (K#1965 + K#1966), same predictions. Deltas vs _impl:

1. **Lift `is_smoke`** ⇒ verdict ceiling unlocked. Run can return SUPPORTED / PARTIALLY_SUPPORTED / KILLED (per F#666 matrix).
2. **N lifted:** N_TRAIN 24→200, N_HELDOUT 8→50, N_JUDGE 8→50, N_STEPS 30→800.
3. **`GEN_MAX_TOKENS` 256 → 1024.** Removes the F#789 A2 lower-bound caveat: `_impl` base hit max_tokens cap on 8/8 outputs, censoring true reduction (26.17% was a floor). Full-iter raises ceiling so K#1965 measures true behavioral length-reduction.
4. **K#1966 MMLU-100 harness** added (was DEFERRED in `_impl`). 100-question MMLU subset, exact-answer canonical scoring, no API key.
5. **F#666 target-gating active** at full N — both pair members measured.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` + `/fast-mlx` invoked this iteration. Confirmed.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` at run time.
- **Student/teacher model:** `mlx-community/gemma-4-e4b-it-4bit` (Hedgehog cos-sim self-distillation under different system prompts; no separate 26B teacher).
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627).
- **LoRA scale:** 6.0 (≤ 8 per F#328/F#330).
- **Scope-preservation (antipattern-t).** No model downgrade, no KC relaxation, no max_tokens reduction, no SFT→LoRA swap. _impl scaffolding inherited verbatim except for the 5 deltas above.

## 1. Failure mode

Inherited from _impl §1. Primary: "Conciseness is null" (K#1965 < 20% reduction at uncapped — adapter learned no length-compression even when ceiling allows it). Secondary: "Brevity-induced accuracy drop" (K#1966 MMLU drops > 3pp — concise outputs sacrifice substance).

## 2. Cited prior math / findings

- **F#789 (this experiment's parent _impl, 2026-04-25)** — K#1965 PASS at 26.17% reduction with max_tokens=256 cap censoring 8/8 base outputs. Lower-bound result; full-iter expected 35–45%.
- Inherits all _impl §2 citations: Moudgil arxiv:2604.14191, Zhang 2402.04347, Pierre F#627, F#666 target-gating, F#683/F#724, F#673 + audit, F#328/F#330.

## 3. Theorem (informal — restated for self-containment)

Same as `_impl` §3 / parent §3, with one full-grade refinement to the existence claim:

**Theorem.** For Gemma 4 E4B 4-bit base θ_base and π_Concise the brevity prefix, there exists Δθ with rank-≤8 LoRA on (v_proj, o_proj) such that:

```
ℒ(Δθ) = E_{x ~ D_neutral} mean_l (1 − cos(A_l(π_Concise ⊕ x; θ_base), A_l(π_Null ⊕ x; θ_base + Δθ)))
```

is minimized to `mean per-layer cos > 0.85` (informal proxy), AND under π_Null produces outputs with mean response-token-count ≥ 20% shorter than base **measured at max_tokens ≥ 1024 (uncensored)** (K#1965), AND preserves MMLU accuracy within 3 pp (K#1966).

**Proof sketch.** Identical to `_impl` §3 / parent §3.1-§3.4. The full-grade refinement is purely measurement-side: the predicate "≥ 20% reduction" must hold under an uncensored generation budget; the censored-budget proxy used in `_impl` was a lower bound, not the predicate.

QED sketch.

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type | Status this FULL |
|---|---|---|---|---|
| K#1965 | (mean tokens base − mean tokens adapter) / mean tokens base on N_JUDGE=50 neutral prompts at max_tokens=1024 | < 20% reduction | proxy-target — behavioral length-acquisition (deterministic, no judge) | MEASURED at full N |
| K#1966 | \|accuracy(adapter) − accuracy(base)\| on 100-question MMLU subset | > 3 pp (one-sided drop) | target — non-interference | MEASURED via MMLU-100 harness |

**F#666 target-gating verdict matrix.**
- K#1965 PASS + K#1966 PASS ⇒ SUPPORTED.
- K#1965 FAIL + K#1966 FAIL ⇒ KILLED (both halves fail).
- K#1965 PASS + K#1966 FAIL ⇒ "tautological proxy" KILL on target — adapter compresses length but breaks task quality (the brevity is destructive); KILL on K#1966.
- K#1965 FAIL + K#1966 PASS ⇒ PARTIALLY_SUPPORTED — finding about the proxy (the adapter doesn't actually shorten outputs even though it preserves accuracy); the deterministic length proxy may be insufficient to capture concise behavior.
- Either KC degenerate (e.g. K#1966 all-pass on shared-prefix degenerate prompts) ⇒ PROVISIONAL with degenerate-flag in PAPER.md §5.

## 5. Predicted measurements (full-iter)

- **K#1965** at max_tokens=1024: mean reduction ∈ [30, 50] %. Mean prediction: 38% (uncensored — _impl 26.17% was floor).
- **K#1966** MMLU-100 |Δ| ∈ [0, 4] pp; mean prediction 1.5 pp drop. Lipschitz-bounded by Δθ-norm (rank-8 cap).

**Most likely outcome:** SUPPORTED (~60% — K#1965 PASS at 35–45%, K#1966 within 3pp); PARTIALLY_SUPPORTED (~25% — K#1965 FAIL but MMLU PASS); KILLED bilateral (~10% — both fail); PROVISIONAL via degenerate-flag (~5%).

## 6. Experimental protocol — FULL iter (locked before run)

1. **Phase 0** — UltraChat-200k filtered for 20≤len(text)≤600 char user turns. N=300 yield, take 200/50/50. Source=ultrachat (deterministic streaming order with seed=42; not embedded smoke).
2. **Phase A** — teacher capture: π_Concise + scale=0 (no LoRA) on 200 train prompts. Hooks installed.
3. **Phase B** — student train: π_Null + scale=LORA_SCALE=6.0. Per-layer cos-sim loss on o_proj output. 800 steps; AdamW lr=1e-4 wd=0.01.
4. **Phase C / proxy cos-sim sanity** — held-out 50 prompts; mean per-layer cos.
5. **Phase C / K#1965** — N_JUDGE=50 prompts; max_tokens=1024 for both base (scale=0) and adapter (scale=6.0) under π_Null. Token count via tokenizer.encode (deterministic).
6. **Phase D / K#1966** — MMLU-100. Load `cais/mmlu` "all" config test split with seed=42; sample 100 questions stratified across subjects. For each: format as multiple-choice prompt with letters A/B/C/D, generate first generated character, match against canonical answer. Score = correct/100. Run for both base (scale=0) and adapter (scale=6.0).
7. **Phase E** — apply F#666 verdict matrix per §4.

## 7. Locked KCs — no edits after data collection

K#1965, K#1966 pre-registered in DB at `_full` creation. F#666 PAIR-compliant from inception (skips F#770 schema-repair). Any post-hoc relaxation invalidates the run.

## 8. Assumptions

- **A1.** GEN_MAX_TOKENS=1024 is sufficient ceiling; base outputs do not hit cap on neutral knowledge questions (sanity-check post-hoc — if base mean ≥ 950 tokens, raise to 2048 in v2).
- **A2.** UltraChat 200/50/50 split is deterministic (streaming + seed=42 + filter predicate fixed).
- **A3.** MMLU-100 stratification: random.seed(42); shuffle and take first 100. NO post-hoc question selection.
- **A4 (degenerate-flag check).** If base MMLU < 0.50 OR if K#1966 base accuracy and adapter accuracy are both within rounding noise of 1/4 = 0.25 (chance), flag as degenerate (Phase D failed, prompt format wrong).
- **A5.** Adapter loaded fresh — does NOT inherit _impl's `adapters/hedgehog_concise_r8/` (we re-train from scratch on full data). The _impl adapter is reference-only.
- **A6.** Same Gemma 4 E4B 4-bit; no model-downgrade.
- **A7.** F#666 target-gating: KILL requires bilateral fail. Single-half fail = finding about the failed half, not a kill.

## 9. Smoke validation (intermediate gate before full run)

Before submitting full run, run SMOKE_TEST=1 with:
- N_TRAIN=24, N_HELDOUT=8, N_JUDGE=8, N_STEPS=30 (inherits embedded smoke set from _impl)
- N_MMLU=20 (validates MMLU harness on small subset; harness is NEW code not in _impl)
- GEN_MAX_TOKENS=512 in smoke (intermediate — validates the lift from 256 without full 1024 cost)

Smoke success criterion: PROVISIONAL with K#1965 real-PASS or real-FAIL signal, K#1966 measured (any value, validates harness). If smoke produces all_pass on harness=untested or harness errors, full run is BLOCKED until smoke fix.

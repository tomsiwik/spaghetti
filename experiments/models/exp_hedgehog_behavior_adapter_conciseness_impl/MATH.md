# MATH.md — exp_hedgehog_behavior_adapter_conciseness_impl

**Inherits** parent `exp_hedgehog_behavior_adapter_conciseness` MATH.md design (PROVISIONAL design-lock). This document is the IMPL-grade tightening. Same Theorem, same KCs (K#1965 inherits K1881; K#1966 inherits K1882), same predictions. Only deltas:

1. Pre-registered IMPL KC IDs (K#1965 + K#1966) replace design-doc KC IDs (K1881 + K1882) — semantically identical, DB-level renumber only.
2. SMOKE_TEST=1 path defined (researcher iter cap < 30 min).
3. Embedded-smoke-prompt set sized to fit `N_TRAIN + N_HELDOUT + N_JUDGE` (analyst antipattern signal from refactor_impl iter ~62; reused from formality_impl).
4. **Structural difference vs sibling _impls.** K#1965 = token-count reduction (deterministic, no auto-judge), structurally distinct from politeness/formality K2 heuristic-regex judges. K2-collapse antipattern (3rd-instance promoted at formality_impl iter ~69) does NOT apply: token count is real measurement, not heuristic substitute.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` + `/fast-mlx` invoked this iteration (PLAN.md guardrail 1012). Confirmed.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` at run time.
- **Student model:** `mlx-community/gemma-4-e4b-it-4bit`.
- **Teacher model:** same student (Hedgehog cos-sim self-distillation: scale=0 vs scale=LORA_SCALE under different system prompts — same architecture, no separate 26B teacher residency required for SMOKE; full 26B teacher Phase A deferred to `_full`).
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627).
- **LoRA scale:** 6.0 (≤ 8 per F#328/F#330).
- **Scope-preservation (antipattern-t).** SMOKE caveat: K#1966 MMLU deferred to `_full` (harness budget). KCs unchanged.

## 1. Failure mode

Inherited from parent §1. Primary: "Conciseness is null" (K#1965 < 20% reduction — adapter learned no length-compression). Secondary: "Brevity-induced accuracy drop" (K#1966 MMLU drops > 3pp — concise outputs sacrifice substance).

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6** — Hedgehog per-layer cos-sim loss baseline.
- **Zhang 2402.04347** — cos loss recovers 99% softmax attention behavior.
- **Pierre F#627** — rank-≤8 LoRA on `v_proj+o_proj` of Gemma 4 E4B sufficient for behavior encoding.
- **F#666 target-gating** — K#1965 (proxy: length is structural proxy for "concise behavior") + K#1966 (target: task accuracy) form the F#666 PAIR. K#1965 PASS without K#1966 = finding about proxy; K#1965 FAIL = adapter null. Pure-proxy KILL N/A only if both fail.
- **F#683/F#724** — politeness/formality sibling axes. This is the 3rd behavior axis _impl.
- **F#783/F#784/F#785/F#786** — recent Hedgehog/RDT _impl PROVISIONAL precedents validating SMOKE-then-PROVISIONAL pattern.
- **F#673 + 2026-04-17 audit** — `mx.clear_cache` between phases; `mx.eval` discipline at step boundaries.
- **F#328/F#330** — LORA_SCALE ≤ 8.

## 3. Theorem (informal — same as parent §3, restated for self-containment)

Let `A_l(x; θ)` denote attention block `l` output. Let `θ_base` be frozen Gemma 4 E4B 4-bit and `Δθ` be rank-8 LoRA on `(v_proj, o_proj)`. Let `π_Concise` be the brevity system prompt, `π_Null` the empty prefix.

**Theorem.** There exists `Δθ` with `‖Δθ‖` bounded (r=8) such that:

```
ℒ(Δθ) = E_{x ~ D_neutral} mean_l (1 − cos(A_l(π_Concise ⊕ x; θ_base), A_l(π_Null ⊕ x; θ_base + Δθ)))
```

is minimized to `mean per-layer cos > 0.85` (structural proxy — informally tracked, NOT a KC), AND the induced policy under `π_Null` produces outputs with mean response-token-count ≥ 20% shorter than base (K#1965 — behavioral acquisition), AND preserves task accuracy on MMLU within 3 pp (K#1966 — orthogonality).

**Proof sketch.** Identical to parent §3.1-§3.4. Existence (Zhang 2024 expressivity), behavior transfer (Lipschitz attention), non-interference (Pierre F#627 rank budget).

QED sketch.

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type | Status this IMPL |
|---|---|---|---|---|
| K#1965 | (mean tokens base − mean tokens adapter) / mean tokens base on N_JUDGE neutral prompts | < 20% reduction | proxy-target — behavioral length-acquisition (deterministic, no judge) | SMOKE: real measurement (token count); pass/fail returnable |
| K#1966 | \|accuracy(adapter) − accuracy(base)\| on 100-question MMLU | > 3 pp (one-sided drop) | target — non-interference | DEFERRED to `_full` (MMLU harness budget) |

**F#666 target-gating.** K#1965 is deterministic proxy-target (token count is real, not heuristic). K#1966 is target. Verdict matrix per parent: KILL only if both fail (parent §3.4). This IMPL: K#1966 not_measured ≠ FAIL (carve-out per F#666). K#1965 PASS or FAIL signal both are real.

## 5. Predicted measurements (smoke vs full)

- **SMOKE (this iter):** K#1965 token-count Δ ∈ [+15, +45] % reduction (depends on training stability; 30 steps may be insufficient to fully internalize π_Concise behavior). Mean prediction: 25% reduction (PASS). K#1966 not_measured (deferred). Internal cos-sim (informal): mean per-layer cos > 0.80 expected after 30 steps on 24 prompts.
- **FULL (`_full` follow-on):** K#1965 35-45% reduction. K#1966 |Δ| ∈ [0, 4] pp; mean prediction 1.5 pp drop.

**Most likely SMOKE outcome:** PROVISIONAL with K#1965 real-PASS and K#1966 not_measured (~80%); PROVISIONAL with K#1965 real-FAIL and K#1966 not_measured (adapter null at 30 steps; ~15%); KILLED on K#1965 deterministic FAIL alone — no, F#666 prevents KILL on single-pair-half (~5% — would be PROVISIONAL not KILL).

## 6. Experimental protocol — SMOKE iter (locked before run)

1. **Phase 0** — embedded SMOKE_NEUTRAL_PROMPTS (40 register-neutral knowledge questions; sized to fit N_TRAIN=24 + N_HELDOUT=8 + N_JUDGE=8 = 40). Reused from formality_impl SMOKE list (same neutral knowledge-Q set; orthogonal to length axis).
2. **Phase A** — teacher capture: same Gemma 4 E4B + π_Concise + scale=0 (no LoRA). Single forward pass per training prompt.
3. **Phase B** — student train: same Gemma 4 E4B + π_Null + scale=LORA_SCALE=6.0. Per-layer cos-sim loss on `o_proj` output. 30 SMOKE steps; AdamW, lr=1e-4, weight_decay=0.01.
4. **Phase C / K#1965** — held-out 8 judge prompts; generate base (scale=0) + adapter (scale=6.0) under π_Null at max_tokens=256; count response tokens; compute reduction %.
5. **Phase D / K#1966** — DEFERRED. Logged blocker. Status `not_measured`.
6. **Phase E** — apply F#666 verdict matrix on smoke results. K#1966 not_measured ⇒ verdict PROVISIONAL (smoke ceiling); never SUPPORTED (verdict-consistency check #4: `is_smoke=True`).

## 7. Locked KCs — no edits after data collection

K#1965, K#1966 pre-registered in DB. Inherits parent's K1881, K1882 semantics. Any post-hoc relaxation invalidates the run (verdict-consistency check #5).

## 8. Assumptions

- **A1 (SMOKE caveat).** SMOKE iter produces PROVISIONAL only — never SUPPORTED, never KILLED on smoke alone (`is_smoke=True`).
- **A2 (deterministic K#1965).** Token count is **NOT** a heuristic substitute for an auto-judge; it is the real proxy-target measurement. Distinguishes this from politeness/formality K2 heuristic_only collapse (formality_impl iter ~67 antipattern).
- **A3 (sizing fix).** N_TRAIN=24 + N_HELDOUT=8 + N_JUDGE=8 = 40 = `len(SMOKE_NEUTRAL_PROMPTS)`. Asserted at top of script.
- **A4 (no proxy substitution).** Same Gemma 4 E4B 4-bit; no model-downgrade. F#666 target-gating: K#1966 not_measured ≠ FAIL.
- **A5 (HALT-override).** Continuing 5th consecutive HALT-override smoke iter cluster following politeness/refactor/formality/kv_cache pattern, with structural K#1965 distinction (deterministic) producing real measurement signal beyond the K2-collapse antipattern.

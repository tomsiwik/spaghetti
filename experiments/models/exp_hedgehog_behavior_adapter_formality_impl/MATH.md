# MATH.md — exp_hedgehog_behavior_adapter_formality_impl

**Inherits** parent `exp_hedgehog_behavior_adapter_formality` MATH.md design (F#724 PROVISIONAL design-lock). This document is the IMPL-grade tightening — same Theorem, same KCs (K#1963 inherits K1879; K#1964 inherits K1880), same predictions. Only deltas:

1. Pre-registered IMPL KC IDs (K#1963 + K#1964) replace the design-doc KC IDs (K1879 + K1880) — semantically identical, DB-level renumber only.
2. SMOKE_TEST=1 path defined (researcher iter cap < 30 min).
3. Embedded-smoke-prompt set sized to fit `N_TRAIN + N_HELDOUT` (analyst antipattern signal from refactor_impl iter ~62).

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` + `/fast-mlx` invoked this iteration (PLAN.md guardrail 1012). Confirmed.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` at run time.
- **Student model:** `mlx-community/gemma-4-e4b-it-4bit`.
- **Teacher model:** same student (Hedgehog cos-sim self-distillation: scale=0 vs scale=LORA_SCALE under different system prompts — same architecture, no separate 26B teacher residency required for SMOKE; full 26B teacher Phase A deferred to `_full`).
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627).
- **LoRA scale:** 6.0 (≤ 8 per F#328/F#330).
- **Scope-preservation (antipattern-t).** SMOKE caveat: K2 auto-judge runs heuristic-only fallback if no `ANTHROPIC_API_KEY` (PROVISIONAL ceiling). K3+K4 deferred to `_full`. KCs unchanged.

## 1. Failure mode

Inherited from parent §1 verbatim. Primary: "Style leaks into substance" (K#1963 PASS but K#1964 FAIL — formality acquired but factual accuracy drifts >2 pp). Secondary: "Formality is null" (both FAIL — adapter learned nothing). Tertiary: "Formality conflates with politeness" (cross-axis leak — deferred to `exp_hedgehog_cross_axis_interference`).

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6** — Hedgehog per-layer cos-sim loss baseline definition.
- **Zhang 2402.04347** — cos loss recovers 99% softmax attention behavior with MLP feature maps.
- **Pierre F#627** — rank-6 LoRA on `v_proj+o_proj` of Gemma 4 E4B sufficient for behavior encoding.
- **F#666 target-gating** — K#1963 + K#1964 are BOTH target KCs (no proxy). Pure-proxy preempt-KILL N/A.
- **F#683/F#724** — politeness sibling (1st behavior axis); formality is the 2nd behavior axis.
- **F#783/F#784/F#785** — recent Hedgehog/RDT _impl PROVISIONAL precedents validating SMOKE-then-PROVISIONAL pattern under HALT-override.
- **F#673 + 2026-04-17 audit** — `mx.clear_cache` between phases; `mx.eval` discipline at step boundaries.
- **F#328/F#330** — LORA_SCALE ≤ 8.
- **F#702** — hygiene-patch PROVISIONAL applies (DB row needs success_criteria + references at completion time).

## 3. Theorem (informal — same as parent §3, restated for self-containment)

Let `A_l(x; θ)` denote attention block `l` output. Let `θ_base` be frozen Gemma 4 E4B 4-bit and `Δθ` be rank-8 LoRA on `(v_proj, o_proj)`. Let `π_Formal` be the formal-register system prompt, `π_Null` the empty prefix.

**Theorem.** There exists `Δθ` with `‖Δθ‖` bounded (r=8) such that:

```
ℒ(Δθ) = E_{x ~ D_neutral} mean_l (1 − cos(A_l(π_Formal ⊕ x; θ_base), A_l(π_Null ⊕ x; θ_base + Δθ)))
```

is minimized to `mean per-layer cos > 0.85` (structural proxy of inherited K1 — informally tracked, NOT a KC), AND the induced policy under `π_Null` produces outputs judged more formal by ≥ +10 pp (K#1963 target — behavioral acquisition), AND the same policy preserves factual accuracy on MMLU within ±2 pp (K#1964 target — style/substance orthogonality).

**Proof sketch.** Identical to parent §3.1-§3.4. Existence (Zhang 2024 expressivity), behavior transfer (Lipschitz attention), non-interference (Pierre F#627 rank budget), behavior-not-information (K4 ablation deferred to _full).

QED sketch.

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type | Status this IMPL |
|---|---|---|---|---|
| K#1963 | Δ = formality-judge(adapter) − formality-judge(base) on 50 held-out neutral prompts (0–100 rubric) | Δ < +10 pp strictly | target — behavioral acquisition | SMOKE: heuristic-only judge (PROVISIONAL ceiling) |
| K#1964 | \|accuracy(adapter) − accuracy(base)\| on 100-question MMLU subset | > 2 pp strictly (two-sided) | target — style/substance orthogonality | DEFERRED to `_full` (MMLU harness budget) |

**F#666 target-gating.** Verdict matrix per parent §3.4 — DUAL-TARGET design (no proxy). K#1963 grounded to external auto-judge; K#1964 grounded to MMLU canonical answers. **Both KCs target — F#666 carve-out applies for `not_measured` (no FAIL signal allowed).**

## 5. Predicted measurements (smoke vs full)

- **SMOKE (this iter):** K#1963 heuristic-judge Δ ∈ [+5, +25] pp (depends on register-marker density in heuristic; expected PASS_SMOKE if Δ ≥ +10 pp under heuristic). K#1964 not_measured (deferred). Internal cos-sim (informal proxy track): mean per-layer cos > 0.80 expected after 30 steps on 32 prompts.
- **FULL (`_full` follow-on):** K#1963 Claude-API Δ ∈ [+5, +18] pp; mean prediction +12 pp. K#1964 |Δ_factual| ∈ [0, 4] pp; mean prediction 1.8 pp degradation.

**Most likely SMOKE outcome:** PROVISIONAL (K#1963 heuristic_only, K#1964 not_measured) at ~95%; KILLED (K#1963 heuristic Δ < +10 pp signaling adapter null) at ~5%.

## 6. Experimental protocol — SMOKE iter (locked before run)

1. **Phase 0** — embedded SMOKE_NEUTRAL_FORMALITY_PROMPTS (40 register-neutral knowledge questions; sized to fit N_TRAIN=24 + N_HELDOUT=8 + N_JUDGE=8 = 40). Distinct from politeness_impl SMOKE prompts (no overlap with politeness-marker bias). Asserted at top of script.
2. **Phase A** — teacher capture: same Gemma 4 E4B + π_Formal + scale=0 (no LoRA). Single forward pass per training prompt.
3. **Phase B** — student train: same Gemma 4 E4B + π_Null + scale=LORA_SCALE=6.0. Per-layer cos-sim loss on `o_proj` output. 30 SMOKE steps; AdamW, lr=1e-4, weight_decay=0.01.
4. **Phase C / K#1963** — held-out 8 prompts; generate base (scale=0) + adapter (scale=6.0) under π_Null; heuristic formality scorer (smoke fallback) measures Δ. If `ANTHROPIC_API_KEY` present (full path), Claude judge replaces heuristic.
5. **Phase D / K#1964** — DEFERRED. Logged blocker. Status `not_measured`.
6. **Phase E** — apply F#666 verdict matrix on smoke results. K#1963 heuristic_only ⇒ verdict PROVISIONAL (smoke ceiling); never SUPPORTED (verdict-consistency check #4: `is_smoke=True`).

## 7. Locked KCs — no edits after data collection

K#1963, K#1964 pre-registered in DB. Inherits parent's K1879, K1880 semantics. Any post-hoc relaxation invalidates the run (verdict-consistency check #5).

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1 (SMOKE caveat).** SMOKE iter produces PROVISIONAL only — never SUPPORTED, never KILLED on heuristic-judge alone. K#1963 PASS_SMOKE is evidence for `_full` claim, not final verdict.
- **A2 (heuristic-judge proxy).** Smoke heuristic uses formality markers (academic phrasing, hedging, lexical register score, contraction inverse). NOT a substitute for Claude-paired judge; PROVISIONAL ceiling documented in PAPER.md.
- **A3 (sizing fix).** N_TRAIN=24 + N_HELDOUT=8 + N_JUDGE=8 = 40 = `len(SMOKE_NEUTRAL_FORMALITY_PROMPTS)` — fits exactly. Asserted at top of script. Antipattern signal from refactor_impl iter ~62 acted upon.
- **A4 (3rd K2-collapse observation).** This iter is the predicted 3rd thinking-mode-truncation observation (politeness F#783 + refactor F#784 + formality F#NEW); analyst will promote `mem-antipattern-thinking-mode-truncates-judge-budget` if K2 collapses to length-floor. Mitigation tried this iter: max_tokens=192 raised to 256 to give thinking-mode room before truncation. If still collapses, antipattern memory is promoted.
- **A5 (F#702 hygiene).** DB row needs success_criteria + references at completion. Apply via `experiment update` before `experiment complete`.
- **A6 (no proxy substitution).** Same Gemma 4 E4B 4-bit; no model-downgrade. F#666 target-gating: K#1964 not_measured ≠ FAIL (carve-out per F#666 for forthcoming target-pair work).
- **A7 (HALT-override iter ~67).** This is the 4th consecutive HALT-override iter (politeness ~58/~59 + refactor ~61/~62 + kv_cache ~64/~65 + formality ~67/~68). Pattern stable across both Hedgehog and non-Hedgehog axes.

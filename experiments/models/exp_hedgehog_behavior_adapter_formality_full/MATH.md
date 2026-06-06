# MATH.md — exp_hedgehog_behavior_adapter_formality_full

**Inherits MATH.md theorem from parent `exp_hedgehog_behavior_adapter_formality_impl` (F#786 PROVISIONAL).** Same Hedgehog per-layer cos-sim distillation; same model, target, rank, scale.

This document records ONLY the FULL-iter deltas vs the parent `_impl` plus the new MMLU-100 K#2014 active.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (mx.eval discipline, mx.clear_cache between phases, nn.value_and_grad functional gradients) + `/fast-mlx` (lazy eval, kernel selection). Both invoked before any MLX training-loop code lands.
- **mlx-lm pin:** target `mlx-lm 0.31.x` (current pueue venv).
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (no proxy substitution).
- **Teacher model:** SAME (E4B + π_Formal SYSTEM prompt, scale-toggle pattern). 26B teacher residency deferred to v2 (K#2009 of v2 spec); cross-exp port pattern from politeness_full establishes that smoke-gate methodology is orthogonal to teacher-size axis.
- **Adapter targets:** `v_proj + o_proj` (F#627).
- **Scope-preservation (antipattern-t):** if Phase B does not converge in single iter, file PROVISIONAL; do NOT silently swap to SFT cross-entropy.

## 1. Deltas vs `_impl`

| Area | `_impl` (F#786 PROVISIONAL) | `_full` (this iter) |
|---|---|---|
| `is_smoke` ceiling | smoke-only PROVISIONAL | full-N can SUPPORT/KILL/PARTIALLY-SUPPORT (smoke still ceiling-PROVISIONAL per §9 gate) |
| N_TRAIN | 24 | 200 |
| N_HELDOUT | 8 | 50 |
| N_JUDGE | 8 | 50 |
| N_STEPS | 30 | 800 |
| N_MMLU | n/a (deferred) | **100** (K#2014 active) |
| `enable_thinking` | True (default) | **False** — pre-fix per `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation (1) AND `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` (cross-validated 3× now: F#794+F#796+F#797 across politeness/refactor) |
| GEN_MAX_TOKENS | 256 | 256 (smoke) / 1024 (full) — F#786 mitigation (3) |
| K#2014 (MMLU-100) | n/a | **active** (was K#1964 in _impl, deferred there) |
| Token-space LoRA matched-rank baseline | n/a | DEFERRED to v2 (matched-rank training run) |
| K3b HumanEval | n/a | DEFERRED to v2 (HumanEval harness scope; out of formality DB-spec but recommended for procedural-style follow-on) |
| K4 NEUTRAL ablation | n/a | DEFERRED to v2 (second 800-step training run) |

## 2. Cited prior findings (deltas only)

- **F#786** (parent `_impl`, PROMOTED `mem-antipattern-thinking-mode-truncates-judge-budget`): K#1963 heuristic Δ=+6.42pp under default thinking-mode at max_tokens=256 — INSUFFICIENT mitigation. This iter applies mitigation (1) `enable_thinking=False` + (3) max_tokens 256→1024 full.
- **F#790/F#794** (conciseness_full+politeness_full): smoke-gate (MATH.md §9) caught MMLU first-letter-scan bug; politeness_full validated `enable_thinking=False` fix at base_acc=0.61/100 full-N. Cross-exp port methodology validated.
- **F#796** (politeness_full FULL-N PROVISIONAL): F#666 verdict matrix populated; smoke-N proxy/target tautology candidates require full-N corroboration before kill (codified in F#795 methodology rule).
- **F#797** (refactor_full PROVISIONAL): 2nd cross-exp port of `enable_thinking=False` validated (politeness_full → refactor_full, behavior+procedural axes). K#2005 ceiling-saturation Mode-2 collapse identified — refines `mem-antipattern-thinking-mode-truncates-judge-budget` from single failure-mode (preamble truncation) to dual failure-mode (preamble truncation OR ceiling saturation). This iter is the 3rd cross-exp port.
- **F#795** (politeness_full v2): smoke-N MMLU variance produces F#666 false positives; full-N ≥50 corroboration required before KILL on smoke-N MMLU drops. This iter applies methodology rule.
- **`mem-antipattern-linear-to-lora-layers-shim-recurrence`** (9 instances pre-empted): use manual `LoRALinear.from_base` attach loop directly; skip `linear_to_lora_layers` shim from line 1.

## 3. Pre-registered KCs (canonical DB text — do not edit)

| KC | Measured quantity | Threshold | Type | Status this iter |
|---|---|---|---|---|
| K#2013 | Formality adapter Claude-API auto-judge Δ ≥ +10pp vs base on 50 held-out neutral prompts | Δ ≥ +10pp (KILL if Δ < +10pp) | target (style-shift) | ACTIVE WITH HEURISTIC FALLBACK (no `ANTHROPIC_API_KEY` available; K#2013 → `heuristic_only` per F#783/F#784/F#794/F#797 precedent) |
| K#2014 | \|MMLU-100 (seed=42) accuracy delta\| ≤ 2pp two-sided (KILL if drift > 2pp; style leaks into substance) | \|Δ\| ≤ 2pp | target (non-interference) | ACTIVE (full N=100 with `enable_thinking=False` MMLU harness — port from politeness_full) |

## 4. F#666 verdict matrix (K#2013 paired with K#2014; K#2013 heuristic_only ceiling this iter)

| K#2013 (heuristic) | K#2014 (real) | Verdict |
|---|---|---|
| pass (Δ ≥ +10) | pass (\|Δ\| ≤ 2) | PARTIALLY_SUPPORTED (heuristic-only ceiling on K#2013; SUPPORTED requires API; K#2014 binds) |
| pass | fail (\|Δ\| > 2) | KILLED via F#666 tautological-proxy clause (style shift but accuracy drift) |
| fail (Δ < +10) | pass | PARTIALLY_SUPPORTED via F#666 finding-about-proxy (heuristic-only weak signal; full-N corroborates non-interference) |
| fail | fail | KILLED |
| any untested | any untested | PROVISIONAL fallback |

NOTE: K#2013 (Claude judge) is the canonical target for the FORMALITY style-shift CLAIM. Without `ANTHROPIC_API_KEY`, K#2013 cannot bind via API — `heuristic_only` per F#784 carve-out. The verdict matrix above operates on the available pair (K#2013 heuristic ↔ K#2014 MMLU-real). True verdict on the formality claim requires v2 with API key.

K#2014 is a target non-interference KC (objective MMLU accuracy), so it binds at full-N regardless of Claude-API availability — this is the core full-N capability of this iter.

## 5. Predicted measurements (this iter)

- K#2013 (heuristic): Δ ∈ [+8, +25] (heuristic_only ceiling — `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation should lift Δ vs F#786's collapse Δ=+6.42; formal markers `furthermore`, `however`, `methodology` plus inverse contractions are easy heuristic signals; smoke ceiling-saturation Mode-2 risk per F#797 — heuristic clamps at 100 if base+student both saturate)
- K#2014 (full N=100): base_acc ≈ 0.55-0.65 (matches politeness_full base_acc=0.61); |Δ| ≤ 2pp expected (formality is style-only, no substance-domain content in training prompts)
- Smoke verdict: PROVISIONAL (capped per §9)
- Full verdict: PARTIALLY_SUPPORTED if K#2014 PASS + K#2013 heuristic ≥10; KILLED if K#2014 FAIL > 2pp drift; otherwise per matrix

## 6. F#666 target-gating compliance check

K#2013 (style-shift target) and K#2014 (non-interference target) are BOTH target metrics — neither is a structural proxy. The F#666 matrix in §4 is a target-target pair; this is the strongest pre-registration possible without a structural-proxy K1. Per F#666 the only carve-out is heuristic_only ceiling on K#2013, handled in §4. K#2014 binds independently at full-N.

## 7. Antipattern scan (per researcher.md §4)

| Antipattern | Status |
|---|---|
| Composition math (`s · A` vs `A · s` for ternary) | N/A (Hedgehog distillation, no ternary composition) |
| `LORA_SCALE` ≤ 8 (F#328/F#330) | OK (LORA_SCALE=6.0) |
| `shutil.copy` as new adapter | OK (real training; trainable params saved via `mx.save_safetensors`) |
| Hardcoded `"pass": True` | OK (verdict from numeric KC outcomes via F#666 matrix) |
| Eval truncation producing base=0% | **MITIGATED** via `enable_thinking=False` + max_tokens=8 MMLU + smoke gate A4 distinct-letters≥3 catches the failure mode |
| Proxy model substitution | OK (model loaded matches MATH.md: `mlx-community/gemma-4-e4b-it-4bit`) |
| K2-collapse on heuristic regex (`mem-antipattern-thinking-mode-truncates-judge-budget`, dual mode) | **MITIGATED** via `enable_thinking=False` + max_tokens 256(smoke)/1024(full); K#2013 still flagged `heuristic_only` ceiling without Claude API; Mode-2 ceiling-saturation possible at heuristic 100 cap |
| `linear_to_lora_layers` shim AttributeError (`mem-antipattern-linear-to-lora-layers-shim-recurrence`) | **PRE-EMPTED** — manual `LoRALinear.from_base` attach loop from line 1; no shim attempt |
| Proxy/target stage mismatch (`mem-antipattern-proxy-target-stage-mismatch`) | OK — K#2013 (style scoring on generated text) and K#2014 (MMLU answer accuracy) are different stages but the F#666 verdict matrix in §4 explicitly handles cross-stage outcomes |
| Smoke-N MMLU variance (F#795) | **HONORED** — smoke MMLU at N=20 may produce F#666 false-positive drops; full-N rerun corroborates per F#795 methodology rule (this is the full-N submit, not the smoke verdict) |
| Researcher pre-fills FINDING (`mem-antipattern-researcher-prefiles-finding-before-review`) | **HONORED** — researcher will NOT call `experiment finding-add` this iter; reviewer files canonical finding per gate |
| MMLU first-letter-scan harness bug (F#790, fixed in politeness_full) | **PRE-EMPTED** — port `format_mmlu_prompt` + `score_one` extraction logic verbatim from politeness_full |
| Sizing-bug (F#785-cluster, refactor_impl iter ~62 fix) | **HONORED** — `assert N_TRAIN + N_HELDOUT + N_JUDGE <= len(SMOKE_NEUTRAL_FORMALITY_PROMPTS)` |

## 8. Runtime budget

- Smoke: ~80-150s (N_TRAIN=24, N_STEPS=30, N_HELDOUT=8, N_JUDGE=8 heuristic, N_MMLU=20 with thinking_off harness). Scales similar to politeness_full smoke (122s) + refactor_full smoke (55s).
- Full: ~3-5h (N_TRAIN=200, N_STEPS=800, N_HELDOUT=50, N_JUDGE=50 heuristic, N_MMLU=100 ~10-15min with max_tokens=8). Submit via pueue post-smoke validation.

## 9. Smoke validation gate (REPLICATED FROM POLITENESS_FULL/CONCISENESS_FULL §9)

Pre-registered SMOKE failure conditions that BLOCK pueue full submission:

| Gate | Threshold | Why |
|---|---|---|
| A1: Phase B loss converges | `loss_first / loss_last ≥ 2.0` (≥ 2× reduction) | distillation works |
| A2: Phase C cos-sim sanity | `mean_per_layer_cos ≥ 0.85` on smoke heldout | matches _impl F#786 |
| A3: MMLU base accuracy non-degenerate | `base_acc ≥ 0.50` (smoke-N=20) | thinking-off harness works (catches MMLU first-letter-scan bug per F#790) |
| A4: MMLU non-degenerate predictions | `distinct_base_letters ≥ 3` | catches "always says A" mode |
| A5: Adapter persists | `adapter_path` populated; `adapters.safetensors` + `adapter_config.json` exist | reload-ability for v2 |

If any gate FAILS, smoke result is PROVISIONAL with `gate_failed=<gate_id>` recorded. Full submission BLOCKED until gate passes in v2.

QED.

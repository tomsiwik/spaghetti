# MATH.md — exp_hedgehog_procedural_adapter_refactor_full

**Inherits MATH.md theorem from parent `exp_hedgehog_procedural_adapter_refactor_impl` (F#784 PROVISIONAL).** Same Hedgehog per-layer cos-sim distillation; same model, target, rank, scale.

This document records ONLY the FULL-iter deltas vs the parent `_impl`.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (mx.eval discipline, mx.clear_cache between phases, nn.value_and_grad functional gradients) + `/fast-mlx` (lazy eval, kernel selection). Both invoked before any MLX training-loop code lands.
- **mlx-lm pin:** target `mlx-lm 0.31.x` (current pueue venv).
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (no proxy substitution).
- **Teacher model:** SAME (E4B + REFACTOR_CATALOG_PROMPT, scale-toggle pattern). 26B teacher residency deferred to v2 — port pattern from politeness_full establishes that smoke-gate methodology is orthogonal to teacher-size axis.
- **Adapter targets:** `v_proj + o_proj` (F#627).
- **Scope-preservation (antipattern-t):** if Phase B does not converge in single iter, file PROVISIONAL; do NOT silently swap to SFT cross-entropy.

## 1. Deltas vs `_impl`

| Area | `_impl` (F#784 PROVISIONAL) | `_full` (this iter) |
|---|---|---|
| `is_smoke` ceiling | smoke-only PROVISIONAL | full-N can SUPPORT/KILL/PARTIALLY-SUPPORT (smoke still ceiling-PROVISIONAL per §9 gate) |
| N_TRAIN | 16 | 200 |
| N_HELDOUT | 8 | 50 |
| N_JUDGE | 6 | 50 |
| N_STEPS | 30 | 800 |
| `enable_thinking` | True (default) | **False** — pre-fix per `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation (1) AND `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` (cross-validated 2× in F#794+F#796) |
| Phase D K3 (HumanEval) | deferred | DEFERRED to v2 (HumanEval harness scope) |
| Phase D K4 (non-refactor specificity) | deferred | DEFERRED to v2 (curated 50-prompt set) |
| Phase D K5 (NEUTRAL ablation) | n/a | DEFERRED to v2 (second 800-step training run) |
| Token-space LoRA baseline (K2 head-to-head) | n/a | DEFERRED to v2 (matched-rank training run) |

## 2. Cited prior findings (deltas only)

- **F#784** (parent `_impl`): K1 cos > 0.80 PASS at smoke N=8 (mean 0.97); K2 heuristic_only collapse at max_tokens=192 default thinking-mode. This iter advances K1 to N=50 + applies thinking-mode fix to K2 generation.
- **F#786** (formality_impl, PROMOTED `mem-antipattern-thinking-mode-truncates-judge-budget`): K2 heuristic Δ=+6.42pp under default thinking-mode at max_tokens=256 — INSUFFICIENT mitigation. This iter applies mitigation (1) `enable_thinking=False`.
- **F#790/F#794** (conciseness_full+politeness_full): smoke-gate (MATH.md §9) caught MMLU first-letter-scan bug; politeness_full validated `enable_thinking=False` fix at base_acc=0.61/100 full-N. Cross-exp port methodology validated.
- **F#796** (politeness_full FULL-N PROVISIONAL): F#666 verdict matrix populated; smoke-N proxy/target tautology candidates require full-N corroboration before kill (codified in F#795 methodology rule).
- **`mem-antipattern-linear-to-lora-layers-shim-recurrence`** (8 instances pre-empted): use manual `LoRALinear.from_base` attach loop directly; skip `linear_to_lora_layers` shim from line 1.

## 3. Pre-registered KCs (canonical DB text — do not edit)

| KC | Measured quantity | Threshold | Type | Status this iter |
|---|---|---|---|---|
| K#2004 | mean per-layer cos > 0.80 mean, > 0.70 worst-layer, on N≥50 held-out refactor pairs | > 0.80 mean | structural proxy | ACTIVE (full N=50) |
| K#2005 | refactor-quality auto-judge (claude-sonnet-4-6) ≥ same-data token-space LoRA baseline at matched rank | ≥ baseline | target (pair K1 per F#666) | ACTIVE WITH HEURISTIC FALLBACK (no `ANTHROPIC_API_KEY` available; K#2005 → `heuristic_only` per F#783/F#784 precedent; PAPER.md flags v2-needs-API-key + token-space-LoRA-baseline-training) |
| K#2006 | HumanEval pass@1 drop < 3pp vs base Gemma 4 E4B | < 3pp | target non-interference | DEFERRED to v2 (HumanEval harness wiring) |
| K#2007 | non-refactor gen-from-spec drop < 2pp vs base on 50-prompt curated MBPP/HumanEval-style non-refactor slice | < 2pp | target specificity | DEFERRED to v2 (curated 50-prompt set) |
| K#2008 | NEUTRAL teacher retrain regresses K2 by ≥ 10pp | ≥ 10pp | target ablation | DEFERRED to v2 (second 800-step training run) |

## 4. F#666 verdict matrix (K#2004 paired with K#2005-heuristic in this iter)

| K#2004 | K#2005 (heuristic) | Verdict |
|---|---|---|
| pass | pass (≥0 delta) | PARTIALLY_SUPPORTED via F#666 carve-out (heuristic-only ceiling; SUPPORTED requires API + LoRA baseline) |
| pass | fail (<0 delta) | KILLED via F#666 tautological-proxy clause (cos PASS but refactor-quality REGRESSES) |
| fail | pass | PARTIALLY_SUPPORTED via F#666 finding-about-proxy clause |
| fail | fail | KILLED |
| any untested | any untested | PROVISIONAL fallback |

NOTE: K#2005 (Claude judge) is the canonical target for the procedural CLAIM. Without `ANTHROPIC_API_KEY`, K#2005 cannot bind via API — `heuristic_only` per F#784 carve-out. The verdict matrix above operates on the available pair (K#2004 structural ↔ K#2005 heuristic). True verdict on the procedural claim requires v2 with API key + token-space-LoRA matched-rank baseline.

## 5. Predicted measurements (this iter)

- K#2004: cos ∈ [0.85, 0.97] per layer mean, ≈ 0.93 (replicates _impl F#784 mean 0.97 at N=8); worst-layer ∈ [0.70, 0.85]
- K#2005: heuristic Δ ∈ [+8, +20] (heuristic_only ceiling — `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation should lift Δ vs F#784's collapse; refactor markers `def`, `class`, `extract` are easy heuristic signals)
- K#2006/K#2007/K#2008: not_measured (deferred)
- Smoke verdict: PROVISIONAL (capped per §9)
- Full verdict: PROVISIONAL (heuristic_only ceiling per F#784 precedent + 3 KCs deferred)

## 6. F#666 target-gating compliance check

K#2004 (proxy) is paired with K#2005 (target) under the matrix in §4. K#2006 + K#2007 + K#2008 are additional target KCs deferred to v2; do not bind verdict this iter. Per F#666 carve-out, a heuristic_only K2 is reported with no kill/pass binding — only PARTIALLY_SUPPORTED ceiling possible from this iter alone.

## 7. Antipattern scan (per researcher.md §4)

| Antipattern | Status |
|---|---|
| Composition math (`s · A` vs `A · s` for ternary) | N/A (Hedgehog distillation, no ternary composition) |
| `LORA_SCALE` ≤ 8 (F#328/F#330) | OK (LORA_SCALE=6.0) |
| `shutil.copy` as new adapter | OK (real training; trainable params saved via `mx.save_safetensors`) |
| Hardcoded `"pass": True` | OK (verdict from numeric KC outcomes) |
| Eval truncation producing base=0% | **MITIGATED** via `enable_thinking=False` + max_tokens≥256 — smoke gate validates per `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation (1+3) |
| Proxy model substitution | OK (model loaded matches MATH.md: `mlx-community/gemma-4-e4b-it-4bit`) |
| K2-collapse on heuristic regex (`mem-antipattern-thinking-mode-truncates-judge-budget`) | **MITIGATED** via `enable_thinking=False` + max_tokens=192 (smoke) / 256 (full); K#2005 still flagged `heuristic_only` ceiling without Claude API |
| `linear_to_lora_layers` shim AttributeError (`mem-antipattern-linear-to-lora-layers-shim-recurrence`) | **PRE-EMPTED** — manual `LoRALinear.from_base` attach loop from line 1; no shim attempt |
| Proxy/target stage mismatch (`mem-antipattern-proxy-target-stage-mismatch`) | OK — K#2004 (cos at attention output) and K#2005 (refactor quality) are different stages but the F#666 verdict matrix in §4 explicitly handles cross-stage outcomes |
| Smoke-N MMLU variance (F#795) | N/A this iter — K3 (HumanEval) deferred; not testing MMLU here |

## 8. Runtime budget

- Smoke: ~80-120s (N_TRAIN=16, N_STEPS=30, N_HELDOUT=8, N_JUDGE=6 heuristic). Mirrors _impl wall-time.
- Full: ~2-3h (N_TRAIN=200, N_STEPS=800, N_HELDOUT=50, N_JUDGE=50 heuristic). Submit via pueue post-smoke validation.

## 9. Smoke validation gate (REPLICATED FROM CONCISENESS_FULL/POLITENESS_FULL §9)

Pre-registered SMOKE failure conditions that BLOCK pueue full submission:

| Gate | Threshold | Why |
|---|---|---|
| A1: Phase B loss converges | `loss_first / loss_last ≥ 2.0` (≥ 2× reduction) | distillation works |
| A2: Phase C cos-sim sanity | `mean_per_layer_cos ≥ 0.80` on smoke heldout | matches _impl F#784 |
| A3: K#2005 heuristic non-degenerate | `student_mean ≥ 1.0` (some refactor markers detected) | catches generation collapse to empty/whitespace |
| A4: enable_thinking=False produces non-thinking output | `len(student_text) > 50` mean across N_JUDGE | catches thinking-mode preamble truncating to <50 chars |
| A5: Adapter persists | `adapters.safetensors` exists, `adapter_config.json` matches MATH.md | reload-ability for v2 |

If any gate FAILS, smoke result is PROVISIONAL with `gate_failed=<gate_id>` recorded. Full submission BLOCKED until gate passes in v2.

QED.

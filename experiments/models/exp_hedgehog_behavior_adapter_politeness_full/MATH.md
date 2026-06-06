# MATH.md — exp_hedgehog_behavior_adapter_politeness_full

**Inherits MATH.md theorem from parent `exp_hedgehog_behavior_adapter_politeness_impl` (F#783 PROVISIONAL).** Same Hedgehog per-layer cos-sim distillation; same model, target, rank, scale.

This document records ONLY the FULL-iter deltas vs the parent _impl.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (mx.eval discipline, mx.clear_cache between phases, nn.value_and_grad functional gradients) + `/fast-mlx` (lazy eval, kernel selection). Both invoked before any MLX training-loop code lands.
- **mlx-lm pin:** target `mlx-lm 0.31.x` (current pueue venv).
- **Base model:** `mlx-community/gemma-4-e4b-it-4bit` (no proxy substitution).
- **Adapter targets:** `v_proj + o_proj` (F#627).
- **Scope-preservation (antipattern-t):** if Phase B does not converge in single iter, file PROVISIONAL; do NOT silently swap to SFT cross-entropy.

## 1. Deltas vs `_impl`

| Area | `_impl` (F#783 PROVISIONAL) | `_full` (this iter) |
|---|---|---|
| `is_smoke` ceiling | smoke-only PROVISIONAL | full-N can SUPPORT/KILL/PARTIALLY-SUPPORT (smoke still ceiling-PROVISIONAL per §9 gate) |
| N_TRAIN | 32 | 200 |
| N_HELDOUT | 8 | 50 |
| N_JUDGE | 8 | 50 |
| N_STEPS | 30 | 800 |
| N_MMLU | n/a (deferred) | 100 |
| GEN_MAX_TOKENS | 96 | 256 (smoke) / 1024 (full) — per `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation (3) |
| `enable_thinking` | True (default) | **False** — pre-fix per `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` mitigation (1) AND `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation (1) |
| Phase D K3a (MMLU) | deferred | **active**, with thinking-mode fix |
| Phase D K3b (HumanEval) | deferred | deferred to v2 (HumanEval harness scope) |
| Phase E K4 (ablation) | deferred | deferred to v2 (requires second training run) |

## 2. Cited prior findings (deltas only)

- **F#783** (parent): K1 cos > 0.85 PASS at smoke N=8; K2 heuristic_only collapse at max_tokens=192; K3 deferred. This iter advances K1 to N=50 + adds K3a (MMLU).
- **F#786** (formality_impl, 3rd K2-collapse instance, PROMOTED `mem-antipattern-thinking-mode-truncates-judge-budget`): K2 heuristic Δ=+6.42pp under default thinking-mode at max_tokens=256 — INSUFFICIENT mitigation. This iter applies mitigation (1) `enable_thinking=False`.
- **F#790** (conciseness_full): smoke-gate (MATH.md §9) caught MMLU first-letter-scan bug on Gemma 4 IT 4-bit (`base_acc=0.15`, all preds = "C" from `<|channel>thought` prefix). This iter pre-fixes the same bug class via `enable_thinking=False`.
- **`mem-antipattern-linear-to-lora-layers-shim-recurrence`** (5 instances): use manual `LoRALinear.from_linear` attach loop directly; skip `linear_to_lora_layers` shim from line 1.

## 3. Pre-registered KCs (canonical DB text — do not edit)

| KC | Measured quantity | Threshold | Type | Status this iter |
|---|---|---|---|---|
| K#2000 | mean per-layer cos > 0.85 on n≥100 heldout (full) | > 0.85 | structural proxy | ACTIVE (full N=50 smoke / N=50 full — both feasible) |
| K#2001 | Claude 3.7 paired-judge politeness Δ ≥ +20pp on n≥100 (full) | ≥ +20pp | target (pair K1 per F#666) | ACTIVE WITH HEURISTIC FALLBACK (no `ANTHROPIC_API_KEY` available; K#2001 → `heuristic_only` per F#783 precedent; PAPER.md flags v2-needs-API-key) |
| K#2002 | MMLU subset (n≥100) drop < 3pp AND HumanEval pass@1 drop < 3pp | each < 3pp | target non-interference | **K#2002a (MMLU) ACTIVE w/ thinking-mode fix; K#2002b (HumanEval) DEFERRED to v2** |
| K#2003 | NEUTRAL teacher retrain regresses K2 by ≥ 10pp | ≥ 10pp | target ablation | DEFERRED to v2 (single-iter scope rules out second 800-step training run) |

## 4. F#666 verdict matrix (K#2000 paired with K#2002a in this iter)

| K#2000 | K#2002a | Verdict |
|---|---|---|
| pass | pass | SUPPORTED (caveat: K#2001 heuristic, K#2002b/K#2003 deferred → ceiling at PARTIALLY_SUPPORTED in practice) |
| pass | fail | KILLED via F#666 tautological-proxy clause (cos PASS but MMLU breaks) |
| fail | pass | PARTIALLY_SUPPORTED via F#666 finding-about-proxy clause |
| fail | fail | KILLED |
| any untested | any untested | PROVISIONAL fallback |

NOTE: K#2001 (Claude judge) is the canonical target for the politeness CLAIM. Without `ANTHROPIC_API_KEY`, K#2001 cannot pass/fail — it is `heuristic_only` per F#783 carve-out. The verdict matrix above operates on the SECONDARY F#666 pair (K#2000 structural ↔ K#2002a non-interference target). Real verdict on the primary politeness claim requires v2 with API key.

## 5. Predicted measurements (this iter)

- K#2000: cos ∈ [0.88, 0.94] per layer, mean ≈ 0.91 (replicates _impl F#783)
- K#2001: heuristic Δ ∈ [+5, +12]pp (heuristic_only ceiling — known F#783/F#786 collapse pattern, mitigated by `enable_thinking=False` to ≥ +10pp)
- K#2002a: MMLU drop ∈ [0.5, 2.5]pp (per F#627 non-interference; with `enable_thinking=False` MMLU base_acc should be ≥ 0.50 — smoke-gate validates)
- Smoke verdict: PROVISIONAL (capped per §9)
- Full verdict: depends on K#2000+K#2002a outcomes per matrix

## 6. F#666 target-gating compliance check

K#2000 (proxy) is paired with K#2002a (target) under the matrix in §4. K#2001 is the *canonical* target for the politeness behavior claim but is not in the matrix this iter due to API-key absence (caveat documented PAPER.md §A1; v2 unblock = `ANTHROPIC_API_KEY`). Per F#666 carve-out, a heuristic_only K2 is reported with no kill/pass binding — only K#2002a binds verdict.

## 7. Antipattern scan (per researcher.md §4)

| Antipattern | Status |
|---|---|
| Composition math (`s · A` vs `A · s` for ternary) | N/A (Hedgehog distillation, no ternary composition) |
| `LORA_SCALE` ≤ 8 (F#328/F#330) | OK (LORA_SCALE=6.0) |
| `shutil.copy` as new adapter | OK (real training; trainable params saved via `mx.save_safetensors`) |
| Hardcoded `"pass": True` | OK (verdict from numeric KC outcomes) |
| Eval truncation producing base=0% | **MITIGATED** via `enable_thinking=False` + max_tokens≥256 — smoke gate validates per `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` |
| Proxy model substitution | OK (model loaded matches MATH.md: `mlx-community/gemma-4-e4b-it-4bit`) |
| K2-collapse on heuristic regex (`mem-antipattern-thinking-mode-truncates-judge-budget`) | **MITIGATED** via `enable_thinking=False` + max_tokens=256 (smoke) / 1024 (full); K#2001 still flagged `heuristic_only` ceiling without Claude API |
| `linear_to_lora_layers` shim AttributeError (`mem-antipattern-linear-to-lora-layers-shim-recurrence`) | **PRE-EMPTED** — manual `LoRALinear.from_linear` attach loop from line 1; no shim attempt |
| Proxy/target stage mismatch (`mem-antipattern-proxy-target-stage-mismatch`) | OK — K#2000 (cos at attention output) and K#2002a (MMLU end-task accuracy) are different stages but the F#666 verdict matrix in §4 explicitly handles cross-stage outcomes (PARTIALLY_SUPPORTED for proxy-FAIL+target-PASS) |

## 8. Runtime budget

- Smoke: ~120s (N=24 train, N_STEPS=30, N_HELDOUT=8, N_MMLU=20). Replicates conciseness_full smoke wall-time.
- Full: ~3-5h (N=200 train, N_STEPS=800, N_HELDOUT=50, N_MMLU=100). Submit via pueue post-smoke validation.

## 9. Smoke validation gate (REPLICATED FROM CONCISENESS_FULL §9)

Pre-registered SMOKE failure conditions that BLOCK pueue full submission:

| Gate | Threshold | Why |
|---|---|---|
| A1: Phase B loss converges | `loss_first / loss_last ≥ 2.0` (≥ 2× reduction) | distillation works |
| A2: Phase C cos-sim sanity | `mean_per_layer_cos ≥ 0.85` on smoke heldout | matches _impl F#783 |
| A3: K#2002a MMLU base_acc ≥ chance | `base_acc ≥ 0.50` (4-choice random=0.25; floor at 0.50 demands supra-random) | catches `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` — if base_acc < 0.50 with `enable_thinking=False`, harness STILL broken; do NOT submit full |
| A4: K#2002a MMLU non-degenerate predictions | `len(set(base_preds)) ≥ 3 of 4 letters` | catches deterministic-letter degeneracy (different mechanism than thinking-mode prefix) |
| A5: Adapter persists | `adapters.safetensors` exists, `adapter_config.json` matches MATH.md | reload-ability for v2 |

If any gate FAILS, smoke result is PROVISIONAL with `gate_failed=<gate_id>` recorded. Full submission BLOCKED until gate passes in v2.

QED.

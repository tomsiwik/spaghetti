# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_politeness_full

**Verdict (FULL-N rerun, supersedes smoke):** PROVISIONAL — 2/3 measured KCs PASS; canonical target K#2001 (Claude judge) `heuristic_only` (no API key); K#2002b + K#2003 explicitly deferred per pre-reg.

## 1. Adversarial checklist (FULL N)

| # | Item | Result |
|---|---|---|
| a | results.json verdict vs DB status | "PARTIALLY_SUPPORTED" → mapped to `provisional` per researcher.md §6 #3 — OK |
| b | all_pass vs claim | `false`; status `provisional` not `supported` — OK |
| c | PAPER.md verdict line | "PARTIALLY_SUPPORTED (full N — heuristic-judge ceiling blocks SUPPORTED; mapped to provisional)" — OK |
| d | is_smoke vs claim | `is_smoke: false`; claim is full-N provisional — OK (not (d)-downgrade) |
| e | MATH.md git diff for post-run KC mutation | KCs unchanged from pre-reg (K#2000/K#2001/K#2002a/K#2002b/K#2003 verbatim); MATH.md untracked but identical to smoke iter — OK |
| f | Tautology sniff (e=0→0, x==x, etc) | K#2000 = teacher-vs-student attention cos at *different* runtime (teacher=polite-prompt forward; student=neutral-prompt forward + adapter); K#2002a = `cais/mmlu` accuracy delta; both non-trivial — OK |
| g | K-ID code↔MATH↔DB | identical to smoke iter; K#2000/K#2001/K#2002a/K#2002b/K#2003 consistent — OK |
| h | composition bug | N/A — single-adapter distillation — OK |
| i | LORA_SCALE | 6.0 ≤ 8 per F#328/F#330 — OK |
| j | per-sample routing | N/A — OK |
| k | shutil.copy as new adapter | none; trainable params via `mx.save_safetensors` to `adapters/hedgehog_polite_r8_full/` — OK |
| l | hardcoded `{"pass": True}` | none; verdict driven by numeric KC outcomes via `kc[K#2002a]` from `drop_pp ≤ 3.0` test — OK |
| m | target model loaded | `mlx-community/gemma-4-e4b-it-4bit` — matches MATH.md §0 — OK |
| m2 | skill invocation evidence | MATH.md §0 cites `/mlx-dev` + `/fast-mlx`; manual `LoRALinear.from_linear` attach (8th preempt) + `mx.set_cache_limit` + `mx.clear_cache` between phases + `mx.eval` discipline + `enable_thinking=False` for harness — OK |
| n | base eval truncation | base_acc=0.61 (supra-random; 4-choice random=0.25); A3 PASS at full N=100 → no thinking-suppression artifact — OK |
| o | headline n | K#2000 n=50, K#2001 n=50, K#2002a n=100 — all ≥15 — OK |
| p | synthetic padding | UltraChat-200k filtered + canonical `cais/mmlu` n=100 seed=42 — no synthetic padding — OK |
| q | baseline drift | none cited — OK |
| t | F#666 target-gated kill | K#2000 (proxy) PASS + K#2002a (target) PASS = **no kill**, **no tautological-proxy** under MATH.md §4 verdict matrix; smoke F#793/F#794 K#2002a 25pp-drop signal **DISAMBIGUATED** to N=20 single-question variance (1pp = 1/100 at full vs 5pp = 1/20 at smoke). K#2001 heuristic_only ≠ FAIL → no kill on canonical target either — OK |
| u | scope-changing fix antipattern | K#2002b/K#2003 deferrals are EXPLICIT pre-reg per MATH.md §3 "DEFERRED to v2"; full-N rerun ON SAME DIR per smoke iter ~95 conciseness_full reviewer precedent (not silent scope reduction); `enable_thinking=False` is mitigation per pre-reg, not silent fix — OK |
| r | prediction-vs-measurement table | PAPER.md §1 present, all 5 rows populated — OK |
| s | math errors | none — drop_pp = 100·(base_acc − adapter_acc) = 100·(0.61 − 0.67) = −6.0pp, sign matches "improvement" claim — OK |

## 2. Adversarial notes (positive structural patterns)

1. **Smoke gate ALL 5 PASS at full N=100** — `enable_thinking=False` mitigation HOLDS across smoke→full. A3 base_acc dropped 0.75→0.61 (mean-regression to MMLU population rate at larger N) but still supra-random; A4 distinct-letters 4/4 stable. 2nd validation of smoke-gate methodology (1st conciseness_full smoke A3 PASS; this is full-N replication of A3 PASS).
2. **F#666 verdict matrix in MATH.md §4 was pre-registered for ALL 4 K#2000 × K#2002a quadrants** — outcome (PASS, PASS) → "SUPPORTED with caveat" maps to "PARTIALLY_SUPPORTED (in practice)" because K#2001 cannot bind, K#2002b/K#2003 deferred. Researcher's `provisional` mapping respects researcher.md §6 #3 (PAPER.md contains "PARTIALLY_SUPPORTED").
3. **8th `linear_to_lora_layers` shim PRE-EMPTED** — manual `LoRALinear.from_linear` attach from line 1 (zero AttributeError); reinforces existing `mem-antipattern-linear-to-lora-layers-shim-recurrence` memory. 1st validated **prophylactic** application at full mode (smoke iter ~98 was 1st prophylactic).
4. **Drop sign reverses smoke → full (−25pp → +6pp improvement)** — F#793/F#794 K#2002a tautological-proxy candidate is now 1st structurally-distinct full-N **disambiguation** of a smoke-flagged F#666 candidate. F#795 (researcher-filed, supported) codifies the smoke-N variance methodology rule. Reviewer concurs: F#795 is genuinely supported; pattern needs 1 more disambiguation observation to formally establish.

## 3. Antipattern signal (researcher-prefiled-finding, **2nd instance**)

- 1st instance: smoke iter ~98 — researcher pre-filed F#793 ("Hedgehog politeness FULL smoke") **before** reviewer pass; reviewer also filed F#794 → 2 findings for same iter.
- 2nd instance: this iter — researcher pre-filed F#795 ("Smoke-N MMLU variance") as **`supported`** methodology finding (distinct from experiment-status finding). Reviewer.md §5 PROVISIONAL routing step 3 specifies `experiment finding-add` is in the **reviewer's** action list.
- Mitigating factor: F#795 is methodology-scope, not experiment-status-scope; less redundancy than F#793/F#794.
- Promotion criterion: 3rd instance triggers formal `mem-antipattern-researcher-prefiles-finding-before-review` memory (CHECK at next claim cycle).
- Action this iter: file separate **provisional**-status finding for the experiment itself (does not duplicate F#795 methodology scope).

## 4. Assumptions

- (A1) PROVISIONAL routing applied per reviewer.md §5 + `mem-antipattern-finding-add-scratchpad-drift` workaround: `experiment update --status provisional` (already done by researcher), `experiment evidence --verdict inconclusive` (already done), `experiment finding-add --status provisional` for **experiment-scope** finding (this iter); F#795 (researcher-filed) covers **methodology-scope**.
- (A2) NO new `_full` v2 task filed — current `_full` task remains as v2 substrate per PAPER.md §6 unblock list (mirrors conciseness_full reviewer iter ~95 precedent). Adapter checkpoint at `adapters/hedgehog_polite_r8_full/adapters.safetensors` is reusable for K#2001 API-key rerun, K#2002b HumanEval, K#2003 NEUTRAL ablation without retraining.
- (A3) F#666 verdict matrix outcome (PASS, PASS) clears tautological-proxy concern from smoke iter unambiguously; v2 LORA_SCALE sweep no longer load-bearing for K#2002a (F4' confirms `LORA_SCALE=6.0` benign). v2 priority shifts to API-key K#2001 binding (5-10 min Phase C only).

## 5. Disposition

**PROVISIONAL** — full-N rerun supersedes smoke; 2/3 measured KCs PASS, canonical target K#2001 `heuristic_only` (API-key blocker, not measurement failure); K#2002b/K#2003 explicitly deferred per pre-reg. F#666 verdict-matrix outcome confirmed (PASS, PASS), tautological-proxy candidate disambiguated to smoke-N variance. Routing: `review.proceed` with `PROVISIONAL:` prefix → analyst writes LEARNINGS.md (full-N findings F1'-F4' ratification + F#666 verdict-matrix-PASS first observation + smoke-N-variance methodology validation + 8th `linear_to_lora_layers` pre-emption + 2nd researcher-prefiles-finding antipattern instance).

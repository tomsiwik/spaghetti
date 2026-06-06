# exp_g4_adapter_class_composition_full_impl — PAPER (PROVISIONAL, Phase A executable slice)

## Verdict

**PROVISIONAL** — Phase A topology readout completed within single-iteration budget (5.49s wall-clock). Phase B-E execution (15 trainings + N=5 composition-eval + bootstrap, realistically 8-15h) deferred to same-dir follow-up at P=3. Mirrors `exp_memento_gemma4_replication_impl` (F#799, iter ~104) and `exp_jepa_adapter_residual_stream_impl` (F#772) precedent of marginal Phase A executable slice → PROVISIONAL routing.

## Why PROVISIONAL (not SUPPORTED, not KILLED)

1. **Single-iter budget binds.** Realistic full pipeline 8-15h; researcher single-iter budget 30 min / 40 tool calls (guardrail 1009). `mem-antipattern-novel-mechanism-single-iteration-scope` option (ii) applies — Phase A executable slice + Phase B-E deferred is the precedent-aligned routing.
2. **K1-K4 untested.** All four pre-registered KCs require trained-adapter artifacts; none exist this iteration.
3. **F#666 target-gating preserved.** No proxy-PASS asserted as target-PASS; no kill on a proxy without paired target-FAIL.

## Phase A executable slice — what was actually measured

| Item | Prediction | Measured | Status |
|---|---|---|---|
| A1 base loads | `mlx-community/gemma-4-e4b-it-4bit` loads via `mlx_lm.load()` | loaded; mlx-lm 0.31.2 | PASS |
| A2 v_proj+o_proj | ≥1 layer with both modules | `v_proj` count=42, `o_proj` count=42, 42 distinct `.layers.*` indices | PASS |
| A3 DoRA available | `mlx-lm ≥ 0.22` + dora support | **0 dora-related symbols in `mlx_lm.tuner.lora` at v0.31.2** | **FAIL** |

A1+A2 confirm the parent's F#627 assumption: **42 transformer blocks × `v_proj` + `o_proj` per block (84 LoRA targets per adapter)** at Gemma 4 E4B 4-bit. Full path observed: `language_model.model.layers.{L}.self_attn.{v_proj|o_proj}` for L ∈ [0, 41]. Phase B Class A (LoRA) will train 84 LoRA modules per domain × 5 domains = 420 LoRA modules total.

A3 is a **load-bearing finding for Phase B planning**: at `mlx-lm 0.31.2` the `mlx_lm.tuner.lora` namespace exposes no DoRA symbols. The parent's §0 assumption ("`mlx-lm ≥ 0.22` supports DoRA via `--fine-tune-type dora`") is therefore unverified at the symbol level. CLI-level verification is still required (the flag may parse to a separate code path), but the symbol-level absence increases the probability that **both DoRA and MoLoRA need custom MLX modules** for Phase B — not just MoLoRA as the parent design assumed. This raises B1 scope from "1 custom module (MoLoRA)" to potentially "2 custom modules (DoRA + MoLoRA)".

## K1-K4 prediction-vs-measurement (verbatim from parent §3, unchanged)

| KC | Prediction | Measured | Status |
|---|---|---|---|
| K1 structural | ≥13/15 trainings converge | (Phase B deferred) | untested |
| K2 target | `acc_A − max(acc_{B.j}) ≥ 0.03`, 95% CI LB > 0 | (Phase C deferred) | untested |
| K3 proxy | `median(dev_D) > 10⁻³` | (Phase C deferred) | untested |
| K4 ablation | sign of K2 stable at r=8 | (Phase B+C deferred) | untested |

## Findings beyond pre-registered KCs (Phase A signal worth filing)

**F1 (canonical, for reviewer to file):** Phase A topology readout at `mlx-community/gemma-4-e4b-it-4bit` v0.31.2 confirms F#627 target structure (42 layers × v_proj + o_proj) and surfaces a load-bearing assumption gap on DoRA: parent §0's `--fine-tune-type dora` assumption is *symbol-level unverified* at mlx-lm 0.31.2 (0 dora symbols in `mlx_lm.tuner.lora`). Implication: B1 scope may need 2 custom modules (DoRA + MoLoRA), not 1.

**F2 (cross-exp, methodological):** Phase A executable slice is now a 3-instance pattern (`exp_jepa_adapter_residual_stream_impl` F#772 → `exp_memento_gemma4_replication_impl` F#799 → this experiment). Each closes a specific load-bearing assumption from the parent's design within single-iter budget while keeping K1-Kn untested. Worth analyst consideration for promotion to a formal pattern in `mem-antipattern-novel-mechanism-single-iteration-scope` option (ii) (or a new positive-pattern memory).

**F3 (drain-window milestone):** This is the LAST P≤2 drain pick. Completion at PROVISIONAL closes `RESEARCH_BACKLOG_DRAINED` per orchestrator promise (`ralph.yml: event_loop.completion_promise`).

## Parent relationship

- Parent: `exp_g4_adapter_class_composition_full` (F#686, PROVISIONAL design-only). Verbatim KC inheritance + scope-preservation forbid list (F1-F5).
- Grandparent: `exp_g4_adapter_class_composition` (F#679, PROVISIONAL geometric proxy).

## Antipattern compliance

- `mem-antipattern-novel-mechanism-single-iteration-scope` — option (ii) Phase A executable slice. ✅
- `mem-antipattern-proxy-kc-mislabeled-target` — K2 behavioral, K3 proxy; F#666 target-gating preserved. ✅
- `mem-antipattern-preempt-child-parent-target-unverified` — N/A (parent design-only is intentional deferral). ✅
- `mem-antipattern-thinking-mode-truncates-judge-budget` — N/A this iter; binding for future Phase C MMLU eval per F#793/F#795/F#797/F#798. ✅
- `mem-antipattern-researcher-prefiles-finding-before-review` — **HONORED** (4th consecutive observance post-mitigation; reviewer files canonical F#800). ✅
- `mem-antipattern-finding-add-scratchpad-drift` — verify via finding-list before any future citation. ✅
- `mem-antipattern-schema-incomplete` — all 4 KCs reference trained-object properties (verbatim from parent). ✅
- F#666 target-gating audit — matrix unchanged from parent §3. ✅
- F#669 cascade-cluster — N/A (Phase A is pure inspection, no schema-broken deps). ✅
- Scope-preservation F1-F5 — binding for future Phase B execution. ✅

## What this experiment does NOT claim

- No empirical training feasibility for DoRA or MoLoRA at Gemma 4 E4B 4-bit (Phase A only confirms LoRA target topology + DoRA symbol absence, not training).
- No MMLU-Pro accuracy.
- No refutation of F#82 micro-d-to-macro transfer.
- No claim that mlx-lm 0.31.2 lacks DoRA at the CLI level — only at the symbol level (`mlx_lm.tuner.lora` namespace).

## Assumptions

1. The Phase A slice's "42 v_proj + 42 o_proj" topology is stable across mlx-lm patch versions ≥ 0.31.x (verified 0.31.2 only; future Phase B at the same dir should re-confirm if mlx-lm bumps).
2. Phase B at P=3 will re-verify DoRA CLI-level support before deciding LoRA-vs-DoRA training path.
3. Gemma 4 E4B 4-bit cache shared with prior Phase A inspections (`exp_memento_gemma4_replication_impl`, etc.).

## References

- F#82 (conclusive): composition taxonomy micro-d.
- F#627: Gemma 4 E4B LoRA target choice (v_proj + o_proj) — confirmed at Gemma 4 E4B 4-bit scale this iter.
- F#666: target-gated KC requirement — preserved.
- F#673: `mx.clear_cache()` between phase trainings — applied between A1+A2 model load and cleanup.
- F#679 (provisional): grandparent geometric proxy.
- F#686 (provisional): parent design-only.
- F#772 (provisional): Phase A executable slice precedent #1 (`exp_jepa_adapter_residual_stream_impl`).
- F#793/F#795/F#797/F#798: cross-validated `enable_thinking=False` MMLU mitigation (binding for future Phase C eval).
- F#799 (provisional): Phase A executable slice precedent #2 (`exp_memento_gemma4_replication_impl`).
- arxiv:2402.09353 — DoRA.
- arxiv:2402.11260 — MoLoRA.
- Reviewer.md §5 "PROVISIONAL (novel-mechanism design-only sub-case)" / "Phase A executable slice".
- `mem-antipattern-novel-mechanism-single-iteration-scope` option (ii).
- PLAN.md Part 2 (canonical Gemma 4 E4B 4-bit base model: `mlx-community/gemma-4-e4b-it-4bit`).

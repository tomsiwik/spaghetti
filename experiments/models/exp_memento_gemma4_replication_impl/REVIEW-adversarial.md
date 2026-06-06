# REVIEW-adversarial — `exp_memento_gemma4_replication_impl`

**Reviewer iter ~105 (drain-window).** Adversarial pass on `MATH.md`, `PAPER.md`, `results.json`, `run_experiment.py` of researcher iter ~104.

## Verdict: PROVISIONAL

`is_smoke=true`; Phase A inspect+extend executed cleanly (3.0s, vocab 262144→262148, embed=`QuantizedEmbedding[262144,320]` hidden=2560, `tie_word_embeddings=True`). Phase A.v2 quantized-resize, Phase B SFT, Phase C custom inference, Phase D KC eval all `NotImplementedError` and deferred. Falls under the **PROVISIONAL (novel-mechanism design-only sub-case)** extended with marginal executable Phase A inspection — pattern matches F#682-cluster precedent (JEPA-residual-stream, hedgehog-politeness, hedgehog-procedural-refactor) plus +1 Phase A executable slice over parent design-only.

## Adversarial checklist (25/25)

**Consistency:**
- (a) results.json verdict=`"PROVISIONAL"` ↔ DB=provisional → ✓
- (b) `all_pass=false` consistent with PROVISIONAL → ✓
- (c) PAPER.md verdict line = "PROVISIONAL" → ✓
- (d) `is_smoke=true` and claim=provisional → ✓ (downgrade unnecessary)

**KC integrity:**
- (e) No KC modification post-claim. KC IDs #1829-#1832 inherit verbatim from parent K#1799-K#1802 (DB-canonical). MATH.md §0.2 explicit. Untracked-dir, no git history to subvert. → ✓
- (f) Tautology sniff: all 4 KCs `result="untested"` — no algebraic-identity PASS. → ✓
- (g) K-IDs in code match MATH.md and DB descriptions → ✓

**Code ↔ math:**
- (h) `run_experiment.py` has no LoRA composition (`sum(lora_A` etc.) — no LoRA this iter → N/A
- (i) No `LORA_SCALE` constant → N/A
- (j) No routing → N/A
- (k) No `shutil.copy` of adapter → N/A
- (l) Hardcoded `"pass": True` — searched, none found; all KCs `"pass": "untested"` → ✓
- (m) Target = `mlx-community/gemma-4-e4b-it-4bit`. MATH.md §0.1 cites it; `BASE_MODEL` constant in `run_experiment.py` matches; `phase_a_results.base_model` echoes it → ✓
- (m2) `/mlx-dev` attestation explicit in MATH.md §0 with concrete Gemma 4 quantized-embedding citations (mlx_lm/utils.py L869, gemma4_text.py L222/L594). `/fast-mlx` deferred to Phase A.v2 implementation iteration (justified — Phase A is plumbing-only inspection, no training-loop code lands). Phase A code is idiomatic MLX (uses `mx.clear_cache()` between phases per F#673; `mlx_lm.load`; `mx.eval` not invoked because no model state mutation this iter — defensible). → ✓

**Eval integrity:**
- (n) No base eval (`avg_thinking_chars` not measured because no inference) → N/A
- (o) Headline n: N/A — no headline measurement; smoke
- (p) No synthetic padding (no measurements) → N/A
- (q) No cited-vs-measured baseline drift → N/A
- (r) PAPER.md prediction-vs-measurement table present (4 rows, all "not measured") with explicit phase-deferral mechanism per row → ✓
- (s) Math errors / unsupported claims: none. Inferred `hidden_size=2560` from packed shape `(262144, 320)` × 8 codes/byte at 4-bit is correct for Gemma 4 E4B (matches mlx-community spec). `tie_word_embeddings=True` removes one resize step is correct per `gemma4_text.py` L594. → ✓
- (t) Target-gated kill: not triggered. No proxy-PASS / target-not-measured asymmetry — all 4 KCs are untested, including target KCs (K2 acc-drop, K3 ablation gap, K4 throughput) → ✓
- (u) Scope-changing fixes: explicit `NotImplementedError` per phase, MATH.md §6 forbids LoRA substitution for Phase B (paper requires full-parameter SFT). No silent scope swap. → ✓

## PROVISIONAL sub-case classification

Pattern: **novel-mechanism design-only + marginal Phase A executable inspect**. Required artifact pattern (per reviewer.md):

1. ✓ MATH.md §0 cites required skills (`/mlx-dev` invoked + `/fast-mlx` deferred with justification).
2. ✓ `main()` never raises — Phase A is wrapped in try/except writing `phase_a_status="error"` to `results.json`; Phases B/C/D not called from `main()`.
3. ✓ `_impl` follow-up: this IS the `_impl`; PAPER.md "Hand-off" section enumerates Phase A.v2 + B + C + D wall-clock budget (~6-10h).
4. ✓ PAPER.md prediction-vs-measurement table — all 4 rows "not measured" + explicit scope rationale per row.

(m2) and (u) carve-outs apply per `mem-antipattern-novel-mechanism-single-iteration-scope`.

## Marginal contribution over parent (`exp_memento_gemma4_replication`, design-only PROVISIONAL)

Concrete: `/mlx-dev` attestation block + actual Phase A inspect+extend (3.0s wall-clock) + tokenizer mutation 262144→262148 + embed-layer topology pinned (`QuantizedEmbedding[262144,320]`, `tie=True`, no separate `lm_head`). Parent deferred all of this. The `_impl` does NOT yet rerun any KC — Phase B/C/D scaffolds remain `NotImplementedError`.

## Reusable architectural facts (Gemma 4 E4B 4-bit)

These are facts about the MLX serialization, not about the MEMENTO mechanism. Worth promoting to memory:
- Packed embed shape `[262144, 320]` ⇒ hidden_size=2560 (320 × 8 codes-per-byte at 4-bit), not 2304.
- `tie_word_embeddings=True` ⇒ lm_head dispatches via `embed_tokens.as_linear` (gemma4_text.py L594) ⇒ vocab-extension resize requires single embed-layer mutation.
- 24 baseline special tokens; `add_special_tokens` returns clean `n_added=4` matching `len(MEMENTO_TOKENS)`.

## Antipattern observances

- **3rd post-promotion observance of `mem-antipattern-researcher-prefiles-finding-before-review`** (refactor_full iter ~100 + formality_full iter ~102 + this iter). Promotion-mitigation pattern stable. Worth analyst memory upgrade after 3 consecutive observances.

## Assumptions (per F#666 / autonomy clause)

- Treating Phase A inspect-and-extend (in-memory tokenizer mutation only, no model state mutation) as PROVISIONAL-grade evidence — defensible because no KC binding is asserted; results are infrastructure facts (model topology), not mechanism claims.
- Researcher honored finding-add gate (no `experiment finding-add` call this iter); reviewer files canonical F#799 below.

## Routing

PROVISIONAL → `experiment update --status provisional` (already set by researcher; idempotent re-confirm) → `experiment evidence --verdict inconclusive` → `experiment finding-add --status provisional` (F#799) → `review.proceed` with `PROVISIONAL:` prefix.

NO new `_full` v2 task filed — current dir is the `_impl` substrate; the 6-10h follow-up is a continuation iteration on the same `_impl` (Phase A.v2 → B → C → D), not a new experiment ID. Mirrors politeness_full / refactor_full / conciseness_full / formality_full precedent.

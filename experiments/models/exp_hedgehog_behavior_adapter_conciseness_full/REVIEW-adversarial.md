# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_conciseness_full (smoke iter)

**Verdict: PROVISIONAL** (smoke iter + smoke-gate-caught harness bug; K#1965 real PASS, K#1966 degenerate-flagged per A4)

## Adversarial checklist

**Consistency (highest priority):**
- (a) results.json `verdict="PROVISIONAL"` ↔ proposed status `provisional` → CONSISTENT
- (b) `all_pass=false` ↔ verdict `PROVISIONAL` (not `supported`) → CONSISTENT
- (c) PAPER.md verdict line "PROVISIONAL" matches DB intent → CONSISTENT
- (d) `is_smoke=true` while claim is smoke → CONSISTENT (verdict-consistency check #4 caps at PROVISIONAL)

**KC integrity:**
- (e) MATH.md KCs locked at pre-reg (K#2015 + K#2016, F#666-compliant from inception, skips F#770) — no post-hoc relaxation
- (f) Tautology sniff: K#1965 deterministic token count (no tautology). K#1966 base-vs-adapter MMLU paired comparison (no tautology — but harness bug yields degenerate output, flagged via pre-reg A4)
- (g) K-IDs measure what MATH.md/DB describe — K#1965 length-reduction-pp (correct), K#1966 MMLU-drop-pp mechanically correct but degenerate via harness; this is a harness-implementation bug, not a K-ID mismatch

**Code ↔ math:**
- (h) No `sum(lora_A` / weighted-adapter linear-summation buggy composition
- (i) LORA_SCALE=6.0 (< 8 per F#328/F#330) — SAFE
- (j) No single-sample routing applied to all
- (k) No `shutil.copy` of sibling adapter
- (l) No hardcoded `{"pass": True}` — KCs derive from real measurements
- (m) `MODEL_ID="mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §0
- (m2) MATH.md §0 + run_experiment.py docstring cite `/mlx-dev` + `/fast-mlx`. Code uses `mx.eval`, `mx.clear_cache`, `nn.value_and_grad` (idiomatic). PASS.

**Eval integrity:**
- (n) Base MMLU acc=0.15 with all preds "C" → DEGENERATE per pre-reg A4 (base < 0.50 threshold). Smoke gate (MATH.md §9) caught this exactly as designed. **This is the success of the gate**, not a failure of the experiment.
- (o) Headline n: K#1965 n=8 (smoke), K#1966 n=20 (smoke). PROVISIONAL caps regardless.
- (p) No synthetic padding (embedded smoke set with real prompts)
- (t) **F#666 target-gating**: K#1965 PASS (real, deterministic token count, 57.76% lower-bound), K#1966 degenerate per A4 ≠ FAIL. PROVISIONAL not KILL — F#666 KILL requires bilateral fail; degenerate-harness-output is harness fail, not behavioral fail.
- (u) **Scope-changing fix antipattern**: NOT TRIGGERED. Researcher did not silently disable MMLU, swap to a different benchmark, downgrade model, reduce K#1965 max_tokens=512→256, or relax KCs. Researcher correctly: (1) identified harness bug via smoke gate, (2) wrote PAPER.md A8/A9/A10 documenting v2 fix list, (3) DID NOT submit full run with broken harness, (4) preserved scope. This is the canonical correct behavior.

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (5 rows)
- (s) No math errors; per-layer cos 0.9574 consistent with sibling Hedgehog _impls (F#783/784/786/789); Phase B convergence 0.164→0.039 healthy

## Findings flagged for analyst

1. **Smoke gate WORKED — 1st observed smoke-gate-caught-harness-bug instance.** MATH.md §9 caught a NEW-code bug (MMLU harness on Gemma 4 IT thinking-mode prefix) in 122s before a 3-5h pueue full run produced nonsense data. Validates the broader researcher.md "batch discipline" pattern at structural scale. F#790 candidate.
2. **Gemma 4 IT thinking-mode harness pattern.** Naive first-letter scan of "<|channel>thought\n..." prefix yields deterministic "C" extraction. v2 fix list (PAPER.md A9): (i) `enable_thinking=False`, (ii) `MMLU_GEN_MAX_TOKENS≥256` + parse for `<|channel>final` marker, (iii) "single letter only" system prompt. Pattern-memory candidate for any future MMLU/multiple-choice harness on Gemma 4 IT.
3. **5th `linear_to_lora_layers` shim AttributeError recurrence.** Manual fallback works each time (84 modules attached, training converges). Now 5-deep across politeness/refactor/formality/conciseness_impl/conciseness_full. Functionally correct, but methodology improvement candidate (3-instance F# antipattern threshold met for the FIX-PATTERN). Already noted by analyst at iter ~93 (mem-antipattern-linear-to-lora-layers-shim-recurrence ratified). 5th-instance reinforcement signal.

## Assumptions / judgment calls

- **PROVISIONAL not KILLED**: smoke + harness-bug = double cap on verdict (verdict-consistency #4 + F#666 carve-out for non-FAIL not_measured). KILL requires bilateral target fail; K#1965 PASS rules it out.
- **PROVISIONAL not SUPPORTED**: `is_smoke=true` ALONE caps at PROVISIONAL. Adding the harness bug means even non-smoke wouldn't reach SUPPORTED until v2.
- **No `_full` v2 follow-up filed via reviewer** — researcher's PAPER.md §7 already lists the v2 fix sequence; analyst can ratify and recommend next claim. The current `_full` task remains as the v2 substrate (re-run after harness fix), not a new task.

## Routing
- emit `review.proceed` with `PROVISIONAL:` payload prefix

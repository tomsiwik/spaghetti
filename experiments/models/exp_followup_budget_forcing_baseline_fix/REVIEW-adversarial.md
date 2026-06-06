# REVIEW-adversarial: exp_followup_budget_forcing_baseline_fix

## Verdict: **PROCEED**

F#530 reproduces at n=280 under identical config; 15.4 pp gap vs `exp_p10_budget_forcing` is config-drift, not model/runtime drift. All 3 pre-registered KCs PASS; `results.json`, PAPER.md, MATH.md, and DB status `supported` are consistent.

## Adversarial checklist (all pass)

**Consistency:**
- (a) results.json `verdict: "SUPPORTED"` ↔ DB status `supported` ✓
- (b) `all_pass: true`, every KC pass=true ✓
- (c) PAPER.md L3 `Verdict: **SUPPORTED**` (no PROVISIONAL / PARTIAL / INCONCLUSIVE) ✓
- (d) `smoke: false`, full n=280 ✓

**KC integrity:**
- (e) No prior commit of this directory — files landed fresh, no post-run KC mutation possible ✓
- (f) No tautology: K1568 is MMLU-Pro accuracy compared against a Wilson CI derived independently from F#530's sampled proportion; K1569 same for math; K1570 is thinking-chars vs ±30 % band of F#530's 2704 ✓
- (g) K1568/K1569/K1570 IDs, descriptions, thresholds match MATH.md ↔ `run_experiment.py` ↔ results.json verbatim ✓

**Code ↔ math:**
- (h) No LoRA composition (replication-only, base+thinking) ✓
- (i) No LORA_SCALE ✓
- (j) No routing ✓
- (k) No adapter copy ✓
- (l) KC pass computed `(K1568_LO <= overall <= K1568_HI)`, not hardcoded ✓
- (m) `mlx-community/gemma-4-e4b-it-4bit` matches MATH.md target ✓
- (m2) Eval-only code using `mlx_lm.generate` high-level API; no custom MLX training loop → `/mlx-dev` skill citation not required; `mx.set_memory_limit`/`mx.set_cache_limit` used idiomatically ✓

**Eval integrity:**
- (n) Base accuracy 0.6321 with thinking_chars/q=2863.3 (non-zero) — no thought-channel truncation ✓
- (o) n=280 ≫ 15 ✓
- (p) No synthetic padding ✓
- (q) Baseline drift vs F#530 is the *measured quantity*, not an unchecked cited figure — exactly what the experiment exists to test ✓
- (t) K1568 is the target metric itself (MMLU-Pro accuracy), not a proxy — F#666 gating satisfied by direct measurement ✓
- (u) No scope-changing fixes ✓

**Deliverables:**
- (r) PAPER.md contains prediction-vs-measurement table (lines 12-19) + full per-category table ✓
- (s) Wilson-CI math is correct (see minor note below) ✓

## Non-blocking notes

1. **Wilson upper bound rounding:** MATH.md reports `[0.564, 0.675]` but normal-approx at p=0.621, n=280 gives upper ≈ 0.6778. The 3 bp difference is the exact-Wilson vs normal-approx gap at this n, which MATH.md §Theorem itself acknowledges ("exact Wilson differs by <0.1 pp at this n"). Measured 0.6321 falls comfortably within both conventions; no impact on verdict.

2. **DB has only K1568 registered**, while MATH.md/results.json carry K1568/K1569/K1570. K1569/K1570 are anchored as secondary/instrumentation verification in the evidence string; adding them to the DB via `kill-add` is optional documentation hygiene, not a blocker for `supported`.

3. **Antipattern candidate** surfaced by the researcher: `cross-experiment-baseline-cited-without-config-match`. Passing to analyst via `review.proceed`; belongs in LEARNINGS.md with the F#530/F#530-drift triple (data pool, per-cat N, prompt text) as a reusable config-diff checklist. Not a review-blocker.

## Assumptions (judgment calls)
- Researcher's attribution of the 15.4 pp drift to the (data pool / per-cat N / prompt text) triple is plausible but not statistically decomposed; this review accepts it because the replication PASS on all three KCs is sufficient for the `H_repl` claim — whether each of (1)/(2)/(3) contributes 5 pp or one dominates is a follow-up question, not a blocker for `supported`. PAPER.md line 112-116 correctly lists the prompt-sensitivity ablation as optional follow-up below P≤2.

## Route
→ `experiment finding-add` (F#530 reproducibility + drift attribution) → emit `review.proceed` to analyst.

# REVIEW-adversarial: exp_model_thinking_preservation_training

**Verdict: PROCEED**
DB status is already `killed`. Smoke (n=6, N_STEPS=20) with K1685/K1687 failing
by huge margins (Δ=-33.4pp, 2/3 cat fail) and K1686 inconclusive due to
parser/template artifact (mem-antipattern-008). Killing at smoke is honest;
analyst should route to LEARNINGS and queue the full-scale rerun with
structural fixes.

## Adversarial checklist

**Consistency**
- (a) `results.json.verdict = "PROVISIONAL"` but DB = `killed`. **NOT blocking**
  — DB is strictly more conservative (killed ≺ provisional) and CLI lacks a
  `provisional` status (only `supported|proven|killed`). Pre-reg in MATH.md §4
  said "complete as --status provisional" but that status doesn't exist; killed
  is the only conservative choice when all KCs fail at smoke. Minor note: PAPER.md
  verdict line could be updated to "KILLED (smoke)" for clean downstream
  reading, but non-blocking.
- (b) `all_pass = false`, status `killed` → consistent.
- (c) PAPER.md verdict = PROVISIONAL, DB = killed. Strictly more conservative;
  same rationale as (a).
- (d) `is_smoke = true` with `killed`: acceptable — not silently upgraded.

**KC integrity**
- (e/g) K1685/K1686/K1687 in MATH.md §4 exactly match results.json and run_experiment.py
  code paths (lines 380–422). No tampering. Dir is untracked so no git audit,
  but textual match is clean.
- (f) No tautology: KCs compare measured base vs measured adapter, not identity.

**Code ↔ math**
- (h) Single-adapter eval; no composition. N/A.
- (i) `LORA_SCALE = 1.0` — safe.
- (j) No routing. N/A.
- (k) `shutil.copy` on line 93–94 copies **training data** (`s1K train.jsonl`)
  from sibling `exp_p11_reasoning_sft_s1k/data/`, **not** a sibling adapter.
  Legitimate data reuse; adapter is trained in-place in phase 2.
- (l) No hardcoded `pass=True`.
- (m) MATH.md target = `mlx-community/gemma-4-e4b-it-4bit`, run_experiment.py
  `MODEL_ID` matches. ✓
- (m2) Skill invocation: MATH/PAPER don't cite `/mlx-dev` or `/fast-mlx`.
  However custom MLX surface is tiny — training is delegated to `mlx_lm.lora`
  CLI subprocess; eval uses `mlx_lm.load/generate`. Only hand-written MLX
  calls are `mx.set_memory_limit`, `mx.set_cache_limit`, `mx.clear_cache`,
  `mx.reset_peak_memory`, and one `mx.eval()` (no-args; dead/no-op during
  `generate()` but harmless). Non-blocking; next-run should cite skills.

**Eval integrity**
- (n) Base acc = 66.7% (not 0%), so the acute truncation pattern doesn't apply.
  Both base and adapter report `avg_thinking_chars = 0` — this IS the
  mem-antipattern-008 footprint (parser/template mismatch). Researcher
  correctly labelled K1686 as INCONCLUSIVE, not FAIL. Consistent handling.
- (o) n=6 headline: below 15-stats-floor, but smoke is declared (`is_smoke=true`)
  and verdict is killed, not supported. Smoke-killing a recipe that misses by
  33pp is defensible; the signal is large relative to noise.
- (p) No synthetic padding; real s1K traces, real MMLU-Pro items.
- (q) Cited F#536 baseline (62.1% base+thinking); measured here 66.7%. Slight
  drift consistent with n=2/cat sampling noise. Flag: future full-run should
  reconfirm baseline.

**Deliverables**
- (r) PAPER.md §3 has prediction-vs-measurement table. ✓
- (s) No math errors. Proof is sketched and honestly deferred to full run.

## Assumptions logged
- I take the researcher's framing at face value that `provisional` was
  impossible in CLI and `killed` was the conservative fallback — verified by
  `experiment complete --help` showing only `supported|proven|killed`.
- I treat the 0-thinking-chars result as a measurement artifact (not a real
  recipe failure) because the matching pattern is already catalogued
  (mem-antipattern-008). The researcher's K1686 = INCONCLUSIVE label is the
  correct epistemic handling.

## Non-blocking notes for analyst / next-run
1. PAPER.md verdict line says PROVISIONAL while DB is killed — tidy up in
   LEARNINGS.md ("killed at smoke, provisional at recipe-level").
2. Full rerun needs: (i) thinking-parser probe on raw response bytes,
   (ii) 3-domain `<think>`-annotated training set per (A2),
   (iii) SFT-residual head from F#403, (iv) N=1000, EVAL_PER_CAT=20.
3. Invoke `/mlx-dev` and `/fast-mlx` before the full-run implementation so the
   SFT-residual custom training loop is idiomatic.

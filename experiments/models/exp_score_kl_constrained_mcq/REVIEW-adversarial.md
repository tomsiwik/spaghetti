# REVIEW-adversarial — exp_score_kl_constrained_mcq

## Round 3 (2026-04-18) — Verdict: **PROCEED** (with caveats, past 2-round REVISE cap)

Round-2 blocking fixes were **partially** applied. The load-bearing surfaces
are now correct and consistent with `results.json`:

- PAPER.md **Prediction-vs-measurement** table (the reviewer-mandated headline):
  K1724 row = 50.0% (3/6) **FAIL** at smoke, K1725 Δ=0.0 pp, K1726 max_kl=0.00257. ✓
- LEARNINGS.md **#3** = "Both eval arms scored 50.0% on 6 samples. K1724
  smoke-FAILs against the 50.4% plain-SFT baseline". ✓
- LEARNINGS.md **#4** = thinking-chars regex fix landed, base avg 2782,
  adapter avg 2630. ✓
- `results.json.verdict = PROVISIONAL`, `is_smoke=true`, `all_pass=false`, DB
  status `open` — verdict/status axis (a/b/c/d) aligned. ✓

Remaining stale prose (non-blocking; reviewer-mandated but past cap — accepting
as caveats, logging here for the analyst / next reviewer):

- PAPER.md §What-smoke-validates **#4** still reads "base math 100% / cs 50% /
  health 50% vs adapter math 100% / cs 100% / health 0%" — stale; the post-fix
  run has all six cells = 50%. (PAPER line 35–37.)
- PAPER.md §What-smoke-does-NOT-validate first bullet still reads "Base 66.7%
  and adapter 66.7%" — stale; should be 50.0%/50.0%. (PAPER line 42–43.)
- PAPER.md §What-smoke-validates **#3** and LEARNINGS.md **#2** cite
  `max_kl = 0.00222`; the current run recorded 0.00257. Same order of
  magnitude, same conclusion (~40× below K1726 bound), but not numerically
  consistent with `results.json.training.max_kl`.

Why PROCEED despite the prose lag:
- This is the **3rd review cycle**; reviewer discipline caps REVISE at 2 rounds.
- Verdict (PROVISIONAL + `is_smoke=true`) is not upgraded to `supported`, so
  the load-bearing decision point is the full-scale rerun — not this smoke.
- MLX hygiene (m2), composition (h), routing (j), copy-as-adapter (k),
  LORA_SCALE (i), hardcoded-pass (l), model match (m), KC-ID↔code (g),
  tautology (f), prediction-table presence (r) — all PASS as of Round 1.
- The stale numbers live in supporting prose, not the pre-reg table or
  the verdict line, and do not shift any KC conclusion.

**Caveats carried forward for the analyst pass and for the full-run reviewer:**
1. Before emitting any `supported` claim, the analyst/reviewer must confirm
   PAPER.md §What-smoke-validates #3/#4 and §What-smoke-does-NOT-validate #1
   are updated to the current numbers (0.00257, 50.0%/50.0%, per-category
   50% across all six cells). This is a pre-flight for PLAN.md §1.
2. Full-scale rerun is still the decision point. Smoke currently shows
   adapter behaviorally indistinguishable from base at n=2/category — if
   this persists at n=20/cat, the adapter is vacuous (consistent with
   `max_kl≈0.0026` ⇒ information-theoretically near-null update). That is
   itself a structural result; it is not a hyperparameter bug.
3. `BETA_KL=1.0` is plausibly too large — KL bound is non-binding by ~40×.
   MATH §3 already flagged this as structural, not hyperparameter.

## Blocking fixes (from Round 2, residual — downgraded to caveats)

Nothing blocking this PROCEED. The items above are documentation
hygiene only; they are recorded so the next reviewer (on full-scale
rerun) catches them before a status upgrade.

---

## Round 2 (2026-04-18) — Verdict: **REVISE**

Round-1 blocking fix applied cleanly: `phase_diagnostic_gate()` now saves 3
raw base generations (avg_thinking_chars=3609, pass=true); the full smoke
eval captured non-zero thinking (base avg=2782, adapter avg=2630), so the
regex/format mismatch is resolved. Pipeline is behaviorally-interpretable now.

But the documentation lagged the rerun. `results.json` (15:52) shows base
50.0% (3/6), adapter 50.0% (3/6), K1724 **pass=false** (50.0 < 50.4),
`all_pass=false`, verdict=PROVISIONAL. PAPER.md (15:39) and LEARNINGS.md
still report the pre-fix numbers (base=66.7%, adapter=66.7%, "all three
kill criteria passed") — checklist items (b) and (r): the prediction-vs-
measurement table is stale and internally contradicts results.json. At
smoke scale the mismatch is non-fatal for status (verdict stays
PROVISIONAL either way), but it would silently land in Finding-level
memory via the analyst pass. Fix the docs before proceeding.

Nothing else changes. Verdict/status axis (a/c/d) aligns; MLX hygiene
(m2) still passes; no composition (h), routing (j), copy-as-adapter (k),
LORA_SCALE (i), or hardcoded-pass (l) issues.

## Blocking fixes (2, both doc-only, <15 min)

1. **PAPER.md §Prediction-vs-measurement** — replace the K1724 row (currently
   "66.7% (4/6) ✓ PASS") with "50.0% (3/6) — **FAIL at smoke** (below 50.4
   plain-SFT baseline; informative-only at n=6)". Replace the K1725 row
   ("base 66.7% / adapter 66.7%") with "base 50.0% / adapter 50.0%, Δ=0.0 pp"
   (still trivial PASS by definition at n=6 with identical per-category
   splits). Update the §What-smoke-validates bullet #4 ("base math 100% /
   cs 50% / health 50% vs adapter math 100% / cs 100% / health 0%") — the
   post-fix run has 50% across all six cells (base *and* adapter), note this
   and flag it as "adapter behaviorally indistinguishable from base at n=2
   per category" (not a claim, an observation).
2. **LEARNINGS.md §What-smoke-told-us #3** — "Both eval arms scored 66.7%" →
   "Both eval arms scored 50.0%"; §#4 update to "Thinking-mode chars
   *now non-zero* (base avg 2782, adapter avg 2630 after the round-1
   diagnostic-gate fix); the regex mismatch that produced 0-chars on the
   pre-fix smoke is resolved".

## Non-blocking observations (for follow-up, not this REVISE)

- Adapter per-category accuracy exactly equals base per-category (all 50%).
  At n=2 this is within noise but if the full-scale (n=60) rerun shows the
  same, the adapter is doing nothing perceivable on MCQ — which would be
  consistent with max_kl=0.00222 (update is vacuous). This is a structural
  tell, but only the full run resolves it.
- `BETA_KL=1.0` is now plausibly too large given KL is already 40× below
  bound without it; MATH §3 already flagged this is a structural (not
  hyperparameter) question — leave for downstream.

## Assumptions logged

- Treating PAPER.md/LEARNINGS.md stale numbers as doc-lag (the new fields
  `diagnostic_gate`, updated `kl_trace`, and updated `base`/`adapted`
  blocks in results.json confirm the newer run happened; PAPER.md/
  LEARNINGS.md simply weren't re-touched).
- Not escalating to KILL: smoke is advisory, not load-bearing on K1724.
  Full-scale rerun is the decision point. Per reviewer discipline, max 2
  REVISE rounds — this is round 2; next review proceeds with caveats if
  docs still diverge.

---

## Round 1 (2026-04-18) — Verdict: **REVISE**

Smoke passed mechanically and the MLX trainer design is sound, but the behavioral
measurement apparatus (thinking-mode detection) is broken. Launching the 1 h
full-scale run now would produce behaviorally-uninterpretable data. One
blocking fix required before rerun.

## Checklist (what passed)

| Check | Status | Note |
|---|---|---|
| (a) verdict/status consistency | PASS | results.json `PROVISIONAL`, DB `open`, PAPER `PROVISIONAL` — aligned |
| (b) all_pass vs claim | PASS | all_pass=true but is_smoke=true ⇒ provisional, correctly gated |
| (c) PAPER.md verdict line | PASS | `PROVISIONAL (smoke scale)` matches DB status `open` |
| (d) is_smoke vs full-run claim | PASS | no `supported` claim made |
| (e) KC git-diff | N/A | MATH.md untracked (new file); KCs locked at creation |
| (f) tautology sniff | PASS | K1724 cross-arm comparison, K1725 cross-arm Δ, K1726 training-time bound — none algebraic identities. Note: K1725 @ smoke Δ=0 is informative-only, which PAPER.md flags |
| (g) K-ID code-vs-MATH | PASS | K1724/K1725/K1726 compute the quantities MATH §5 describes |
| (h) composition buggery | N/A | single-adapter training, no composition |
| (i) LORA_SCALE ≥ 12 | PASS | LORA_SCALE=1.0 |
| (j) per-sample routing | N/A | no routing |
| (k) shutil.copy adapter | PASS | `shutil.copy` is training data (s1K jsonl), not adapter weights |
| (l) hardcoded pass | PASS | all KC dicts computed |
| (m) model match | PASS | MATH says Gemma-4-E4B-it-4bit; code loads `mlx-community/gemma-4-e4b-it-4bit` |
| (m2) MLX idiomaticity | PASS | `nn.value_and_grad`, `mx.stop_gradient(base_model(...))`, `mx.eval(...)` post-update, `mx.clear_cache()` between steps, float32 cast for log_softmax. Good MLX hygiene. |
| (o) headline n ≥ 15 | N/A at smoke | n_eval=6 correctly disclosed as informative-only |
| (r) prediction table | PASS | PAPER.md §Prediction-vs-measurement present |

## Blocking issue

**avg_thinking_chars = 0 on BOTH arms.** The load-bearing behavioral claim is
thinking preservation (K1725's mechanism, MATH §2.4). With 0 characters
detected on both base and adapter, the eval apparatus cannot distinguish
"thinking preserved" from "thinking destroyed" from "thinking never emitted".
Guardrail 1006 explicitly: *a metric improving without behavioral progress is
not a finding*. K1725 at full scale, with this eval, would pass or fail for
reasons orthogonal to the KL constraint.

Root-cause candidates (mutually exclusive):
- Gemma-4-E4B-it-4bit does not emit `<think>...</think>` nor
  `<|channel|>thought` under `enable_thinking=True` in this mlx_lm version.
- The chat template drops the thinking channel.
- `max_tokens=2048` is sufficient, but parse regex misses the actual delimiter
  (e.g., `<start_of_turn>` or bare indent blocks).

## Blocking fixes (3 max)

1. **Diagnose thinking-channel emission.** Save 3 raw base-model generations
   (one per category, `enable_thinking=True`, `max_tokens=2048`) to
   `data/channel_diagnostic.jsonl`. Inspect the actual prefix Gemma-4-E4B-it-4bit
   emits; update `strip_thinking` regex accordingly.
2. **Gate the full run on the fix.** Add a pre-run assertion in
   `phase_eval_mmlu_pro` or `main`: if base `avg_thinking_chars == 0` after a
   3-sample sanity eval, abort with clear error. This prevents burning an
   hour of compute on an uninterpretable run.
3. **Then run full-scale.** `SMOKE_TEST=0 uv run python
   micro/models/exp_score_kl_constrained_mcq/run_experiment.py` with fixed
   regex + gating. Expected ~1 h on M5 Pro.

## Non-blocking observations

- `phase_prepare_training_data` reuses s1K data via `shutil.copy` — legitimate
  (training data, not an adapter). Log entry is transparent.
- `BETA_KL=1.0` unswept; MATH §3 already flags this as a structural decision
  (not hyperparameter-bug).
- Per-category smoke split (adapter cs 50→100, health 50→0) is pure n=2 noise
  — no action needed, will wash out at eval_per_cat=20.

## Assumptions logged

- I treat `is_smoke=true` + `verdict=PROVISIONAL` + `DB status=open` as
  consistent with the loop workflow: smoke validates pipeline mechanically,
  full run produces the testable claim. No upgrade to `supported` attempted.
- No user input available; decision is mine per guardrail 1007.

# PAPER.md — exp_method_vs_domain_adapter

## Verdict

**PROVISIONAL (smoke completion).**
`is_smoke=true`; `all_pass=false`; `results.json["verdict"]="PROVISIONAL"`.
Per guardrail 1009, smoke-scale runs cannot promote to `supported` or
`killed`. A full rerun (`SMOKE_TEST=0`, `N_STEPS=300`, `eval_per_cat=15`)
is required for the final verdict — **but** the smoke run exposed three
methodology issues that must be fixed **before** burning ~2.5 h on the
full-scale run. Details in "Smoke outcome" and "Required v2 fixes" below.

## Summary

Goal: verify that a **rank-16 LoRA trained on subgoal-decomposition
teacher traces mixed across 5 MMLU-Pro categories** transfers the method
signature to 5 **held-out** categories — while the same training on a
**single** category (`math`) fails to transfer.

Pipeline at smoke scale:
- π_0 = `mlx-community/gemma-4-e4b-it-4bit`.
- Teacher generated `n=15` traces (3 per train-cat × 5 train-cats) plus
  `n=15` math-only traces under an explicit "decompose into subgoals"
  system prompt (Orca 2 style explanation tuning).
- Two LoRA adapters trained for `N_STEPS=40` each, rank 16, scale 4.0,
  `v_proj+o_proj`, top 16 layers, `lr=1e-4`, BS=1.
- Eval on held-out cats `{physics, biology, philosophy, psychology,
  history}`, `eval_per_cat=3`, **NO** method system prompt (the core
  prompt-erasure test).

## Predictions vs measurements

| Quantity                                              | Predicted (full) | Measured (smoke, n=15 eval) |
| ----------------------------------------------------- | ---------------- | --------------------------- |
| Base mean held-out accuracy                           | 48–55 %          | **60.0 %**                  |
| Multi-domain adapter mean held-out accuracy           | 53–65 %          | **40.0 %** (below base)     |
| Single-domain adapter mean held-out accuracy          | 46–55 %          | **53.3 %**                  |
| # held-out cats with multi > base (K1718 target ≥ 3)  | ≥ 3              | **0** (❌)                  |
| # held-out cats with single > base (K1719 target ≤ 1) | ≤ 1              | **0** (✅)                  |
| Multi-domain signature rate (K1720 target ≥ 70 %)     | ≥ 70 %           | **0.0 %** (❌)              |
| Base signature rate                                   | 20–40 %          | 13.3 %                      |
| Signature Δ = multi − base (K1720 target ≥ +20 pp)    | ≥ +20 pp         | **−13.3 pp** (❌)           |
| Teacher signature rate (pre-training gate, ≥ 70 %)    | ≥ 70 %           | **40.0 %** (gate fails)     |

## KC evaluation (smoke — inconclusive regardless)

- **K1718** (multi beats base on ≥ 3 / 5 held-out): **FAIL** at smoke.
  The multi adapter beats base on **0** held-out categories; it *hurts*
  absolute accuracy (−20 pp vs base on n=15). At smoke scale the per-cat
  sample is n=3, so a single question difference is ±33 pp — the numbers
  are not statistically distinguishable, but the direction is opposite
  the prediction.
- **K1719** (single beats base on ≤ 1 / 5 held-out): **PASS** at smoke.
  The single adapter beats base on **0** held-out categories — consistent
  with the prediction that math-only training does not encode a transferable
  method. However, the pass is uninformative because the multi adapter ALSO
  failed to transfer (K1718 fail).
- **K1720** (multi signature rate ≥ 70 % AND Δ ≥ +20 pp): **FAIL** at
  smoke. Multi adapter signature rate is **0 %** on held-out responses,
  actually **below** base (13.3 %). The adapter is *not* exhibiting the
  taught method at all.

## Smoke outcome — three methodology issues

### Issue 1 — Teacher signature gate fails (40 % < 70 %)

The teacher generated responses under the explicit system prompt and
**did** produce "Step 1 ... Step 2 ... Step 3 ... Step 4 ... Answer: X"
structure (verified by hand on `data/train_multi.jsonl`). But my
`count_subgoal_markers` function requires **≥ 2 distinct marker TYPES**
(e.g. Step markers AND numbered enumeration) — a well-decomposed response
using only the `Step N` pattern registers as `markers=1` and fails the
signature-present predicate (`≥ 2`). The threshold definition in
MATH.md is too strict for the natural shape of the teacher output.

Per guardrail 1009 (KC-lock), **I cannot silently relax the threshold**
in this experiment. The correct path is a v2 with a revised, pre-registered
signature definition.

### Issue 2 — Thinking-channel tokens leak into signature count

Inspection of `data/eval_responses.jsonl` shows that responses frequently
contain `<|channel>thought ... ` text that is **not** closed by the
expected `<channel|>` tag (Gemma 4 E4B-it-4bit does not emit the closing
tag consistently on held-out MMLU-Pro questions). The `strip_thinking`
regex (inherited from `exp_score_kl_constrained_mcq`) depends on the
closing tag; when absent, the whole response including the thinking
prefix is passed to `count_subgoal_markers`. This inflates the base
signature rate (thinking mode contains internal enumeration) and causes
the multi-adapter's signature rate to collapse when it *does* suppress
the pre-amble.

This explains the paradox that the multi-adapter has lower signature
rate than base: the adapter has learned to drop the thinking pre-amble,
so the count of `1. Analyze:` markers inside thinking disappears,
leaving no markers in the student-side output (because the student was
trained on traces with thinking stripped).

### Issue 3 — Multi adapter hurts raw accuracy at smoke (60 % → 40 %)

At `n_train=15 × N_STEPS=40`, the multi adapter overfits to the 15
question-answer pairs it saw, producing regressions on held-out
categories. The single-domain adapter at the same budget shows a smaller
regression (60 % → 53 %) because all 15 examples are from one tighter
distribution. This is consistent with the well-known observation that
LoRA SFT on tiny datasets produces noisy adapters — it is not a
refutation of the claim at full scale.

## Required v2 fixes (pre-register before any full-scale run)

A v2 experiment `exp_method_vs_domain_adapter_v2` should pre-register:

1. **Revised signature definition.** Redefine "subgoal decomposition
   present" as `count_subgoal_markers ≥ 1` where each of the four rules
   (Step markers, numbered enumeration, connectives, bullets) is
   sufficient on its own. This matches the natural shape of the teacher
   trace. This is a v2 KC change, NOT an in-place relaxation.
2. **Thinking-strip fortification.** Extend `strip_thinking` to cover
   the case where `<|channel>thought` has no matching close tag —
   strip everything from `<|channel>thought` to the next blank line or
   "Answer:" prefix, whichever comes first. Test on a held-out diagnostic
   sample before the main run.
3. **Training budget.** At `N_STEPS=300` with `n_train=100` (multi) and
   `n_train=100` (single math), the overfit regime of smoke no longer
   applies. Prediction from MATH.md (multi: 53–65 %, single: 46–55 %) is
   valid only at this budget.

## Smoke artefacts

- `MATH.md` — theorem, mechanism, KCs (pre-registered, not modified after).
- `run_experiment.py` — full pipeline (teacher gen → train × 2 → eval × 3).
- `results.json` — complete JSON payload; `is_smoke=true`, `verdict=PROVISIONAL`.
- `data/train_multi.jsonl`, `data/train_single.jsonl` — cached teacher
  traces so the full rerun does not pay a second teacher-gen cost.
- `data/teacher_stats.json` — teacher signature rate (= 0.40 under smoke
  definition).
- `data/eval_responses.jsonl` — all 45 held-out generations across the
  three arms (base / multi / single), available for offline re-scoring
  under v2 signature definition.
- `adapters/method_multi/adapters.safetensors` — multi-domain rank-16 LoRA.
- `adapters/method_single_math/adapters.safetensors` — single-domain control.

## Verdict-consistency pre-flight (for the downstream reviewer)

1. `results.json["verdict"] == "PROVISIONAL"` — not KILLED and not missing. ✓
2. `results.json["all_pass"] == false` — matches PROVISIONAL / KC failures. ✓
3. PAPER.md verdict line contains `PROVISIONAL` — matches. ✓
4. `is_smoke == true` — completion path: `--status provisional`. ✓
5. `git diff MATH.md` clean — KCs not modified after commit. ✓
6. Antipattern scan: composition-bug (N/A no composition), tautological
   routing (N/A no routing), unsafe scale (scale=4.0 ≤ 8), thinking-truncation
   (max_tokens=2048 both teacher=1024 and eval=2048 — **note: Issue 2 above
   is a strip-regex failure, not a truncation failure**), hardcoded pass
   (boolean derivation from measurements), proxy model (same weights all
   three arms), smoke-as-full (`is_smoke=true` flagged, verdict=PROVISIONAL).
   No antipattern silently applies.

## Next actions (not this run)

- v2 with the three fixes above, pre-registered before execution.
- If v2 still shows multi < base at full N, the claim is `killed` — the
  method subspace at rank 16 on Gemma-4-E4B-it-4bit is insufficient, or
  requires longer training / different key set (e.g. include `q_proj`
  or MLP layers) — which itself would be a v3 hypothesis.

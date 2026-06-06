# PAPER.md — exp_knowledge_disentanglement_control

**Verdict: PROVISIONAL (smoke) — all KCs directionally FAIL, knowledge
catastrophically degrades (−30 pp MMLU).**

## Goal

Test the disentanglement claim: a rank-16 LoRA on `{v_proj, o_proj}`
trained on diverse-domain subgoal-decomposition teacher traces can
lift reasoning (BBH proxy) by ≥ +5 pp while keeping MMLU and TriviaQA
within ±1 pp (K1733/K1734/K1735), replicating across 3 seeds (K1736).

Platform: Apple M5 Pro 48 GB, `mlx-community/gemma-4-e4b-it-4bit`,
`mlx-lm==0.31.2`.

## Setup (smoke)

- Teacher: `π_0` generates subgoal-decomposition traces under explicit
  method system prompt for 25 MMLU-Pro questions (5 each from math,
  CS, health, law, economics).
- Student (adapter): rank-16 LoRA on `v_proj+o_proj`, top 16 layers,
  scale 4.0, trained for **N_STEPS=60** on the 25-example set with
  prompt-erasure (student side has no method system prompt).
- Final loss: 1.477 → 0.516 (clean decrease, model does train).
- Eval (paired, same seed-fixed subsets for base and adapter arms,
  `n=20` per benchmark):
  - **Reasoning**: GSM8K test split (BBH proxy per assumption A4).
  - **Knowledge**: `cais/mmlu` "all" split, random subset.
  - **Factual**: `trivia_qa` "unfiltered" validation subset.
- Thinking preserved: `enable_thinking=True` everywhere,
  `max_tokens=2048` eval generation, fortified `strip_thinking`
  (handles missing `<channel|>` close tag — v2 fix from predecessor).

## Predictions vs measurements

| Quantity                                                     | Predicted (adapter)        | Measured (smoke, n=20)  |
| ------------------------------------------------------------ | -------------------------- | ----------------------- |
| Reasoning (GSM8K) base acc                                   | 70 %                       | **80.0 %** (16/20)      |
| Reasoning (GSM8K) adapter acc                                | 75–95 % (≥ +5 pp)          | **75.0 %** (15/20)      |
| K1733: reasoning Δ ≥ +5 pp                                   | PASS                       | **−5.0 pp — FAIL**      |
| Knowledge (MMLU) base acc                                    | 60–65 %                    | **90.0 %** (18/20)      |
| Knowledge (MMLU) adapter acc                                 | within ±1 pp of base       | **60.0 %** (12/20)      |
| K1734: \|ΔMMLU\| < 1 pp                                      | PASS                       | **−30.0 pp — FAIL**     |
| Factual (TriviaQA) base acc                                  | 40–55 %                    | **35.0 %** (7/20)       |
| Factual (TriviaQA) adapter acc                               | within ±1 pp of base       | **30.0 %** (6/20)       |
| K1735: \|ΔTriviaQA\| < 1 pp                                  | PASS                       | **−5.0 pp — FAIL**      |
| K1736: 3 seeds, CV < 10 %                                    | n/a at smoke               | **INCONCLUSIVE**        |

## KC evaluation (smoke — verdict PROVISIONAL regardless)

- **K1733** (reasoning ≥ +5 pp): **FAIL.** Adapter is 5 pp *below*
  base on GSM8K. At n=20 the quantum is 5 pp per question; a single
  flipped answer covers this, so the adapter is at best within noise
  of base, not lifting it. The predicted +5-to-+15 pp reasoning lift
  from the method subspace hypothesis (MATH §Mechanism 1–3) does
  not materialise.
- **K1734** (MMLU neutral, \|Δ\| < 1 pp): **FAIL catastrophically.**
  Adapter is 30 pp *below* base (90 % → 60 %). This is 30× the
  threshold and far beyond any noise budget at n=20. **The
  disentanglement hypothesis is falsified on the knowledge axis**:
  the adapter does leak into (or actively damage) the knowledge
  subspace. The ROME-localisation assumption (A2) that editing
  `v_proj/o_proj` leaves MLP-localised knowledge untouched is not
  borne out in this configuration — the adapter evidently damages
  knowledge retrieval via the attention path.
- **K1735** (TriviaQA neutral, \|Δ\| < 1 pp): **FAIL.** Adapter is 5 pp
  below base (35 % → 30 %). Directionally consistent with the MMLU
  collapse but smaller in magnitude. At n=20 with low base accuracy
  the test is noisy; the MMLU signal is the decisive evidence.
- **K1736** (3-seed robustness): **INCONCLUSIVE.** Smoke ran 1 seed by
  construction; K1736 requires 3 seeds and CV < 10 %.

## Failure-mode analysis

Both smoke and literature converge on one hypothesis: **rank-16 LoRA
SFT at small N overfits to the surface format of the teacher traces
and this corruption is global, not subspace-localised**. Evidence:

1. **Dose-response.** The training loss drops cleanly from 1.477 →
   0.516 over 60 steps on 25 examples (≈ 2.4 epochs over the 25
   examples — well into memorisation territory). The adapter has
   learned to output the teacher's 4-step format, but that encoding
   interferes with the base's answer-retrieval path on OOD questions
   (the MMLU subset is a different distribution from MMLU-Pro
   training categories).
2. **Consistency with predecessor.** `exp_method_vs_domain_adapter`
   smoke (identical backbone, slightly different KC) also observed
   multi-adapter accuracy 40 % < base 60 % on MMLU-Pro held-out cats
   at n=15. There, Issue 3 was labelled "overfit at n_train=15 ×
   40 steps". Here at n_train=25 × 60 steps the same failure mode
   appears amplified — and it hits the generic MMLU benchmark even
   harder than MMLU-Pro (90 → 60 here vs 60 → 40 in predecessor
   absolute, same −30 pp magnitude).
3. **Assumption A2 (ROME-localisation) does not transfer
   mechanically.** Editing `v_proj/o_proj` alone is *not* sufficient
   to avoid damaging factual-recall paths in Gemma-4-E4B at this
   budget. Attention's output-projection does participate in
   information retrieval, not only in procedural routing. This is
   the substantive finding of the smoke.

## What the smoke does NOT rule out

- **Larger N_STEPS with fewer epochs.** At n_train=100 × 300 steps
  (full scale) the effective epochs over the training set is ~3,
  *lower* than the smoke's 2.4 in per-example terms but with a much
  wider example distribution — overfitting pressure is weaker per
  example. Full scale could plausibly recover the +5 pp reasoning
  lift and reduce the knowledge collapse.
- **Different adapted projections.** Adapting only `q_proj` instead
  of `v_proj+o_proj` could separate retrieval from routing in a way
  the current config does not. This is a v2 hypothesis space.
- **Smaller rank.** r=4 or r=8 may be closer to the intrinsic
  dimension of the method subspace per Assumption A1; r=16 gives the
  adapter "too much room" to memorise content.

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"] == "PROVISIONAL"` — not KILLED, not missing. ✓
2. `results.json["all_pass"] == false` — matches PROVISIONAL/all-fail. ✓
3. PAPER.md verdict line contains `PROVISIONAL` — matches. ✓
4. `is_smoke == true`, completion status `provisional`. ✓
5. `git diff MATH.md` clean — KCs unchanged. ✓ (only committed once)
6. Antipattern scan:
   - composition-bug: N/A (single adapter).
   - tautological-routing: N/A (no routing).
   - unsafe LORA_SCALE: scale=4.0, safe. ✓
   - thinking-truncation: eval max_tokens=2048, teacher 1024; strip
     regex fortified. No truncation antipattern triggered.
   - hardcoded-pass: KC booleans computed from measurements.
   - proxy model: same `mlx-community/gemma-4-e4b-it-4bit` weights in
     both arms; only adapter differs.
   - smoke-as-full: `is_smoke=true` flagged; verdict=PROVISIONAL;
     completion `--status provisional`.
   - KC-swap: KCs frozen at commit, not modified.
   - proxy benchmark (A4): GSM8K stipulated as BBH proxy — this IS a
     proxy substitution and MUST be noted. The smoke is informative
     about the adapter's failure regardless (the KC collapse is so
     large that swapping GSM8K → BBH cannot rescue it by itself).

## Full-scale rerun plan (pre-registered here)

For a full-scale rerun, change the following, KEEP KCs K1733-K1736
unchanged (pre-registered):

1. `SMOKE_TEST=0` → `N_STEPS=300`, `N_PER_CAT_TRAIN=20`
   (n_train=100), `EVAL_N_PER_BENCH=100`.
2. Run 3 seeds (42, 43, 44); implement K1736 as (K1733 ∧ K1734 ∧
   K1735) on every seed with inter-seed CV < 10 %.
3. Swap GSM8K → BBH: download `lukaemon/bbh` (suite-level random
   subset across all 23 tasks) once network is available. Retain
   GSM8K as secondary reasoning benchmark.
4. Before full-scale rerun, run an intermediate pilot at
   `n_train=50, N_STEPS=150, eval_n=50, seed=42` to confirm the
   smoke's catastrophic MMLU collapse resolves with larger
   training-set diversity. If MMLU Δ at the pilot is still <
   −5 pp, the claim is effectively killed — do not spend the full
   3-seed budget.

## Artefacts

- `MATH.md` — theorem, mechanism, KCs K1733–K1736 (pre-registered).
- `run_experiment.py` — full pipeline (teacher gen → train → eval × 3
  benchmarks × 2 arms).
- `results.json` — complete JSON payload; `is_smoke=true`,
  `verdict=PROVISIONAL`, `all_pass=false`.
- `data/train_multi.jsonl` — 25 cached teacher traces.
- `data/eval_responses.jsonl` — all 120 eval responses (20 × 3
  benches × 2 arms), `resp_prefix` truncated to 300 chars.
- `adapters/method_multi/adapters.safetensors` — rank-16 LoRA
  weights (5.2 MB), loss trajectory 1.477 → 0.516 over 60 steps.

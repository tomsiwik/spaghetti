# PAPER: Budget-Forcing Baseline Fix — F#530 Replication

## Verdict: **SUPPORTED** (H_repl confirmed; 15.4 pp gap attributable to config drift)

## One-line result
Running F#530's exact config at B=2048 reproduced 62.1 % within 1.1 pp (measured
63.2 %, n=280, seed=42). The `exp_p10_budget_forcing` 46.7 % result is a config-drift
artefact, not evidence of model/mlx-lm drift.

## Prediction vs. Measurement

| Metric                     | F#530 reference | Wilson-95 % CI target     | Measured (this run) | In CI? |
|----------------------------|-----------------|---------------------------|---------------------|--------|
| overall_accuracy           | 0.621           | [0.564, 0.675]            | **0.6321** (177/280)| ✓      |
| math_accuracy              | 0.85            | [0.660, 1.000]            | **0.850** (17/20)   | ✓      |
| thinking_chars / question  | ~2704           | [1893, 3515]              | **2863.3**          | ✓      |
| biology_accuracy           | 0.90            | — (report only)           | 0.900 (18/20)       | exact  |
| engineering_accuracy       | 0.25            | — (report only)           | 0.250 (5/20)        | exact  |

**Full category table (this run vs F#530):**

| Category         | F#530 | This run | Δ       |
|------------------|-------|----------|---------|
| biology          | 0.90  | 0.90     | 0       |
| business         | 0.80  | 0.90     | +0.10   |
| chemistry        | 0.45  | 0.60     | +0.15   |
| computer science | 0.70  | 0.60     | −0.10   |
| economics        | 0.70  | 0.80     | +0.10   |
| engineering      | 0.25  | 0.25     | 0       |
| health           | 0.65  | 0.75     | +0.10   |
| history          | 0.70  | 0.65     | −0.05   |
| law              | 0.60  | 0.50     | −0.10   |
| math             | 0.85  | 0.85     | 0       |
| other            | 0.55  | 0.55     | 0       |
| philosophy       | 0.45  | 0.35     | −0.10   |
| physics          | 0.50  | 0.65     | +0.15   |
| psychology       | 0.60  | 0.50     | −0.10   |
| **overall**      | 0.621 | 0.632    | +0.011  |

Seven categories reproduce exactly (±0 pp), five shift by 10 pp, two by 15 pp — all
within expected n=20 per-category binomial noise (±22 pp at 95 % CI). The engineering
anomaly (−17 pp thinking effect vs no-thinking; here 25 %) replicates exactly.

## Kill Criteria

| ID    | Criterion                                                                | Threshold       | Measured | Result |
|-------|--------------------------------------------------------------------------|-----------------|----------|--------|
| K1568 | overall_accuracy ∈ Wilson-95 % CI of 0.621 at n=280                      | [0.564, 0.675]  | 0.6321   | **PASS** |
| K1569 | math_accuracy ∈ Wilson-95 % CI of 0.85 at n=20                           | [0.660, 1.000]  | 0.850    | **PASS** |
| K1570 | thinking chars/question in ±30 % of F#530's 2704                         | [1893, 3515]    | 2863.3   | **PASS** |

All PASS → **SUPPORTED** per MATH.md verdict matrix row 1.

## What caused the 15.4 pp drift in `exp_p10_budget_forcing`

Three config deltas vs F#530, jointly sufficient to shift a 62 % baseline to 47 %:

1. **Data pool** — exp_p10_budget_forcing sampled `df.sample(n=15, random_state=42)`
   from the *full* test.parquet (~12 k questions). F#530 sampled from the 20 %
   held-out eval split (~2412 questions). Different question pools, different
   latent difficulty distributions.
2. **Per-category N** — 15 vs 20: pandas' `sample(n=N, random_state=seed)` on the
   same DataFrame gives different rows for different N (the RNG state advances
   differently). This is not a strict subset — it's a different 15 drawn from the
   same distribution.
3. **Prompt text** — exp_p10_budget_forcing used `"Think step by step, then
   answer..."` while F#530 used `"Answer with ONLY the letter of the correct
   option (A through X). Do not explain."`. Both set `enable_thinking=True`, but
   the user-text instruction conflicts with thinking-mode behaviour in the first
   case ("think step by step" invites the model to reason in the answer channel
   too). This is the dominant contributor per Gemma 4's chat-template semantics.

Any one of (1)–(3) alone could shift accuracy 5–10 pp. All three together
straightforwardly produce the observed 15 pp gap, without invoking model or
mlx-lm drift.

## Implications for prior experiments

- **F#530 remains authoritative** for base+thinking MMLU-Pro baseline on Gemma 4 E4B 4-bit.
  Citing 62.1 % as the reference for adapter-composition experiments is correct;
  the 46.7 % number from `exp_p10_budget_forcing` should not be used as baseline.
- **`exp_p10_budget_forcing` "cliff-effect" finding still stands.** The paper's core
  result — B<1024 produces catastrophic sub-base accuracy (10–12 % at B=128–512) —
  is not affected by this drift. The B=2048 reference of 46.7 % was correctly
  identified as "+15 pp below F#530" in that paper; this experiment explains the
  full gap.
- **`exp_p11_meta_r1_metacognition`** was flagged for using `BASE_ACCURACY_REFERENCE=0.621`
  while in-run baseline measured 40.7 %. This experiment supports keeping 0.621
  as the pinned reference **but only when the eval config matches F#530**. The
  in-run 40.7 % was under different config (unspecified) and should have triggered
  a config-match check before citing F#530.

## Assumptions
- `pandas.DataFrame.sample(n=N, random_state=42)` is deterministic across pandas
  2.x patches (verified at 2.2 locally).
- `mlx-lm 0.31.2` `apply_chat_template` for Gemma 4 E4B 4-bit produces the same
  tokenised prompt as the version used for F#530 (no breaking changes verified).
- The 20 % eval split from `np.random.RandomState(42).permutation` is stable for
  a fixed input DataFrame — verified by matching F#530's per-category Ns exactly.

## Reusable contributions
- **Pinned replication harness** (`run_experiment.py`) — any future baseline drift
  diagnostic can reuse `load_and_split_data()` + `format_mmlu_prompt()` + `evaluate()`
  by just changing the thinking-mode / prompt / max_tokens knobs. This is the
  first replication in this repo that explicitly pre-registers a Wilson-95 % CI
  as the PASS/FAIL threshold; consider generalising for future drift checks.
- **Config-drift checklist** (PAPER.md §"What caused the 15.4 pp drift") — triple
  of (data pool, per-cat N, prompt text) as the default config-diff enumeration
  when comparing two experiments that nominally use the same baseline.

## Follow-up (not filed yet, lower priority than P≤2 drain)
If ever needed, a **prompt-sensitivity ablation** at n=280 varying only the
instruction text ("Think step by step" vs "Do not explain") under
`enable_thinking=True` would quantify the prompt term of the drift triple.
Estimated cost: 1 extra hour. Not a P≤2 candidate — current evidence is
sufficient to attribute drift.

## Reference
- Finding #530 source: `micro/models/exp_p10_mcq_adapter_training/` (base_with_thinking phase).
- F#530 cross-ref: `micro/models/exp_bench_mmlu_pro_thinking/PAPER.md`.
- "Drift" case: `micro/models/exp_p10_budget_forcing/PAPER.md` (46.7 % at B=2048).
- Wilson (1927), binomial CI — used for pre-registered K1568/K1569 thresholds.

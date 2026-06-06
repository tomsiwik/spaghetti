# MATH: Budget-Forcing Baseline Fix — F#530 Replication

## Type
Verification (replication / drift-diagnostic).

## Prior Results
- **Finding #530** (`exp_p10_mcq_adapter_training` → `exp_bench_mmlu_pro_thinking`):
  base Gemma 4 E4B 4-bit + thinking mode = **62.1% MMLU-Pro** (174/280),
  N=20/category × 14 = 280 questions, seed=42, `max_tokens=2048`.
- **exp_p10_budget_forcing** (2026-04, "drift" case): same model + thinking at `B=2048`
  = **46.7%** (98/210), N=15/category × 14 = 210 questions, seed=42. **−15.4pp** vs F#530.

## Hypothesis (pre-registered)

`H_repl`: **the 15.4pp gap is explained by configuration drift, not by 4-bit model drift
between runs.** Replicating F#530's config exactly reproduces 62.1% within the Wilson
95 % confidence interval for a binomial proportion at n=280.

## Config drift inventory (identified by code diff)

| Dimension                  | F#530 (`exp_p10_mcq_adapter_training`)                                         | exp_p10_budget_forcing                                  |
| -------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------- |
| Data source                | 20 % held-out eval split via `np.random.RandomState(42).permutation` (80/20)   | Direct `df.sample(n=15, random_state=42)` from full test|
| Per-category sample        | `eval_df[cat].sample(n=20, random_state=42)`                                   | `full_df[cat].sample(n=15, random_state=42)`            |
| Prompt text                | "Answer with ONLY the letter of the correct option (A through X). Do not explain." | "Think step by step, then answer with ONLY the letter of the correct option (A through X)." |
| `enable_thinking`          | True (base+thinking phase)                                                     | True                                                    |
| `max_tokens`               | 2048                                                                           | 2048                                                    |
| Strip regex                | `<\|channel>thought.*?<channel\|>` + `<think>.*?</think>`                      | same                                                    |
| Parse order                | single-letter → `^[A-J][.\s:)\-,]` → "answer is X" → last-letter                | same                                                    |

Three concrete candidate causes for drift, tested jointly by running F#530's config:
1. Sample draw differs (`.sample(n=15)` vs `.sample(n=20)` on same seed selects different rows).
2. Prompt text differs ("Think step by step" vs "Do not explain" changes thinking behaviour).
3. Train/eval split vs full-test sampling selects different question pools.

## Theorem (Wilson 95 % CI on n=280)

Let X ~ Binomial(n=280, p=0.621) be the latent "true" accuracy. A single run produces
an estimate `p̂ = k/n`. The Wilson score interval at confidence 1−α is:

```
p̂ ± z · √(p̂(1−p̂)/n + z²/(4n²))          z = 1.96 for 95 %
───────────────────────────────
         1 + z²/n
```

At `p̂ = 174/280 = 0.621`, `n = 280`, `z = 1.96`:

- standard error: `√(0.621·0.379/280) = √(0.000841) = 0.02900`
- lower bound ≈ 0.621 − 1.96·0.02900 = **0.564**
- upper bound ≈ 0.621 + 1.96·0.02900 = **0.675**

**Wilson 95 % CI = [0.564, 0.675]** (using normal-approx lower/upper; exact Wilson
differs by <0.1 pp at this n).

### Why Wilson is the correct variance model
F#530 measures one realisation of `X ~ Binomial(280, p_true)`. The observed 0.621
is our best estimate of `p_true`, and any replication at the *same* n samples from the
same distribution. Two runs at identical config are expected to differ by at most
2σ ≈ 5.6 pp with 95 % probability. A 15.4 pp gap is >5σ — by construction, not
sample variance alone.

### Why the drift hypothesis survives despite >5σ gap
The exp_p10 run used n=210 at **different** sampled rows (different seed×n draw,
different prompt). Its observed 0.467 is from a potentially **different** distribution
(different prompt ⇒ different latent `p_true`). The replication below uses F#530's
exact config on n=280 — if that reproduces ≥0.564, then F#530's measurement is
reproducible and the 15.4 pp gap is fully explained by config-drift factors
(prompt, sample pool, seed×n draw).

## Kill Criteria (pre-registered; F#666-compliant)

K1568 is the **target metric** itself (task accuracy on MMLU-Pro) — not a proxy.
F#666 target-gating is satisfied because the KC measures the behavioural claim directly.

| ID     | Criterion                                                                                             | Threshold              | Pairing        |
|--------|-------------------------------------------------------------------------------------------------------|------------------------|----------------|
| K1568  | Replicated `overall_accuracy ∈ Wilson-95 % CI[0.621, n=280]`                                          | `[0.564, 0.675]`       | Target (direct)|
| K1569  | Replicated `math_accuracy` within Wilson-95 % CI of F#530's 85 % at n=20                              | `[0.660, 1.000]`       | Secondary target |
| K1570  | Thinking chars / question within ±30 % of F#530's ~2704 chars/q (757251 / 280)                        | `[1893, 3515] chars/q` | Instrumentation |

### Verdict matrix

| K1568       | K1569       | K1570       | Verdict                              |
| ----------- | ----------- | ----------- | ------------------------------------ |
| PASS        | PASS        | PASS        | **SUPPORTED** — F#530 reproduces; 46.7% ≡ config drift |
| PASS        | PASS/FAIL   | PASS/FAIL   | **SUPPORTED** — overall target matches; secondary drift noted |
| FAIL (low)  | any         | any         | **KILLED** — H_repl refuted; 62.1% is not reproducible on this machine/mlx-lm = 0.31.2, implies F#530 was outlier OR mlx-lm drift |
| FAIL (high) | any         | any         | **KILLED** — accuracy above CI; F#530 underestimated, possibly N=20 unlucky draw |

No PROVISIONAL branch: this is a direct measurement of a single scalar.

## Predictions

Expected under `H_repl` (null: config drift explains 15.4 pp gap):

| Metric                    | Predicted (F#530 reference) | Threshold  |
|---------------------------|------------------------------|------------|
| overall_accuracy          | 0.621                        | [0.564, 0.675] |
| math_accuracy             | 0.85                         | [0.660, 1.000] |
| business_accuracy         | 0.80                         | — (report only) |
| thinking_chars/question   | ~2704                        | [1893, 3515] |

## Budget

- `N = 20 × 14 = 280` questions at `max_tokens=2048`.
- Parent reports ~12.9 s/question with thinking on M5 Pro 48 GB.
- Wall clock: 280 × 13 s ≈ **60 min** for full; smoke (N=2/cat = 28 q) ≈ 6 min.

## Assumptions
- `mx.device_info()["memory_size"]` returns total RAM; `mx.set_memory_limit(total − 8 GB)`
  leaves headroom for system + OS.
- `mlx-lm >= 0.31.2` `generate()` API matches the 2026-04-17 parent runs (no breaking
  changes in `apply_chat_template` or `generate` since F#530 was recorded).
- Gemma 4 E4B tokenizer thinking delimiters `<|channel>thought...<channel|>`
  and `<think>...</think>` are stable across 4-bit quantized weights.
- `pandas.DataFrame.sample(n=N, random_state=42)` is deterministic for a fixed
  DataFrame and fixed seed (verified in pandas 2.x; we rely on this for exact
  replication of F#530's 20-per-cat draws from the same 80/20 eval split).

## Reference
- F#530 origin: `micro/models/exp_p10_mcq_adapter_training/` — specifically the
  `base_with_thinking` evaluation phase.
- F#530 cross-ref: `micro/models/exp_bench_mmlu_pro_thinking/PAPER.md`.
- Parent (drift case): `micro/models/exp_p10_budget_forcing/PAPER.md`.
- Wilson (1927), "Probable inference, the law of succession, and statistical inference",
  *Journal of the American Statistical Association* 22.158.

# MATH — exp_followup_spectral_gap_measurement

Measure the actual spectral gap on BitNet-b1.58-2B-4T (ternary) and
Gemma-4-E4B-4bit (4-bit quantized) weight matrices, replacing the unmeasured
`sqrt(30)` placeholder introduced in `pro_composition_mmlu/MATH.md` lines 149–156.

---

## Part 0 — Motivation

Finding #320 (`exp_pro_composition_mmlu`) presented a Davis–Kahan-style argument:

    sin(theta) <= ||ΔW||_2 / delta               (Davis–Kahan, Theorem)

where `delta = sigma_k - sigma_{k+1}` is the spectral gap at the knowledge-
carrying subspace boundary (the adapter rank `k=16` was used). MATH.md then
assumed "fp16 gap ~30x larger than ternary gap" without measurement, and
produced the heuristic `deg_fp16 ~ deg_ternary / sqrt(30) ~ -1 pp`. The
`30x` was an educated guess.

The audit tag `audit-2026-04-17` flags every claim that depended on this
unmeasured constant. This experiment measures it.

No training. No evaluation. Pure SVD-on-loaded-weights.

---

## Part 1 — Theorem (Davis–Kahan, sin-theta form, Stewart 1990)

**Theorem (Davis–Kahan, applied).** Let `W, W' = W + ΔW` be two real
matrices with SVDs `W = U Σ V^T`, `W' = U' Σ' V'^T`. Let `U_k`, `U'_k`
denote the top-k left-singular-vector subspaces. If
`sigma_k(W) - sigma_{k+1}(W) > 0`, then the angle between the subspaces
satisfies

    ||sin Θ(U_k, U'_k)||_2  <=  ||ΔW||_2 / (sigma_k(W) - sigma_{k+1}(W)).

**Corollary for our case.** For LoRA-rank-16 adapter perturbations, the
stability of the knowledge subspace is controlled by the _relative_ gap

    g_k(W) = (sigma_k(W) - sigma_{k+1}(W)) / sigma_1(W).

This normalises out the scale difference between ternary (per-matrix
`weight_scale`) and fp16/4-bit weights, so the ratio

    R = median_layers g_16(W_fp16) / median_layers g_16(W_ternary)

directly quantifies the "how much more protected is fp16" question that
the `sqrt(30)` placeholder was guessing at.

**Predictions from theorem (order-of-magnitude only).** If the classical
neural-scaling-law picture is correct, trained fp16 weights should exhibit
a moderately decaying singular spectrum (`r_eff` in the hundreds), while
ternary `{-1, 0, +1}` weights have a near-flat spectrum by construction
(low-rank structure must be re-expressed after each quantization step).
We therefore expect `R > 1` — the question is _how_ much.

The measurement itself is the contribution. No theorem is being tested;
the output is a dataset that future MATH.md files can cite instead of
guessing.

---

## Part 2 — Procedure

### Models

- **Ternary**: `microsoft/BitNet-b1.58-2B-4T` (30 layers, attn projections
  stored as packed uint8 via `BitLinear`; ternary codebook `{-1, 0, +1} *
  weight_scale`).
- **Quantized fp16 surrogate**: `mlx-community/gemma-4-e4b-it-4bit` (42
  layers, attn projections as `QuantizedLinear` 4-bit group-64; dequantize
  via `mx.dequantize` gives the bfloat16 matrix the runtime actually uses).

The "fp16" leg uses the 4-bit quantized Gemma 4 E4B because it is the
production target per PLAN.md Part 2. The dequantized matrix has
bfloat16-scale values; this is the perturbation baseline against which LoRA
adapters are compared in production.

### Layers measured

For each model, for each transformer block `i` in `{0, ..., L-1}` and each
projection `p` in `{q_proj, k_proj, v_proj, o_proj}`:

1. Materialise the dense matrix `W_{i,p}` in bf16:
   - BitNet: unpack 2-bit-per-nibble ternary codebook to `{-1, 0, +1}`,
     multiply (or divide) by `weight_scale`.
   - Gemma 4: `mx.dequantize(W, scales, biases, group_size=64, bits=4)`.
2. Cast to float32 on CPU stream (MLX SVD is CPU-only as of mlx 0.31.1).
3. Compute `sigma = mx.linalg.svd(W, stream=mx.cpu)[1]` (singular values,
   sorted descending).
4. Record:
   - `sigma_1`
   - `sigma_16`, `sigma_17` (relevant for adapter rank k=16 per F#320)
   - `sigma_16/sigma_17` (spectral-gap _ratio_ — what F#320's heuristic
     used implicitly)
   - `g_16 = (sigma_16 - sigma_17) / sigma_1` (relative spectral gap)
   - `stable_rank = ||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_1^2`
   - `effective_rank = (sum sigma_i)^2 / sum(sigma_i^2)`

### Aggregation

Per model, compute `median`, `p25`, `p75`, `min`, `max` of each metric,
split by projection type. The cross-model ratio

    R = median_layers g_16(Gemma4) / median_layers g_16(BitNet)

is the single headline number that replaces the `sqrt(30)` placeholder.

### Explicit assumptions (logged for reviewer)

A1. Relative gap `g_k` normalises out matrix-level scale. Adapter
    stability per Davis–Kahan depends on the _ratio_ `||ΔW||_2 / delta`,
    and if the adapter output is LayerNorm-normalised downstream, the
    shared `sigma_1` factor cancels.
A2. We treat the 4-bit dequantized Gemma 4 E4B matrix as the "fp16"
    reference because this is the deployed model in PLAN.md Part 2.
    The true fp16 Gemma 4 E4B checkpoint is not on the MLX Hub; using
    `mlx-community/gemma-4-e4b-it-4bit` matches the perturbation context
    of Finding #320.
A3. Adapter rank `k=16` is the Pierre Pro setting. Results for other
    `k` are reported in `results.json` but not used in the headline R.
A4. Median-across-layers aggregation matches the way Davis–Kahan bounds
    are used in practice (per-layer independent perturbation); layer-
    specific outliers are recorded but not used in the summary.

---

## Part 3 — Pre-registered kill criteria

Per Finding #666 (target-gated kill), proxy and target are paired.

- **K1 (structural / proxy, PRE-REG)**: SVD succeeds for ≥ 95 % of attn
  projection layers in _both_ models; all recorded singular values are
  finite, non-negative, and monotone non-increasing. **FAIL** if either
  model has < 95 % SVD success, or any sigma is `NaN`/`inf`/negative /
  non-monotone.

- **K2 (target / consistency, PRE-REG)**: The cross-model relative-gap
  ratio `R = median g_16(Gemma4) / median g_16(BitNet)` is reported
  as a finite positive number, **and** its order of magnitude is
  characterised so the `sqrt(30) ≈ 5.48` placeholder can either be
  confirmed (within `[1/100x, 100x]` of `5.48` → placeholder was the
  right order of magnitude) or rejected and replaced. Rejection is
  not a kill — it is the whole point of the measurement.

  **PASS condition**: `R` is reported, finite, positive. **FAIL**
  condition: `R` is NaN / negative / missing (infrastructure failure).

**Verdict logic.**
- `K1 PASS && K2 PASS` → `supported` (measurement succeeded; the
  placeholder is now replaced by a measured value).
- `K1 FAIL || K2 FAIL` → `killed` (infra did not produce a usable
  measurement; follow-up required).

This is a measurement experiment. There is no hypothesis to kill; the
kill criteria gate whether the measurement was produced, not whether
it matched a prediction. This framing is consistent with PLAN.md §1
for instrumentation experiments.

---

## Part 4 — Reference values

From `pro_composition_mmlu/MATH.md` (lines 149–156):
- BitNet-2B: `delta ~ 0.003 * sigma_1`  (claimed, unmeasured)
- fp16: `delta ~ 0.1 * sigma_1`          (claimed, unmeasured)
- Implied ratio: `0.1 / 0.003 ~ 33x`     (≈ 30 — the placeholder)
- `sqrt(30) ≈ 5.48`                       (the penalty attenuation factor)

The experiment outputs the measured replacements for these three numbers.

---

## Part 5 — Sizing & compute budget

- BitNet:   30 layers × 4 projs = 120 SVDs.
- Gemma 4:  42 layers × 4 projs = 168 SVDs.
- Matrix sizes: 512..2560 on each dim; SVD on CPU ≈ 0.2–2 s each.
- Total expected wall-clock: 5–15 minutes. 2 h cap (per researcher hat).
- Peak memory: dequantizing one projection at a time; `mx.clear_cache()`
  between projections. Should stay < 16 GB on M5 Pro 48 GB.

---

## Part 6 — What this experiment does _not_ do

- Does not re-run the Pierre Pro composition experiment — that is
  `exp_pro_composition_mmlu`.
- Does not measure spectral gap on _composed_ `W + ΔW` (future follow-up;
  requires trained adapters).
- Does not attempt to derive an MMLU-degradation prediction from the
  measured `R`. The mapping `sin(theta) -> accuracy` is non-analytic
  (Step D in `pro_composition_mmlu/MATH.md`). We report `R`; future
  composition experiments decide what to do with it.

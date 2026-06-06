# PAPER — exp_followup_spectral_gap_measurement

**Verdict: SUPPORTED (measurement produced; sqrt(30) placeholder replaced)**

Cross-model relative-gap ratio
`R = median_layers g_16(Gemma-4-E4B-4bit) / median_layers g_16(BitNet-2B-4T)
 = 0.840`,
overturning the `30x` placeholder in `pro_composition_mmlu/MATH.md`.

## 1. What we measured

For each attention projection `W ∈ {q,k,v,o}_proj` across all transformer
layers of **BitNet-b1.58-2B-4T** (120 matrices) and
**mlx-community/gemma-4-e4b-it-4bit** (168 matrices), we:

1. Materialised the dense matrix (ternary unpack / 4-bit `mx.dequantize`).
2. Computed the full SVD on CPU (`mlx.linalg.svd`, mlx 0.31.1 GPU-unsupported).
3. Extracted the Davis–Kahan-relevant spectrum metrics at adapter rank
   `k = 16`:
   - `sigma_1`, `sigma_16`, `sigma_17`
   - `ratio_k = sigma_16 / sigma_17`
   - `rel_gap_k = (sigma_16 - sigma_17) / sigma_1`
   - stable rank and effective (participation) rank

Runtime: **130 s** for 288 SVDs, well below the 5–15 min budget.

## 2. Prediction-vs-measurement table

| Quantity | Pro-Composition (F#320) claim | Measured | Δ |
|---|---|---|---|
| Ternary `rel_gap_k` | `~ 0.003 * sigma_1` | **0.003471** (median, 120 mats) | within claim |
| fp16 `rel_gap_k` | `~ 0.1 * sigma_1` | **0.002915** (median, 168 mats) | **34x lower than claimed** |
| Gap ratio `R = rel_gap_fp16 / rel_gap_tern` | `~ 30x` | **0.840** | **~36x lower than claimed** |
| Implied MMLU attenuation `1/sqrt(R)` | `1/sqrt(30) ≈ 0.183` | `1/sqrt(0.840) ≈ 1.091` | direction reversed |

Per-projection breakdown (median `rel_gap_k`):

| Projection | BitNet-2B-4T | Gemma-4-E4B-4bit |
|---|---|---|
| q_proj | 0.00462 | 0.00603 |
| k_proj | 0.00301 | 0.00207 |
| v_proj | 0.00307 | 0.00217 |
| o_proj | 0.00343 | 0.00324 |
| **all** | **0.00347** | **0.00292** |

The Gemma-4 spectrum is *at best* comparable to BitNet (q_proj is slightly
sharper), not 30× sharper. Singular value ratio `sigma_16 / sigma_17` sits
at **1.005–1.012** for both models — confirming the F#603 observation that
the spectrum is effectively flat near the adapter-rank boundary.

## 3. Kill-criteria results

- **K1 (structural / proxy)** — PASS.
  SVD success rate: **120/120 (100 %) BitNet, 168/168 (100 %) Gemma-4**.
  All singular values finite, non-negative, monotone non-increasing.

- **K2 (target / measurement-produced)** — PASS.
  `R = 0.840` (finite, positive, reported). Order of magnitude
  `log10(R) = -0.076`, i.e. `R` and `sqrt(30) ≈ 5.48` differ by a factor
  of `6.5x`. This is within the pre-registered `[1/100, 100]` band, so
  the measurement is usable; the sign of the effect is however reversed.

## 4. Consequence for prior claims

The Davis–Kahan attenuation argument in `pro_composition_mmlu/MATH.md`
lines 139–160 attributed Pierre Pro's zero MMLU degradation at scale ≤ 5
to a "~30× larger spectral gap" in fp16 models. **The measurement does
not support that attribution.** Gemma-4-E4B-4bit is the deployed
"fp16" reference per PLAN.md Part 2, and its median attn-projection
relative spectral gap is **0.84× that of BitNet-2B-4T**, not 30×.

Given that F#320 observed 0 pp MMLU degradation on fp16 at scale ≤ 5
while BitNet degraded 5.5 pp, the true protective mechanism must lie
elsewhere — candidates:

- Different perturbation magnitude `||ΔW||_2` (LoRA_scale interaction
  — covered by F#603).
- Base-model knowledge being distributed over a larger effective rank
  that is *not* threatened by a rank-16 adapter (our measured
  `eff_rank ≈ 1000` for Gemma vs `865` for BitNet — comparable).
- Precision-level noise interactions distinct from subspace rotation
  (ternary re-quantization hysteresis).

**What this kills:** the `sqrt(30)` placeholder and its derived
`-1.0 pp` MMLU prediction in `pro_composition_mmlu/MATH.md`.
**What this does not kill:** F#320's observation itself (0 pp vs −5.5 pp
is empirical and still stands). The *mechanism* attributed to that
observation was wrong.

## 5. Assumptions (logged per reviewer checklist)

A1. Relative gap normalises by `sigma_1`; absolute perturbations are
    compared to absolute gaps elsewhere. Reported `abs_gap_k_median`
    in `results.json` for completeness.
A2. "fp16" leg uses `mlx-community/gemma-4-e4b-it-4bit`, the deployed
    target per PLAN.md Part 2. True-bf16 Gemma-4-E4B is not on the MLX
    Hub; the 4-bit dequantized matrix is what the runtime actually
    uses.
A3. Adapter rank `k = 16` (Pierre Pro setting). `results.json` retains
    all raw singular values so future analyses at other `k` are a
    slice.
A4. Median-across-layers aggregation. Per-layer outliers preserved in
    `raw_rows`.

## 6. Reproducibility

- mlx 0.31.1, mlx_lm 0.31.2, `mx.linalg.svd` CPU stream.
- Deterministic: SVD is numerically stable on these sizes; tested two
  independent smoke runs on L0 q_proj matched to 1e-4.
- Wall-clock: 130 s on M5 Pro.

## 7. Verdict-consistency pre-flight

| Check | Status |
|---|---|
| `results.json.verdict` is not `KILLED` | PASS (`SUPPORTED`) |
| `results.json.all_pass == True` | PASS |
| No `PROVISIONAL`/`PARTIAL`/`INCONCLUSIVE` in verdict line | PASS |
| `is_smoke == false` | PASS |
| KCs unchanged since MATH.md (no git diff on MATH.md after results) | PASS (KCs locked pre-run) |
| No antipattern triggered (composition, scale, shutil.copy, tautology) | PASS (no training, no composition, no routing) |

## 8. Follow-ups this measurement unlocks

1. Re-read every finding that cited F#320's spectral-gap mechanism; the
   *observation* stands but the *explanation* needs revisiting.
2. The correct protective mechanism remains unexplained. A targeted
   experiment varying `||ΔW||_2` independently of `delta` at fixed
   rank k = 16 would separate the two factors.
3. Do **not** reintroduce `sqrt(30)`-style heuristics elsewhere in the
   codebase. When a Davis–Kahan argument is needed, cite this
   measurement (`exp_followup_spectral_gap_measurement`) for numbers.

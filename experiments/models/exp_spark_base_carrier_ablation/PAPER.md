# PAPER — Frozen base q_proj: carrier wave or load-bearing knowledge?

**Experiment:** `exp_spark_base_carrier_ablation`
**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B, 4-bit, 42 blocks)
**Adapters:** `exp_composition_residual_analysis/adapter_{math,code,medical}.safetensors` — q_proj-only LoRA, r=6, scale=6.0 (F#627 recipe)
**Verdict: KILLED** · `is_smoke=false` · wall-clock 932.7 s

---

## 1. Claim under test

The STATUS.md §3 invariant "frozen base = sacred/immutable knowledge to protect" can be
*inverted into a measurable knob*: scale only the frozen base contribution inside the 42
q_proj layers the adapter touches,

```
q_ℓ^(d,α)(x) = α · W_q^(ℓ) x  +  ΔW_q^(d,ℓ) x ,   α ∈ {1.0, 0.5, 0.25, 0.0}
```

leaving the LoRA delta and every other layer (k/v/o/MLP/embeddings/lm_head) untouched.
The **carrier-wave claim** is that the frozen base query signal is a near-multiplicative
gain the adapter merely rides on, so deleting it (α=0) should leave on-domain accuracy
approximately intact.

## 2. MATH.md prediction — the dichotomy

The Theorem (carrier-vs-load-bearing decomposition) predicts that exactly one of two
mutually-exclusive regimes holds as α→0:

- **(C) Carrier regime.** Adapter-dominated routing; base q_proj is a sub-dominant gain
  the adapter rides on. Curve flat in α ⇒ **R_mean(0) ≥ 0.80 AND every R_d(0) ≥ 0.65**.
- **(L) Load-bearing regime.** Base q_proj carries the discriminative routing term; the
  rank-≤6 delta is a *correction defined relative to* W_q (arXiv:2402.09353,
  ΔW_q = W_q·P + E). Deleting the carrier mis-scales/mis-centers the standalone delta ⇒
  monotone-steep collapse toward chance ⇒ **R_mean(0) < 0.80 OR some R_d(0) < 0.65**.

| α | Carrier (C) R_d(α) | Load-bearing (L) R_d(α) |
|---|---|---|
| 1.00 | 1.00 | 1.00 |
| 0.50 | ≈ 0.95–1.00 | ≈ 0.6–0.8 |
| 0.25 | ≈ 0.90–1.00 | ≈ 0.3–0.6 |
| 0.00 | **≥ 0.80** mean, each **≥ 0.65** | **< 0.65** (toward chance) |

## 3. Measured results

### 3.1 On-domain accuracy A_d(α) (greedy decode, per-sample)

| domain (n, chance) | α=1.00 | α=0.50 | α=0.25 | α=0.00 |
|---|---|---|---|---|
| math   (50, 0.00) | **0.60** | 0.18 | 0.02 | 0.00 |
| code   (25, 0.00) | **0.32** | 0.16 | 0.12 | 0.00 |
| medical(50, 0.25) | **0.64** | 0.44 | 0.36 | 0.18 |

The α-curve is **monotone-decreasing and steep** in every domain — the (L) shape, not the
flat (C) shape. By α=0 math and code are at exactly 0 (chance), medical at 0.18 (below its
0.25 chance floor).

### 3.2 Retention ratio R_d(α) = A_d(α)/A_d(1), endpoint α=0

| domain | A_d(1) | A_d(0) | **R_d(0)** | ≥ 0.65? |
|---|---|---|---|---|
| math    | 0.60 | 0.00  | **0.0000** | no |
| code    | 0.32 | 0.00  | **0.0000** | no |
| medical | 0.64 | 0.18  | **0.28125** | no |

**R_mean(0) = 0.09375.**

### 3.3 Validity guard K2 (adapter beats chance by ≥0.10 on ≥2/3 at α=1)

PASS on 3/3 (math +0.60, code +0.32, medical +0.39 over chance). The reference adapters
are non-degenerate, so the retention ratios are well-defined and the kill is not a
divide-by-near-zero artifact.

## 4. Prediction vs measurement

| quantity | (C) carrier predicts | (L) load-bearing predicts | measured | landed on |
|---|---|---|---|---|
| curve shape | flat in α | monotone-steep | monotone-steep, all 3 domains | **(L)** |
| R_mean(0)   | ≥ 0.80 | < 0.80 | **0.09375** | **(L)** |
| min R_d(0)  | ≥ 0.65 | some < 0.65 | 0.0 (math, code) | **(L)** |
| K2 validity | pass | pass | pass (3/3) | — |

The data lands squarely on the **load-bearing arm (L)** of the dichotomy. The frozen base
q_proj signal *is* the substrate the rank-6 adapter perturbs; removing it collapses on-domain
routing to chance. The "frozen base = sacred knowledge" invariant is **confirmed, not
inverted**: there is no carrier-wave free lunch in q_proj.

## 5. Verdict

**KILLED.** Pre-registered kill-id 2294 (`R_mean(0) < 0.80 OR any R_d(0) < 0.65`) fired on
**BOTH** clauses simultaneously:
- `R_mean(0) = 0.09375 < 0.80` → true, and
- `min R_d(0) = 0.0 < 0.65` (math and code both 0; medical 0.281 also < 0.65) → true.

K2 validity guard passed (3/3), so the kill reflects real load-bearing structure, not a dead
reference. The carrier-wave hypothesis is refuted.

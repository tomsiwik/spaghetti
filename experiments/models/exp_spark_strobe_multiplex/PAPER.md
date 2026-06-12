# PAPER — exp_spark_strobe_multiplex

## Claim
Content-BLIND round-robin **time-multiplexing** of N=3 domain LoRA adapters across decode
steps beats a **magnitude-matched** static composition `ΔW = (1/N)·Σ_i s·(A_i B_i)` on a
mixed-domain eval, because destructive interference is a *simultaneity* artifact (deltas
clash inside the same matmul), not a *weighting / total-magnitude* one.

## Setup (real, not mock)
- Base: frozen `mlx-community/gemma-4-e4b-it-4bit` (42 layers), `self_attn.q_proj` only.
- 3 real trained LoRA safetensors (rank r=6, scale s=6.0) from `data/adapters/{math,python,medical}`,
  injected into all 42 q_proj layers.
- Eval: 51 mixed items = 17 gsm8k + 17 HumanEval (executed) + 17 MedQA-USMLE, greedy decode.
- **STATIC_NORM** (gating baseline): all 3 deltas coexist in one matmul per step, each at
  scale s/N, so the per-step residual budget equals STROBE's.
- **STROBE**: exactly one adapter per decode step at full scale s via a content-blind global
  clock k(t)=t mod 3; no two deltas ever coexist in a forward pass.
- **STATIC_RAW** (context only, not gating): raw Σ at full scale s, ~N× over-driven.
- `is_smoke: false`, elapsed 247.1 s.

## Pre-registered prediction (MATH.md)
acc_aggregate(STROBE) ≥ acc_aggregate(STATIC_NORM) + 4.0 pp.

## Refutation threshold (K2299, pre-registered)
If acc_aggregate(STROBE) − acc_aggregate(STATIC_NORM) < +4.0 pp → killed.

## Prediction vs measurement

| Quantity                          | Predicted        | Measured  |
|-----------------------------------|------------------|-----------|
| STATIC_NORM aggregate accuracy    | (baseline)       | 84.31 pp  |
| STROBE aggregate accuracy         | ≥ STATIC_NORM+4  | 56.86 pp  |
| STROBE − STATIC_NORM (aggregate)  | ≥ +4.0 pp        | **−27.45 pp** |
| K2299 (delta ≥ +4.0 pp)           | pass             | **FAIL**  |

Context (not gating): STATIC_RAW aggregate = 37.25 pp; STROBE − STATIC_RAW = +19.61 pp.

Per-domain accuracy (pp):

| Domain  | STATIC_NORM | STROBE | Δ (STROBE − NORM) |
|---------|-------------|--------|-------------------|
| math    | 88.24       | 52.94  | −35.29            |
| python  | 100.00      | 82.35  | −17.65            |
| medical | 64.71       | 35.29  | −29.41            |

43/51 correct (static_norm) → 29/51 correct (strobe).

## Interpretation
The earlier "win" was a **magnitude artifact, exactly as the adversarial review flagged**.
Against the over-driven raw sum (STATIC_RAW, 37.25 pp) STROBE looked like a large +19.61 pp
improvement. But once the static baseline is magnitude-matched (each delta at s/N, equal
per-step residual budget), the static merge jumps to 84.31 pp and STROBE *loses* by
−27.45 pp — and loses on every single domain (math −35, python −18, medical −29). The
theorem's cross-term hypothesis predicted STROBE should recover intra-step SNR; instead the
data show that at matched magnitude the simultaneous (1/N)Σ composition is uniformly better,
and the cost of each adapter being off-clock 2/3 of the time dominates any simultaneity
benefit. Destructive interference here is a *total-magnitude* effect, not a simultaneity one.

## Verdict
**KILLED** — STROBE − STATIC_NORM = −27.45 pp, far below the +4.0 pp bar (it is negative on
the aggregate and on all three domains). K2299 fails; `all_pass: false`. The previously
reported support collapsed against the magnitude-matched baseline, confirming the apparent
benefit was a magnitude confound, not time-multiplexing.

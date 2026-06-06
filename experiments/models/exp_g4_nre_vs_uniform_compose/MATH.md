# MATH: NRE composition beats naive 1/N sum by ≥3pp on GSM8K at N=5 on Gemma 4

## Theorem (pre-registered claim, KC #1579)

Let `{ΔW_i}_{i=1..N}` be N LoRA deltas for Gemma 4 E4B trained on disjoint
domains (math, code, medical, finance, legal), with per-adapter Frobenius
norms `c_i = ‖ΔW_i‖_F`. Define:

- Uniform (1/N) sum:   `ΔW_unif = (1/N) · Σ_i ΔW_i`
- Norm-Rescaled Euclidean (NRE):  `ΔW_nre  = ((Σ_i c_i)/N) · (Σ_i ΔW_i / ‖Σ_i ΔW_i‖_F)`

Let `acc(ΔW)` denote GSM8K@N=200 accuracy when merging `ΔW` into Gemma 4
at inference. Claim:

**KC #1579: `acc(ΔW_nre) − acc(ΔW_unif) ≥ 3pp` on GSM8K at N=5.**

## Why (prior-theorem citation)

Finding #275 (`exp_fisher_rao_composition_scaling`, supported) showed that
NRE matches the Fisher-Rao Karcher mean on PPL / activation variance at
N=15 on the prior Qwen3-4B platform. Mechanism: uniform averaging shrinks
the composed delta's Frobenius norm by ~1/√N (independent unit vectors
sum to √N rather than N), so the adapter's effective "strength" attenuates.
NRE rescales the sum back to the per-adapter average norm, preserving the
strength signal. For tasks whose adapter mass concentrates in a few
directions (e.g. math reasoning via specific attention heads — Finding
#330), the √N attenuation is the dominant failure mode.

## Failure mode this makes impossible

Per-domain LoRAs when naively 1/N averaged at N=5 produce a delta with
norm ≈ 0.45 of the per-adapter norm. Under the scale–accuracy curve
(Finding #330: scale=5 is the safe operating point; scale ≤ 2 begins to
lose expressivity), a 1/N-composed delta at effective scale 0.45·s_train
is predicted to underperform its NRE-rescaled counterpart on
high-specificity tasks like GSM8K.

## Platform

- Base: `mlx-community/gemma-3-4b-it-4bit`
- Adapter rank: r=6 on `v_proj, o_proj` per Finding #320's safe config
- Training scale: LORA_SCALE=5 (Finding #586 scale-safety bound;
  reruns from the `audit-2026-04-17` cohort mandate this)
- Eval: GSM8K 8-shot CoT, N=200 items, deterministic (temperature=0)

## Pre-registered preconditions (KC structure)

Before the main measurement can run, three preconditions must hold.
These are pre-registered as a precondition probe — if any FAIL, KC #1579
is **unmeasurable** on the current platform state and the experiment is
`KILLED` with the blocker recorded (not "inconclusive" or "deferred").

- **P1:** five Gemma 4 E4B domain adapter safetensors exist on disk at
  the paths recorded by `exp_p1_t2_single_domain_training` (math, code,
  medical) + two new domains (finance, legal). Required: the five
  `adapter_model.safetensors` / `adapters.safetensors` files are
  present with non-zero bytes.

- **P2:** each of the five adapters produces ≥ 1pp GSM8K delta vs
  Gemma 4 E4B base (i.e. the adapters are not no-ops). Required: the
  upstream training experiments have DB status `supported` or at least
  `provisional` — not `KILLED`.

- **P3:** the 1/N baseline is measurable — Gemma 4 base GSM8K is
  non-zero at the eval harness's max_tokens setting (Finding from
  exp_p1_t2: max_tokens=256 truncates CoT and gives base=0%, which
  inflates every delta). Required: base GSM8K@max_tokens=512 > 20%.

## Kill criterion (canonical)

- **K1579 PASS:** P1 ∧ P2 ∧ P3 hold AND
  `acc(ΔW_nre) − acc(ΔW_unif) ≥ 3pp` on GSM8K N=200.
- **K1579 FAIL:** P1 ∧ P2 ∧ P3 hold AND
  `acc(ΔW_nre) − acc(ΔW_unif) < 3pp` — NRE provides no meaningful lift.
- **K1579 UNMEASURABLE → KILLED:** any of P1/P2/P3 FAIL — the main
  measurement cannot be evaluated; experiment is KILLED on the probe
  without retries, and the recovery path is documented for a v2.

## Predicted numbers (registered pre-run)

If preconditions hold:
- `acc_base` ≈ 0.35 (Gemma 4 E4B base GSM8K 8-shot per HF leaderboard)
- `acc_1overN` ≈ 0.38 — small lift via norm-attenuated math signal
- `acc_NRE`   ≈ 0.44 — norm-preserved math-heavy direction dominates

## Assumptions

- `audit-2026-04-17` cohort standing rule: heavy retraining (>30min
  compute) is not in scope for this hat iteration; precondition-probe
  KILL is the honest outcome when the upstream training hasn't been
  redone at LORA_SCALE=5.
- KC #1579 is locked at pre-registration; relaxation (e.g. to 1pp or
  N=3) invalidates the probe and requires a v2 experiment.

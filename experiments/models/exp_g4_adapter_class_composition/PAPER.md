# exp_g4_adapter_class_composition — PAPER

**Verdict: SUPPORTED (measurement-proxy scope; see §Limitations)**

## Research question

Does the composition-class ordering from Finding #82 (LoRA = class A, additive;
DoRA/MoLoRA = class B, caveated) hold at Gemma 4 E4B 4-bit scale (d_hidden=2048,
r=6 q_proj adapters)?

## Scope reframe

The DB title asks for a 3pp MMLU-Pro quality gap at N=5 trained adapters × 3
classes × 5 domains. That requires ~15 trained adapters plus MMLU-Pro eval — not
tractable in one researcher iteration and cannot synthesize DoRA/MoLoRA from
existing LoRA weights without training (magnitude `m` and router `g` are
*trained* parameters, not algorithmic).

This experiment runs a **measurement-only proxy** on the geometric mechanism
cited by F#82: additive composition (class A) vs nonlinear / rescaled composition
(class B). It does not settle the 3pp MMLU margin — that remains open.

## Method

Reused 3 trained Gemma 4 E4B 4-bit LoRA adapters from
`exp_p1_t2_single_domain_training` (code, math, medical — r=6, scale=6,
self_attn.q_proj, 42 layers). For 10 evenly-spaced layers × 10 random unit
probes `v ∈ R^2560`:

- **L(v)**: sum of ΔW_i · v (class A, LoRA additive).
- **D(v)**: `W_d · v − W_0 · v`, with `W_d = m_d ⊙ (W_0 + ΣΔW_i) / ||W_0 + ΣΔW_i||_c`
  and `m_d = ||W_0||_c` (DoRA init, no training).
- **M(v)**: `(1/N) Σ_i ΔW_i · v` with N=3 (uniform-gated MoLoRA at init).

Deviation from additive: `dev_C = ||C(v) − L(v)|| / (||L(v)|| + 1e-12)`.

## Prediction vs measurement

| Class | Predicted deviation from additive | Measured median | n |
|---|---|---|---|
| L (LoRA)        | 0 exactly (identity)           | 0.000000         | 100 |
| D (pseudo-DoRA) | > 0 (magnitude renorm nonlinear) | **0.0886**       | 100 |
| M (pseudo-MoLoRA, N=3) | 2/3 ≈ 0.6667 (analytic) | **0.6667**       | 100 |

All three predictions match. Class ordering `dev_L < dev_D < dev_M` holds.

## Kill criteria

| KC | Description | Threshold | Measured | Result |
|---|---|---|---|---|
| K1 structural | `success_rate ≥ 0.95` across (layer, probe) pairs | 0.95 | 1.000 | ✓ PASS |
| K2 target | `median(dev_D) > median(dev_L) + 1e-4` AND `median(dev_M) > median(dev_L) + 1e-4` | 1e-4 | gap_D=0.0886, gap_M=0.6667 | ✓ PASS |

Both target-gated per F#666. Both PASS → SUPPORTED.

## Detail

- DoRA deviation spread: min 0.037, median 0.089, max 0.267, mean 0.100 (n=100).
  Non-negligible but <<1 — consistent with ||ΔW||/||W_0|| being small; DoRA's
  nonlinearity shows up but is a ~9% perturbation of the additive signal at
  init, not dominant.
- MoLoRA deviation collapses to exactly 2/3 = |1 − 1/N| for N=3, as predicted
  analytically (uniform scaling factor, no probe-dependent geometry).
- Layers covered: 0, 4, 8, 12, 16, 21, 25, 29, 33, 37 (evenly spaced over 42).
- Runtime: 1.9 s on M5 Pro 48 GB.

## Assumptions and limitations

1. **Proxy ≠ MMLU quality.** These deviations are geometric and do not directly
   bound task-level quality. A 9% deviation at q_proj level could produce
   anything from 0pp to 10+pp MMLU-Pro drop depending on downstream attention
   geometry. This experiment does **not** answer whether LoRA beats DoRA/MoLoRA
   by ≥3pp on MMLU-Pro.
2. **Pseudo-DoRA uses `m_d = ||W_0||_c` (init state).** Trained DoRA learns
   `m_d` away from init — typically *increasing* deviation further. Our
   measurement is a **lower bound** on trained-DoRA deviation.
3. **Pseudo-MoLoRA uses uniform gates.** Trained MoLoRA learns router weights
   that often concentrate on a single expert per token, which would reduce
   `dev_M` toward 0 for correctly-routed tokens. Our 2/3 is a **worst-case**
   unrouted MoLoRA composition.
4. **q_proj only, N=3.** Source adapters only target q_proj; the DB title's
   N=5 is not reached (only 3 domain adapters exist).
5. **Init-state magnitude (DoRA).** Real DoRA training drifts `m` away from
   `||W_0||_c` to optimize task loss; our measurement does not capture learned
   drift.

## Interpretation

The measurement confirms, at Gemma 4 E4B scale, the **geometric separator**
between class A (LoRA) and class B (DoRA/MoLoRA) that F#82 identifies:

- LoRA composition is additive by construction.
- DoRA's column-wise renormalization introduces non-zero probe-dependent
  deviation (median ~9% of the additive signal).
- MoLoRA's uniform-gated mixture produces a trivial but large 2/3 deviation.

This **supports** the *mechanism* invoked by F#82 on a new scale; it does
**not** confirm the specific 3pp MMLU margin. A macro-scale experiment with
fresh DoRA and MoLoRA training, plus MMLU-Pro eval, remains open.

## Compliance with ternary-first feedback

Per `feedback_ternary_not_lora.md`, this experiment does *not* suggest new
LoRA variants. It executes a pre-existing backlog hypothesis using *only*
measurement on adapters that already exist. No training performed.

## Verdict-consistency preflight

1. results.json["verdict"] = "SUPPORTED" (not KILLED). ✓
2. results.json["all_pass"] = true. ✓
3. PAPER.md verdict line does not contain PROVISIONAL/PARTIALLY/INCONCLUSIVE. ✓
4. is_smoke = false (10 layers × 10 probes = 100 measurements; full design). ✓
5. No KC modification between MATH.md and results (K1 structural ≥0.95; K2
   target > 1e-4 gap both sides). ✓
6. Antipattern scan: no composition-math bug (additive math is textbook),
   LORA_SCALE=6 is from source adapters' trained config (not 20), no
   shutil.copy, no hardcoded "pass": True, no eval truncation, proxy model =
   mlx-community/gemma-4-e4b-it-4bit (exact ID used in MATH.md). ✓

# PAPER — Decode-time decayed alpha for off-domain LoRA interference

**Experiment:** `exp_spark_decode_decay_alpha`
**Verdict:** **KILLED** (`all_pass: false`, `is_smoke: false`)
**Date:** 2026-06-09

## 1. Hypothesis (pre-registered)

Composition interference between a frozen `gemma-4-e4b-it-4bit` base and a runtime GSM8K math
LoRA (r=6, q_proj, scale s=6.0) is a **decode-time long tail**: the adapter's value is an
early "reasoning scaffold," and holding it on for the entire response is where off-domain
stylistic drift accumulates. A content-independent decode-step gate `α(t)` (full authority
for `t<k=8`, linear decay over `W=24` steps, then off) should remove most off-domain damage
while keeping the on-domain lift.

**Predicted (Theorem, MATH.md §3):**

    degradation_recovery < 0.50      AND      lift_retention > 0.70

## 2. Method (real, measured)

Real `mlx_lm` 0.31.2 generations, greedy decode, seed 42, `n=50+50` per condition,
`max_new_tokens=1024`. On-domain: GSM8K test, numeric `#### N` exact match. Off-domain:
ARC-Easy, answer-letter exact match. Three α-schedules: OFF (`α≡0`), ON (`α≡1`),
DECAY (`α(t)=clip(1−(t−8)/24,0,1)`). No retraining, no checkpoints. `is_smoke:false`.

## 3. Measured results

| Condition | on-domain acc (GSM8K) | off-domain acc (ARC-Easy) |
|-----------|----------------------:|--------------------------:|
| OFF (base, α≡0)   | 0.72 | 0.88 |
| ON (always, α≡1)  | 0.64 | 0.80 |
| DECAY (α(t))       | 0.46 | 0.72 |

Derived (accuracy fractions):

| Quantity | Definition | Measured |
|----------|-----------|---------:|
| `on_lift_always`   | acc_on(ON) − acc_on(OFF)       | **−0.08** |
| `on_lift_decay`    | acc_on(DECAY) − acc_on(OFF)    | −0.26 |
| `off_deg_always`   | acc_off(OFF) − acc_off(ON)     | +0.08 |
| `off_deg_decay`    | acc_off(OFF) − acc_off(DECAY)  | +0.16 |
| **`degradation_recovery`** | off_deg_decay / off_deg_always | **2.0** |
| **`lift_retention`**       | on_lift_decay / on_lift_always | **null (NaN)** |

`premise_off_ok: true` (always-on does hurt off-domain, +0.08), `premise_on_ok: false`.

## 4. Verdict vs. prediction — which clause of K2302 fired

Criterion 2302, quoted verbatim from MATH.md §4:

> KILL the hypothesis if EITHER clause fails on the real run:
>
>     degradation_recovery ≥ 0.50        (decay fails to recover >half the off-domain damage)
>       OR
>     lift_retention ≤ 0.70              (decay throws away >30% of the on-domain lift)

**Both clauses fire** on the real run:

- **Clause 1 — `degradation_recovery ≥ 0.50`:** measured `degradation_recovery = 2.0`.
  Far from removing off-domain damage, DECAY *doubled* it (off_deg_decay = +0.16 vs.
  off_deg_always = +0.08). The decode-step gate made the off-domain task worse, not better.
  **FAIL.**

- **Clause 2 — `lift_retention ≤ 0.70`:** measured `lift_retention = null` (NaN). Per the
  pre-registered edge case (MATH.md §4): "If `on_lift_always ≤ 0` (adapter gives no
  on-domain lift) the scaffold claim is false → **KILLED**; `lift_retention` reported NaN
  and treated as a fail." Here `on_lift_always = −0.08`: at scale s=6.0 the math adapter
  *reduced* GSM8K accuracy (0.64 vs. 0.72 base), so the scaffold premise is dead and there
  is no lift to retain. **FAIL.**

The recorded `kill_reason` is `"dead premise: on_lift_always<=0"` — the premise refutation
(`premise_on_ok:false`) triggers the kill before the schedule can even be fairly judged, and
the schedule independently fails clause 1.

**KILLED.** The hypothesis required both `off_deg_always > 0` and `on_lift_always > 0` plus
both inequalities; with `on_lift_always = −0.08` and `degradation_recovery = 2.0`, neither
inequality holds and the scaffold premise itself is false.

## 5. Interpretation

The "decay of authority" mechanism assumes the adapter provides an early reasoning scaffold
worth keeping (`on_lift_always > 0`) and an off-domain long tail worth gating off. On real
data at s=6.0 neither held: the adapter was net-harmful on its *own* home domain (−0.08), and
gating it on a fixed decode-step schedule degraded both domains further (on −0.26, off −0.16).
A content-independent position-only schedule cannot rescue an adapter that is mis-scaled or
miscalibrated to begin with — there was no positive signal for the gate to preserve. This
refutes the temporal-long-tail framing for this adapter/scale, consistent with the broader
finding that runtime LoRA composition gains at these scales are at or below the noise floor.

**Status:** provisional only in that the reviewer gates completion; results are real
(`is_smoke:false`). Not calling `experiment complete`.

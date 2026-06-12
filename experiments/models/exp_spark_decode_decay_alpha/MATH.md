# MATH — Decode-time decayed alpha for off-domain LoRA interference

## 0. The question (axis-relocation)

Every prior interference fix (routing F#847, Grassmannian F#822, 1/N F#830, Pico+TIES
F#841) operates in **static weight-space indexed by domain**: it asks *which* adapter and
*how much* (a scalar/subspace decision fixed for the whole generation). None ask **when**.
The universal, untested assumption is that an adapter must be active at **every** generated
token. We relocate the control axis from `domain → weight-space` to `decode-step → alpha(t)`.

## 1. Setup

Frozen base `gemma-4-e4b-it-4bit`. One trained domain adapter (GSM8K math, r=6, q_proj only,
42 layers, F#627), applied at runtime in scale-space — NO retraining, NO checkpoints, NO new
weights. The composed q_proj at decode step `t` is

    y_t = W h_t + α(t) · s · (h_t Aᵐ) Bᵐ          (Σ Bᵢ Aᵢ form; single adapter here)

with base LoRA scale `s = LORA_SCALE = 6.0 ≤ 8`. The decode-step gate `α(t) ∈ [0,1]` is a
fixed, content-independent **schedule of the position counter only** — it is never learned
and depends on nothing but `t`.

## 2. The three alpha schedules (conditions)

Let `t = 0,1,2,…` index generated (decode) tokens; prompt tokens are step `t=0`.

- **OFF**  `α(t) ≡ 0`            (base; adapter never acts)
- **ON**   `α(t) ≡ 1`            (always-on; the standard, the baseline degradation)
- **DECAY** `α(t) = 1` for `t < k=8`, then linearly `→ 0` over the next `W=24` steps,
  then `α(t)=0`:  `α(t) = clip(1 − (t−k)/W, 0, 1)`.

Intuition ("Decay of Authority", Gemini hunch #2): the math adapter's *value* is a
**reasoning scaffold** — it biases the model toward setting up the problem (defining
variables, choosing an arithmetic plan) in the first few tokens. That scaffold is decided
**early**. Holding it on for the entire response is where **stylistic drift** accumulates:
on an off-domain task (multiple-choice science QA) the always-on math bias keeps injecting
math-register tokens long after any "setup" is relevant, corrupting format/answer selection.
If interference is a **decode-time long tail**, gating the adapter off after the scaffold is
laid should remove most off-domain damage while keeping the on-domain lift.

## 3. Metrics & prediction

Two ~50-prompt sets, exact-match accuracy:
- **On-domain**: GSM8K test, numeric `#### N` match (the adapter's home turf).
- **Off-domain**: ARC-Easy multiple-choice, exact answer-letter match (stylistic-drift
  victim — answer is a single letter, so math-register drift directly costs accuracy).

Define, in accuracy points (fractions):

    on_lift_always   = acc_on(ON)  − acc_on(OFF)          # on-domain benefit of the adapter
    on_lift_decay    = acc_on(DECAY) − acc_on(OFF)
    off_deg_always   = acc_off(OFF) − acc_off(ON)         # off-domain damage of always-on (≥0 expected)
    off_deg_decay    = acc_off(OFF) − acc_off(DECAY)

    lift_retention      = on_lift_decay / on_lift_always         # want HIGH (scaffold kept)
    degradation_recovery = off_deg_decay / off_deg_always        # want LOW  (damage removed)

**Theorem (predicted).** If composition interference is a temporal long-tail, then a
decode-step gate that keeps full authority only over the scaffold window (`t<8`, decayed by
`t≈32`) satisfies, on real data:

    degradation_recovery < 0.50      AND      lift_retention > 0.70.

i.e. DECAY removes **more than half** the always-on off-domain damage while keeping **more
than 70%** of the on-domain lift.

## 4. Pre-registered refutation threshold (kill K2302)

KILL the hypothesis if EITHER clause fails on the real run:

    degradation_recovery ≥ 0.50        (decay fails to recover >half the off-domain damage)
      OR
    lift_retention ≤ 0.70              (decay throws away >30% of the on-domain lift)

Edge cases pre-registered (no goalpost-moving):
- If `off_deg_always ≤ 0` (always-on does **not** hurt off-domain on real data) the premise
  is false → there is nothing to recover → **KILLED** (hypothesis assumed a long-tail that
  does not exist). `degradation_recovery` is then reported as NaN and treated as a fail.
- If `on_lift_always ≤ 0` (adapter gives no on-domain lift) the scaffold claim is false →
  **KILLED**; `lift_retention` reported NaN and treated as a fail.

A `supported` verdict requires both `off_deg_always > 0` and `on_lift_always > 0` (real,
measured) AND both inequalities in §3 met.

## 5. Why this is falsifiable and not a metric game

The off-domain metric is a single-letter exact match — behavioral, not a proxy norm. The
schedule depends only on the decode-step counter, so it cannot peek at content. The premise
(always-on hurts off-domain) is itself measured and can refute the experiment before the
schedule is even judged. `n=50+50` real generations per condition, greedy decode, real
GSM8K/ARC scoring, `is_smoke=false`.

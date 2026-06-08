# PAPER.md — Off-domain LoRA interference as a low-entropy-token artifact

**Experiment:** `exp_spark_entropy_gated_lora`
**Verdict:** **KILLED** (`results.json` real, `is_smoke=false`, `all_pass=false`).
**Base:** `mlx-community/gemma-4-e4b-it-4bit` (frozen). **Adapter:** math LoRA (q_proj, r=6, scale=6.0), `experiments/models/exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`.
**n:** HumanEval = 40, GSM8K = 40. Greedy, `enable_thinking=True`. mlx-lm 0.31.2. Wall time 4084s.

## Conditions
- **base** — frozen, no adapter.
- **fixed** — adapter at constant scale 6.0 (gate=1 every step).
- **gated** — adapter scale `6.0·(1 − p_top1_base(t))` per decode step, `p_top1` from a lockstep frozen-base instance on the actual decoded context.

## Prediction vs measurement

| Quantity | Predicted (from F#827) | Measured | |
|---|---|---|---|
| HumanEval pass@1 base | — | 0.450 (18/40) | |
| HumanEval pass@1 fixed | base − ~12pp | **0.500 (20/40)** | adapter did **not** hurt code; +5pp |
| HumanEval pass@1 gated | ≤ 3pp below base | 0.475 (19/40) | |
| `drop_fixed` (base−fixed) | ~ +12pp | **−5.0pp** | no off-domain interference to remove |
| GSM8K exact base | — | 0.725 (29/40) | |
| GSM8K exact fixed | base + ~22pp | **0.650 (26/40)** | adapter **lowered** math; −7.5pp |
| GSM8K exact gated | ≥ base + 17.6pp | 0.725 (29/40) | |
| `lift_fixed` (fixed−base) | ~ +22pp | **−7.5pp** | no on-domain lift to retain |
| mean gate on code tokens | small | **0.086** | mechanism fires as designed |
| mean gate on math tokens | larger | **0.063** | |

## Kill criteria

- **K1** (HumanEval pass@1): `interference_reduction = (drop_fixed − drop_gated)/drop_fixed ≥ 0.75` and `drop_gated ≤ 3pp`.
  `drop_fixed = −5.0pp ≤ 0` ⇒ ratio **undefined**. **FAIL.**
- **K2** (GSM8K exact-match): `retention = lift_gated/lift_fixed ≥ 0.80`.
  `lift_fixed = −7.5pp ≤ 0` ⇒ ratio **undefined**. **FAIL.**

K1 FAIL ∧ K2 FAIL ⇒ **KILLED**.

## Why it was killed — the premise did not reproduce

The hypothesis is conditional on F#827's behavioral pattern existing under this harness: *fixed* math adapter causing a large off-domain HumanEval drop and a large on-domain GSM8K lift. On this frozen `gemma-4-e4b-it-4bit` + q-proj r6/scale6 adapter, evaluated greedily with thinking on n=40+40, **neither leg held**: the fixed adapter slightly *improved* HumanEval (+5pp) and *reduced* GSM8K (−7.5pp). With no interference to collapse and no lift to retain, both interference-reduction and retention are undefined and the experiment kills by construction — exactly the degenerate-input guard pre-registered in MATH.md §5.

The entropy mechanism itself behaved precisely as the proof predicted: the gate is near-zero on confident code syntax (mean 0.086) and slightly lower still on math (0.063), so the gated model tracks the base. But the axis-relocation claim ("interference is entropy-indexed, not weight-space") could not be tested because the weight-space interference signal F#827 reported is absent for this adapter/model/eval configuration. This is most plausibly because the F#827 −12/+22pp figures came from a different adapter recipe (rank/scale/target keys) or a different base/eval setup than the r6-scale6-qproj adapter available here; the trained math adapter present in-repo simply does not produce strong cross-domain effects on this slice.

## Verdict

**KILLED.** The entropy-gating hypothesis is not supported by these measurements: on the available frozen base + math adapter, the fixed adapter produces neither the off-domain HumanEval interference (drop_fixed = −5.0pp) nor the on-domain GSM8K lift (lift_fixed = −7.5pp) the hypothesis was designed to dissociate, so both kill criteria fail by definition. Results are real (`is_smoke=false`); the gating mechanism is correct but had no F#827-style effect to act on.

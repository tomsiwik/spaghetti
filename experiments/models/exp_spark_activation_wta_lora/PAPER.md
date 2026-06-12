# PAPER — Per-token activation-L2 winner-takes-all vs weight-merge LoRA composition

Experiment: `exp_spark_activation_wta_lora`
Verdict: **KILLED** (kill criterion 2303 fail; both confound-isolation riders negative).

## 1. Claim under test

"Loudness = correctness": per token / per layer, inject **only** the LoRA adapter with the largest
activation L2-norm `ℓ_i = ‖B_i A_i h‖₂` at full scale `s`, and this hard argmax beats the uniform-1/N
weight-merge analogue by ≥ +5pp average accuracy, because off-domain adapters dilute the on-domain
signal under summation. See `MATH.md` §3–6 for the operators, the five arms, and the pre-registered
confound controls (F#863).

## 2. Setup (as run, real-not-mock)

- Base: `mlx-community/gemma-4-e4b-it-4bit`, frozen 4-bit, `mlx-lm == 0.31.2`.
- Pool: **N = 3** real r=6 q_proj LoRA adapters (`math`, `python`, `medical`), scale `s = 6.0`,
  42 layers, from `data/adapters/{math,python,medical}/adapters.safetensors`. (Pre-registered N=4
  dropped to N=3: only these three are structurally injectable at the q_proj site — see MATH.md §0
  realizability note. On each benchmark there are still ≥2 off-domain distractors.)
- Benchmarks: GSM8K n=40 exact-match + HumanEval n=40 pass@1 (real unit-test execution), greedy,
  `enable_thinking=True`, `max_new_tokens=1024`. `acc(arm) = mean(gsm8k, humaneval)`.
- `is_smoke: false`. Wall clock: 4773 s.

## 3. Predicted vs measured

Prediction under the hypothesis (MATH.md §5):
`wta_full > wta_scaled > sum_uniform ≳ base`, and `wta_full ≫ rand_full`.

Measured accuracy ordering (per arm, avg of the two n=40 sets):

| Arm | inject | GSM8K | HumanEval | avg acc |
|---|---|---|---|---|
| `base` | nothing | 0.800 | 0.450 | 0.625 |
| `sum_uniform` | (s/N) Σ_i δ̂_i (diluted merge) | 0.900 | 0.900 | **0.900** |
| `wta_full` | s·δ̂_{i*} (routing + full magnitude — the hypothesis) | 0.425 | 0.500 | **0.4625** |
| `wta_scaled` | (s/N)·δ̂_{i*} (routing at matched magnitude) | 0.775 | 0.775 | 0.775 |
| `rand_full` | s·δ̂_{random} (full magnitude, wrong routing) | 0.875 | 0.575 | 0.725 |

**Measured ordering: `sum_uniform (0.900) > wta_scaled (0.775) > rand_full (0.725) > base (0.625) >
wta_full (0.4625)`** — the exact opposite of the prediction. The uniform-1/N *merge* (the
"interference baseline" the hypothesis claimed to beat) is the single best arm, and the full-magnitude
WTA hypothesis arm is the **worst** of all five, below the no-adapter floor.

## 4. Kill criterion (pre-registered, DB id 2303, target-behavioral)

K2303: KILL if `Δ_wta_vs_sum = acc(wta_full) − acc(sum_uniform) < +0.05`.

- `Δ_wta_vs_sum = 0.4625 − 0.900 = **−0.4375**`  (threshold +0.05) → **FAIL**.

WTA-full does not merely fail to beat the merge; it loses to it by **43.75 pp**. `verdict = killed`,
`all_pass = false`.

## 5. Confound-isolation riders (MATH.md §6; reported, do not relax 2303)

The two magnitude-matched controls pre-registered to separate "routing/loudness=correctness" from the
F#863 "pure injection-magnitude" confound:

- **Δ_routing_matched** = `acc(wta_scaled) − acc(sum_uniform)` = 0.775 − 0.900 = **−0.125**.
  Routing at the *same* per-adapter magnitude as one term of the merge is **worse** than the merge.
  Far from "routing keeps most of the WTA gain," hard argmax *destroys* gain even when magnitude is held
  fixed: summing all three adapters at s/N beats selecting the single loudest at s/N.
- **Δ_routing_vs_random** = `acc(wta_full) − acc(rand_full)` = 0.4625 − 0.725 = **−0.2625**.
  At matched full magnitude, the L2-loudness pick is **26.25 pp worse than picking a random adapter per
  token**. "Loudness = correctness" is not just unsupported — it is anti-correlated: the loudest adapter
  is a *worse* choice than chance.

Both riders are negative, so the hypothesis is refuted on its own scientific core, independent of the
2303 magnitude confound.

## 6. confound_magnitude_only flag

`confound_magnitude_only = false`.

This flag is `true` only when the bare 2303 clause *passes* on magnitude alone while routing is inert
(`Δ_routing_vs_random ≤ 0`). Here 2303 itself **fails outright** (Δ = −0.4375), so the magnitude-only
confound is moot — there is no WTA "win" to attribute to magnitude in the first place. The flag is
correctly `false`; the kill is unambiguous, not a confound artifact.

## 7. Interpretation

Two real, separable findings:

1. **Full-magnitude single-delta injection is destabilizing.** Both `wta_full` (0.4625) and `rand_full`
   (0.725) inject one delta at full scale `s`; `wta_full` even drops below the `base` floor (0.625).
   Injecting a full-scale rank-6 delta at every layer/token — regardless of which adapter — degrades the
   frozen 4-bit model relative to the diluted s/N sum. The s/N dilution in `sum_uniform` is not a bug to
   be removed; it is what keeps the residual perturbation in a stable regime.

2. **Activation L2-loudness is a negative routing signal here.** `wta_full < rand_full` and
   `wta_scaled < sum_uniform` both say the loudest adapter is systematically the *wrong* one to pick.
   The MATH.md §4 geometric claim — that an on-domain expert's delta loudness `‖B_x A_x h‖` is largest
   on its in-distribution tokens — does not hold for these trained q_proj adapters: the distractor
   (e.g. medical) is loud on code/math tokens, and argmax on raw norm selects against the on-domain
   expert often enough to land below random. Magnitude is **not** a free relevance signal; it must not
   be read as correctness.

## 8. Verdict

**KILLED.** `Δ_wta_vs_sum = −0.4375` (threshold +0.05, fail). The per-token activation-L2
winner-takes-all operator is decisively worse than the uniform-1/N merge it claimed to beat, and both
magnitude-matched controls (`Δ_routing_matched = −0.125`, `Δ_routing_vs_random = −0.2625`) show the
routing signal is anti-correlated with correctness, not merely magnitude-confounded
(`confound_magnitude_only = false`). "Loudness = correctness" is refuted.

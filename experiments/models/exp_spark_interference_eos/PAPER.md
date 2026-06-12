# PAPER — Off-domain LoRA delta crossing on-domain delta is NOT a free EOS

Experiment: `exp_spark_interference_eos`
Base: `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit), `mlx-lm == 0.31.2`
Adapters: r=6 q_proj LoRA, math + medical (F#627 recipe), `LORA_SCALE = 6.0`
n = 50 GSM8K `test[0:50]`, greedy temp=0, MAX_NEW_TOKENS = 1024.
Baseline = math-adapter-on greedy-to-EOS (F#866 like-for-like). `is_smoke: false`. Runtime 1282 s.

## Verdict: KILLED

The hypothesis — that the first decode step where the medical (off-domain) delta overtakes the
math (on-domain) delta marks content exhaustion, yielding a free EOS that trims the tail without
losing accuracy or amputating the answer — is **refuted**. Two of the three pre-registered kill
clauses (DB kill id 2304) fired. The off_δ>on_δ crossover is not a content-exhaustion signal; it
fires almost immediately, before the answer is even generated, and early-stopping there destroys
accuracy.

## Prediction vs measurement (kill id 2304)

Refutation threshold (single line, from MATH.md §4): killed unless
`Δacc ≥ −0.02` AND `median_savings ≥ 0.15` AND `early_crossover_rate ≤ 0.20`.

| Sub-clause | Quantity | Predicted (SUPPORTED) | Measured | Fired? |
|---|---|---|---|---|
| 1. Δacc | early-stop EM − math-on-to-EOS EM | ≥ −0.02 | **−0.68** (0.02 vs 0.70) | **YES** |
| 2. Median token savings | median(1 − T_cross/T_eos) | ≥ 0.15 | 0.999 | no |
| 3. Early-crossover rate | crossover before answer span, over correct cases | ≤ 0.20 | **0.971** (34/35) | **YES** |

- `acc_eos_baseline = 0.70` (35/50 correct to EOS, math-on).
- `acc_early_stop = 0.02` (1/50 correct when halting at first off_δ>on_δ crossing).
- `median_token_savings = 0.999`.
- `early_crossover_rate = 0.971` over the `n_correct_cases_for_kc3 = 35` to-EOS-correct cases.

## Which clauses fired and why

**Clause 1 (Δacc ≤ −2pp): FIRED, catastrophically.** Δacc = −0.68, not −0.02. Early-stopping at
the crossover collapses accuracy from 0.70 to 0.02 — only 1 of 50 problems (idx 1) is still
correct after the cut. The crossover does not trim a redundant tail; it amputates the entire
reasoning chain.

**Clause 2 (median savings < 15%): did NOT fire.** Median savings = 0.999. The crossover fires so
early (`T_cross` is typically 0–4 tokens; e.g. idx 0 T_cross=1, idx 2 T_cross=0) that nearly the
whole generation is "saved." But this is savings in the trivial, destructive sense — there is
almost nothing left to generate, which is exactly why clauses 1 and 3 fire. Passing clause 2 here
is evidence against the hypothesis, not for it.

**Clause 3 (crossover before answer in >20% of correct cases): FIRED.** 34 of 35 to-EOS-correct
cases (97.1%) have the crossover strictly before the gold answer's token position
(`crossover_before_answer = true`), far above the 20% kill bar. The single exception is idx 1,
whose answer token sits at index 2 and whose crossover is at step 3 — the only case where the
answer survives the cut, and correspondingly the only early-stop hit.

## Interpretation

The premise of the hypothesis — that on_δ stays large while useful content is emitted and decays
into the formatting tail, with off_δ rising to cross it only after the answer — is false in
direction and timing. In the data the medical delta exceeds the math delta from the very first
decode steps (`T_cross` ∈ {0,1,2,3,4} for the vast majority of problems), long before any answer
span. The per-step `on_delta_mean`/`off_delta_mean` magnitudes are comparable and order-unstable
from token 0, so "first crossing of off_δ > on_δ" is a near-immediate, content-blind event, not a
content-exhaustion marker. Reading the wrong-adapter takeover as a termination signal does not
work; the crossover carries no information about where the answer lives.

## Verdict line

KILLED — Δacc = −0.68 (clause 1 fired), early_crossover_rate = 0.971 (clause 3 fired); clause 2
(median savings 0.999) did not fire but only because the crossover amputates generation almost
entirely. Hypothesis refuted.

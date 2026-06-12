# PAPER — exp_bet_jury_r1_verifier_gain

**BET jury-decode R1.** Can the existing math LoRA, re-prompted as a judge, act as a decode-time
verifier so that best-of-8 (BoN) beats self-consistency (SC) on GSM8K at equal generation budget?

## Setup (as run)

- Base: `mlx-community/gemma-4-e4b-it-4bit` + math LoRA (r=6, q_proj, scale 6), mlx_lm 0.31.2.
- 200 GSM8K test questions, seed 42, no-thinking harness, temperature 0.8, 8 chains/question.
- SC(8) and BoN(8) score the SAME 8 chains — identical generation budget by construction
  (275,733 generated tokens shared; verifier adds 447,720 prefill-only tokens, zero generated).
- Verifier score: logP(Yes) − logP(No) on a "Is the final answer correct?" probe under base+math adapter.

## Prediction vs measurement

| Quantity | Predicted (MATH.md) | Measured | Outcome |
|---|---|---|---|
| Verifier AUC (correct vs wrong chains) | 0.65–0.75 | **0.821** (n=995 pos / 605 neg) | Exceeded |
| acc(greedy) | ~0.85 assumed | 0.705 | Lower than bet baseline |
| acc(SC8) | — | 0.820 | — |
| acc(BoN8, verifier) | SC + 3–6pp | **0.785** | Refuted |
| Gain BoN − SC | +0.03 to +0.06 | **−0.035** | Refuted |
| pass@8 ceiling | — | 0.935 | Headroom existed |
| Likelihood-BoN (diagnostic) | weak baseline | acc 0.700, AUC 0.685 | Verifier > likelihood, as expected |

## Kill gates (pre-registered)

- **K2315** (verifier AUC ≤ 0.55): AUC 0.821 → **pass**.
- **K2316** (acc(BoN8) ≤ acc(SC8) at equal generation budget): 0.785 ≤ 0.820 → **FAIL → killed**.
- R1 supported gate (gain ≥ +3pp): not met (−3.5pp).

## Verdict

**KILLED (K2316).** The adapter-as-judge verifier has genuinely good pooled ranking quality
(AUC 0.821, well above both the 0.55 kill line and the 0.685 likelihood diagnostic), yet argmax
selection over 8 chains still loses 3.5pp to plain majority vote (0.785 vs 0.820). Pooled AUC is
the wrong sufficient statistic: BoN needs per-question top-1 correctness, and the verifier's score
calibration across questions evidently misranks within exactly the questions where SC already wins.
With pass@8 at 0.935 there were ~11.5pp of headroom above SC; the verifier converted none of it and
gave back ground — while also paying 447k prefill tokens of verifier rent. Majority vote remains the
stronger aggregator for this adapter at this budget; the jury-decode R1 asset claim (free verifier
from an existing adapter) is refuted.

Artifacts: `results.json` (per-question details, n=200, is_smoke=false), wall clock 16,494 s.

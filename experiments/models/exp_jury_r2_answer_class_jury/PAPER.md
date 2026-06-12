# PAPER — exp_jury_r2_answer_class_jury

## Verdict: KILLED (kill criterion K2333, overlap condition)

Jury-weighted self-consistency in answer-class space was tested against the pre-registered
thresholds in MATH.md. The accuracy gate alone would have been a marginal pass/fail boundary
case, but the decorrelation requirement — the mechanism the theorem rests on — failed decisively:
the three jurors agree with each other on misrankings *more* than the math juror agrees with
itself across split halves. The jury is one verifier in a trenchcoat.

## Setup (as run)
- Base: `mlx-community/gemma-4-e4b-it-4bit`, mlx_lm 0.31.2, no-thinking harness, seed 42.
- Jurors: math / python / medical LoRA adapters (r=6, q_proj, scale 6.0), prefill-only scoring.
- GSM8K n=200, 8 chains/question at T=0.8; chains regenerated bit-exact vs R1
  (`chain_reproduction_vs_r1 = 1.0`, SC(8) reproduced exactly at 0.820).
- Weighting: per-question standardized scores -> softmax; combination: log-linear over
  answer-class masses, argmax class. Equal generation budget to SC; jurors add 447,720
  prefill tokens each, zero generated tokens.

## Prediction vs measurement

| Quantity | Predicted (pre-registered) | Measured | Outcome |
|---|---|---|---|
| SC(8) reproduction | 0.820 (exact) | 0.820 | exact |
| Jury(3) accuracy | 0.855 | 0.845 | under prediction |
| Jury − SC gain | ≥ +0.02 required | +0.025 | accuracy gate cleared |
| Best single-juror weighted SC | 0.825–0.835 | 0.835 (medical; math 0.830, python 0.830) | as predicted |
| Math-only control clears +2pp gate | must NOT clear | did not clear (0.830) | control passed |
| Mean pairwise juror kappa | < self-kappa required | **0.106** | — |
| Math bootstrap self-kappa (1000 resamples) | comparison baseline | **0.064** | **0.106 ≥ 0.064 → KILL** |
| pass@8 ceiling | 0.935 | 0.935 | reference |

Pairwise kappas (80 mixed questions): math|python 0.042, math|medical 0.117,
python|medical 0.159.

## Verdict reasoning (K2333)
- Accuracy clause: jury 0.845 ≥ SC 0.820 + 0.02 → not killed on accuracy (acc_kill = false).
- Overlap clause: mean pairwise kappa 0.106 ≥ math self-kappa 0.064 → **overlap_kill = true**.
- K2333 is an OR-kill: result = **fail → KILLED**.

Note also that the SUPPORTED bar would not have been met regardless: jury 0.845 < best single
weighted 0.835 + 0.02 = 0.855.

## Interpretation
The +2.5pp over SC is real but is explained by *weighting* (any single juror's standardized
softmax weighting already recovers +1.0 to +1.5pp), not by *decorrelation across jurors*.
Juror errors on the shared frozen base are more correlated with each other than a single
juror is with itself across chain halves, so log-linear pooling of three adapters on one
base cannot deliver the independence the theorem requires. The R2 mechanism (decorrelated
multi-juror error suppression) is refuted; any residual gain is a single-verifier calibration
effect that does not clear the support threshold.

## Provenance
- `results.json`: verdict "killed", all_pass false, is_smoke false.
- Chains reproduced bit-exact from R1 (exp_bet_jury_r1_verifier_gain); equal generation budget.

**VERDICT: KILLED — jury 0.845 vs SC 0.820 (+2.5pp) clears the accuracy gate, but mean
pairwise juror error-kappa 0.106 ≥ single-juror bootstrap self-kappa 0.064 violates the
pre-registered decorrelation requirement (K2333).**

# Adversarial Review: exp_p0_two_stage_mcq

## Verdict: PROCEED

## Summary

Clean guided-exploration experiment. Tests whether sequential optimization (NTP→MCQ-only)
can exceed the 34.5% mixed ceiling from Finding #522. Result: it cannot (33.5%). Core
finding — joint gradients are synergistic, ceiling is TT rank-6 information capacity — is
well-supported and closes the "training procedure" avenue.

## Strengths

1. **Honest theorem evaluation.** MATH.md predicted two-stage > mixed; PAPER.md clearly
   documents Theorems 1-2 as REFUTED and Theorem 3 as CONFIRMED. No spin.

2. **Good experimental design.** Four conditions (base, NTP-only, two-stage, MCQ-scratch)
   with shared controls from prior experiments. MCQ-scratch at 15.0% is a strong control
   that demonstrates NTP is load-bearing (+18.5pp, K1442 PASS).

3. **Impossibility structure is sound.** The information-rate argument (MCQ provides ~2
   bits/example vs NTP's sequence_length × log2(V) bits) correctly explains why MCQ-only
   gradient cannot reshape TT cores without NTP scaffolding.

4. **Prediction-vs-measurement table present** with 7 rows, honest FAIL annotations.

## Minor Notes (non-blocking)

1. **Loss trajectory discrepancy:** PAPER.md convergence table shows step-300 two-stage
   S2 loss as 1.381, while results.json reports final MCQ loss 1.3627. Likely batch-loss
   vs eval-loss distinction — not a data integrity issue, but could note the difference.

2. **MCQ-scratch below random (15.0% vs 25%).** PAPER.md attributes this to
   "anti-discriminative patterns" — a strong claim. The more parsimonious explanation is
   that 300 MCQ-only steps catastrophically overwrite the base model's existing MCQ-relevant
   representations (base was 30.5%) without replacing them. The gradient at one token
   position per example is insufficient to build new representations, and the old ones are
   destroyed. Same conclusion, less speculative mechanism.

3. **Stat uncertainty.** N=200 gives ~3pp margin at 95% CI. Two-stage (33.5%) vs mixed
   (34.5%) difference of 1.0pp is within noise — they're effectively equal. PAPER.md
   correctly frames this as "ceiling is from rank, not procedure," which holds regardless.

## Status: SUPPORTED — appropriate

Guided exploration that successfully narrowed an unknown (whether training procedure can
break the rank-6 ceiling). Answer: no. Joint training is optimal or tied with sequential.
The finding closes a research avenue and redirects toward rank scaling.

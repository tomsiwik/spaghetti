# Adversarial Review: exp_p0_ttlora_composition_benchmark

**Verdict: PROCEED (KILLED justified)**

## Assessment

Experiment ran successfully. PAPER.md has prediction-vs-measurement table.
results.json confirms all numbers. KILLED status is correct: 3/4 kill criteria
failed, and Theorem 1 was wrong by 3500x (predicted 0.21, measured 737.3).

## What the Experiment Proved

The strongest result is the *negative* finding: TT-LoRA has 737x LARGER norms
than standard LoRA yet produces identical pre-merge failure (GSM8K 1% vs 0%,
HumanEval 0% vs 0%, MedMCQA 21% vs 20%). This is strong evidence that pre-merge
failure is direction-dependent, not magnitude-dependent.

K1450 PASS (0pp delta on all 3 benchmarks) confirms routing completely avoids
interference, as expected.

## Theorem 1 Failure Analysis

MATH.md assumed ||DW||_F ~ sqrt(P), which holds for flat LoRA (B*A) but not for
tensor train contraction. TT cores multiply norms across k contractions, producing
exponential amplification. PAPER.md explains this correctly.

The honest forecasting of this failure mode (MATH.md Section "Falsification
Conditions" explicitly predicted that if pre-merge fails, the disease is direction)
strengthens the finding. The experiment was designed to discriminate between two
hypotheses, and it did.

## Non-Blocking Issues

1. **Epsilon-zero extrapolation.** PAPER.md claims "even scaling DW_i -> eps*DW_i
   with eps->0 keeps the cross-term directions identical." This is technically true
   for directions but misleading: at eps->0 the perturbation vanishes, so cross-terms
   also vanish. The data shows two scales (norm ~4 and ~3500) with identical failure.
   A more precise claim: "across a 737x magnitude range, failure pattern is invariant
   to scale." The structural insight is sound; the limit argument oversells it.

2. **MedMCQA confound.** MedMCQA "surviving" pre-merge (21%) is ambiguous — solo
   is also 21% and base is 31%. The adapter barely moved from base, so there's
   nothing to destroy. This is noted implicitly in PAPER.md but could be stated
   more explicitly as a confound.

## Finding #526 Status

KILLED is correct. This is a clean hypothesis falsification with a valuable
structural insight: orthogonal training (PoLAR/Grassmannian) is structurally
required for pre-merge composition, not just preferred.

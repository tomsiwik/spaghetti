# Peer Review: N-Scaling Under Uniform Composition (RE-REVIEW)

## Previous Review Fix Verification

Five fixes were required in the prior REVISE verdict. Status of each:

### Fix 1: Rename honestly -- APPLIED
PAPER.md title: "N-Scaling Evaluation Under Uniform Composition: Research Digest."
MATH.md title: "N-Scaling Under Uniform Composition: Mathematical Foundations."
Neither document frames this as continual learning. The directory name
(`continual_learning_adapter_growth`) and run_experiment.py docstring still use the old
framing, but these are infrastructure artifacts, not claims. Acceptable.

### Fix 2: Retract "zero forgetting" framing -- APPLIED
PAPER.md explicitly states: "Individual adapter invariance (0% drift) is a mathematical
tautology when parameters are frozen -- it confirms code correctness, not a hypothesis."
The "Individual Adapter Invariance" section is labeled "(Sanity Check)." The "What this
does NOT show" paragraph explicitly disclaims continual learning and forgetting immunity.
MATH.md Section header reads "Parameter Invariance Under Pool Growth (Tautological)."
This is a thorough correction.

### Fix 3: Address noise floor for K2 -- APPLIED (option b: downgrade)
PAPER.md line 70-72: "The apparent +0.09% at N=13 is within noise (single seed, no
error bars) and should NOT be interpreted as composition beating base." The trajectory
description is now "approximately flat across N=5-15, fluctuating within ~0.6pp of zero."
No "sweet spot" or "beats base" claims remain. The option to run 3 seeds was not taken,
but the honest downgrade is sufficient.

### Fix 4: Fix orthogonality metric labeling -- APPLIED
PAPER.md table headers now read "Mean B-|cos|" and "Max B-|cos|" (not "orthogonality").
Explanatory note: "These measurements are pairwise cosine similarity of flattened B-matrix
parameter vectors, NOT Grassmannian A-orthogonality." MATH.md includes a dedicated
"Metric clarification" paragraph distinguishing B-parameter cosine from A-subspace
orthogonality.

### Fix 5: Fix MATH.md interference bound -- APPLIED
The incorrect submultiplicativity chain has been replaced with a correct derivation
(MATH.md lines 59-81). The new text traces the cross-term through the A_j^T projection,
B_i^T B_j mapping, and A_i projection, then uses A_i^T A_j = 0 directly to show the
cross-term vanishes. The final sentence explicitly states: "This is a direct consequence
of A_i^T A_j = 0, not a submultiplicativity bound."

**All 5 fixes substantively applied.**

## NotebookLM Findings

Skipped. The documents are short, the math is straightforward, and the revisions are
targeted corrections to specific issues. Manual verification is sufficient.

## Mathematical Soundness

### Cross-term derivation (Fix 5 verification)

The new derivation in MATH.md goes:

```
delta_W_i^T delta_W_j = (B_i A_i^T)^T (B_j A_j^T) = A_i B_i^T B_j A_j^T
```

Then argues: A_j^T x projects into subspace j, B_i^T B_j maps within r-dim, A_i projects
into subspace i. Since A_i^T A_j = 0, the composition A_i^T (A_j z) = 0 for any z.

This is correct. One minor pedantic note: the derivation shows ||delta_W_i^T delta_W_j x|| = 0
for all x (which implies ||delta_W_i^T delta_W_j|| = 0), but does not make the for-all-x
to operator-norm step explicit. This is trivial and not worth a revision cycle.

### Dilution analysis

The 1/N dilution model is sound. At N=15, each adapter contributes s/15 = 1.33 of its
delta (with s=20). The trajectory from -0.59% at N=5 to -0.51% at N=15 is consistent
with dilution being the dominant effect, with minor fluctuations from domain coverage.

### Tautological invariance framing

The reframing is mathematically honest. Frozen A_i + frozen B_i = identical delta_W_i
at any N. The paper correctly identifies this as a code correctness check.

## Novelty Assessment

The delta over prior work (real_data_25_domain_adapters) is narrow: this experiment
evaluates the same adapters at intermediate N values (5-15) that were previously
evaluated at N=5 and N=24. The new contribution is the step-by-step trajectory
showing non-monotonic behavior. This is a minor but useful data point for the
project's evidence base.

## Experimental Design

### Remaining concern: K2 criterion is weak

K2 ("composition quality does not degrade monotonically") passes trivially because the
first step (N=5 to N=6) already shows improvement (-0.59% to -0.51%). This makes K2
essentially a check that the trajectory is not perfectly monotonically decreasing --
a very low bar. The paper acknowledges the trajectory is "approximately flat" which
is the honest interpretation.

This is a presentation issue, not a mechanism failure. The kill criterion was pre-registered
and the result is honestly reported. No revision needed.

### Remaining concern: K1 threshold at 5% is generous

At N=15 under uniform composition, observed worst-case composition degradation is 0.53%.
The 5% threshold means K1 would only fail under catastrophic interference. Given the
Grassmannian guarantee, this threshold is too loose to be informative. However, the
paper does not overclaim based on passing K1 -- the actual numbers are reported clearly.

### No new concerns identified

The revised PAPER.md and MATH.md are honest about what the experiment shows and does not
show. The limitations section is thorough (5 items including no training, single seed,
arbitrary domains, uniform-only, and N range). Claims match evidence.

## Macro-Scale Risks (advisory)

1. **Uniform 1/N is not the production path.** This experiment validates the least favorable
   composition strategy. Routing (already validated in per_token_routing experiment) would
   show better N-scaling. This is acknowledged in Limitation 2.

2. **Domain quality matters at scale.** Arbitrary dolly-15k slices produce weak "experts."
   Macro experiments should use genuinely specialized datasets. Acknowledged in Limitation 4.

3. **The approximately-flat trajectory suggests routing is mandatory, not optional.** If
   uniform composition cannot beat base even at N=13, the architecture depends entirely
   on routing quality for value delivery. This is consistent with prior findings
   (composition_weight_normalization: killed, per_token_routing: supported).

## Verdict

**PROCEED**

All 5 required fixes from the prior REVISE verdict have been substantively applied.
The experiment is honestly framed as an N-scaling evaluation, not continual learning.
Claims match evidence. Mathematical derivations are correct. Limitations are thorough.
The experiment provides a useful (if narrow) data point: uniform 1/N composition of
orthogonal adapters stays within ~1% of base across N=5-15, with no monotonic degradation.

No further revisions required.

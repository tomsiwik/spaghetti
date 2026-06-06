# LEARNINGS: exp_p5_spectral_surgery_polar

## Status: KILLED

## Core Finding
Spectral surgery (arXiv:2603.03995) is structurally incompatible with PoLAR adapters.
Three independent impossibility theorems all confirmed empirically with exact predictions.

## Why
Stiefel retraction forces all singular values to exactly 1.0 (flat spectrum, CoV = 1.5e-9).
With equal SVs: (1) surgery has nothing to differentiate, (2) the SVD basis is non-unique so
sensitivity scores are arbitrary (cosine = 0.619 between two valid bases), and (3) any
non-trivial reweighting breaks the Stiefel constraint, destroying composition guarantees.
This is a mathematical invariant, not a tuning issue.

## Implications for Next Experiment
Spectral surgery is permanently closed for PoLAR (joins Finding #278, #64, #488).
The flat-spectrum invariant is a *property to exploit*, not a defect — it guarantees
uniform contribution of all rank-r components. Next direction should target composition
scaling (null-space isolation, Fisher-Rao merging) or the SOAP adapter from Phase 2
(exp_p4_c1_vproj_soap_adapter), which addresses format conditioning structurally.

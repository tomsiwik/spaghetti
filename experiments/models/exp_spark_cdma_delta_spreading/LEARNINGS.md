# LEARNINGS — exp_spark_cdma_delta_spreading

## Core finding
A fixed seeded orthogonal rotation applied to the off-domain adapter's post-B delta-output at decode time (no retrain) removes 89% of the interference gap (code pass@1: C=0.18 → D=0.68 vs ceiling B=0.74). The null hypothesis (rotation inert, D≈C) is decisively rejected.

## Why
Rotating a coherent bias vector by a random orthogonal P scatters its energy quasi-isotropically; RMSNorm then suppresses the resulting incoherent signal proportional to sqrt(d), deflating the interference before it reaches attention. Whether this constitutes true decoherence (math adapter survives) or effective deletion (math adapter nullified) is unresolved — math accuracy under condition D was never measured.

## Implication for the next experiment
Re-run adding GSM8K pass@1 for math-solo and condition D: if math-under-D ≈ math-solo, the CDMA / multiple-access claim is confirmed (SUPPORTED); if math-under-D ≈ base, the mechanism is interference deletion and the multiple-access framing must be dropped — reframe as a decoupled interference suppressor and seek a mechanism that preserves both adapters.

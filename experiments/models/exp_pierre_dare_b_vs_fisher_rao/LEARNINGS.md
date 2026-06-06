# LEARNINGS: B-space DARE vs Fisher-Rao

## Core Finding

DARE (Drop And REscale) fails when applied only to B matrices in LoRA's A@B factorization. B-space DARE achieved 55.3% vs Fisher-Rao's 64.7% (-9.3pp), while full-delta DARE hit 71.3%.

## Why

DARE preserves expectations via random dropout + rescaling, but this guarantee holds for additive deltas. In LoRA, the effective delta is A@B — a multiplicative interaction. Dropping entries in B and rescaling doesn't preserve the expectation of A@B because A amplifies the perturbation non-linearly. Full-rank deltas (A@B pre-composed) don't have this coupling problem.

## Implication

Fisher-Rao Karcher mean remains the correct default for Pierre's shared-A architecture. Any future merge method must operate on the composed delta (A@B) or prove expectation-preservation through the multiplicative interaction. B-space is not a valid shortcut.

# LEARNINGS — exp_pierre_stiefel_b_composition

## Core Finding

Composition operators on B matrices alone cannot fix the A-matrix coupling problem.
This experiment was correctly killed: no weights exist, and the upstream proof
(cross-contribution 3.4%–501k%) makes any B-only composition operator moot.

## Why

LoRA's contribution is A@B, not B alone. Orthogonal B row spaces guarantee nothing
about the full A@B subspaces. Smarter merging (TIES, Fisher-Rao, DARE) operates on
the wrong object when A is unconstrained.

## Implication for Next Experiment

The composition problem must be solved at the A@B level, not B alone. Options:
double-Stiefel (constrain both A and B), or factored composition that reasons about
the joint A@B contribution directly.

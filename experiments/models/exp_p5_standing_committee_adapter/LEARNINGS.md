# LEARNINGS: exp_p5_standing_committee_adapter

**Status: KILLED (1/3 pass)**

## Core Finding
Module-disjoint LoRA (committee on q_proj+o_proj, domain on v_proj+down_proj) achieves
parametric orthogonality (cos=0) but produces severe functional interference: −10pp reasoning,
−60pp format, degeneration loops. Parametric orthogonality ≠ functional independence.

## Why
Attention creates obligatory multiplicative coupling: y = softmax((W_Q+ΔQ)x·(W_K·x)^T/√d)·(W_V+ΔV)x.
Even with ΔQ and ΔV in disjoint modules, the output depends on their product. Softmax argmax
sensitivity makes this first-order interference, not O(ε²). Theorem 2 was fundamentally flawed
by treating softmax as linear under small perturbations (arXiv:2601.03425 assumed linear transfer;
attention nonlinearity violates this assumption).

## Impossibility Structure
No module partition of a transformer layer isolates "reasoning" from "format" — Q×V coupling
is architectural, not accidental. Module-disjoint composition is provably insufficient for
functional independence in attention-based models.

## Implications for Next Experiment
Same-module Grassmannian isolation (Finding #49) is the correct path: both adapters on the
SAME modules with weight updates in orthogonal subspaces of the same weight matrix avoids
Q×V coupling entirely. Next: verify that A-matrix orthogonal initialization on shared q_proj
gives functional isolation across capability types.

## Confound
Base model 10% reasoning is a generation budget artifact (Gemma 4 `<|channel>thought` tokens
consume 120-token budget). Future reasoning evaluations should use max_tokens ≥ 256.
